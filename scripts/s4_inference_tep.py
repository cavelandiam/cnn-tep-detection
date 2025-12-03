# INFERENCIA CLÍNICA FINAL → Compatible con estructura HUCSR (ST0)

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pydicom
from scipy.ndimage import zoom
from skimage.transform import resize
from scripts.s2_load_images_hucsr import process_dicom_image, resize_volume_depth, discover_patient_series
import time

from utils import logger, config

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# =============================================================================
# CARGAR MODELO GANADOR (Fold 3 - AUC 0.6779)
# =============================================================================
def load_best_model():
    logger.info("CARGANDO MODELO FINAL - Fold 3 - AUC 0.6779")
    
    from torchvision.models.video import R3D_18_Weights
    model = torch.hub.load('pytorch/vision', 'r3d_18', weights=R3D_18_Weights.KINETICS400_V1)
    
    # Adaptar entrada
    old = model.stem[0]
    new = nn.Conv3d(1, 64, kernel_size=old.kernel_size, stride=old.stride,
                    padding=old.padding, bias=False)
    with torch.no_grad():
        new.weight = nn.Parameter(old.weight.mean(dim=1, keepdim=True))
    model.stem[0] = new
    
    model = model.to(device)  # ← modelo base a GPU
    
    # CABEZA NUEVA Y MOVIDA A GPU
    model.fc = nn.Sequential(
        nn.Dropout(0.6),
        nn.Linear(model.fc.in_features, 1)
    ).to(device)  # ← AQUÍ ESTABA EL ERROR
    
    # Cargar pesos
    checkpoint = torch.load("/home/cavm/cnn-tep-detection/models/hucsr_finetuned.pth_fold3.pth", map_location=device)

    # 1. ¿Qué AUC tiene guardado?
    print("AUC guardado en checkpoint:", checkpoint.get('val_auc', 'NO EXISTE'))

    # 2. ¿Qué pesos tiene la última capa?
    fc_weight = checkpoint['model_state_dict']['fc.1.weight']
    fc_bias = checkpoint['model_state_dict']['fc.1.bias']
    print("fc.1.weight mean:", fc_weight.mean().item())
    print("fc.1.bias:", fc_bias.item())

    # 3. ¿Es realmente el Fold 3?
    print("Keys en checkpoint:", list(checkpoint['model_state_dict'].keys())[:10])

    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Modelo cargado → AUC: {checkpoint.get('val_auc', 0.6779):.4f}")
    
    model.eval()
    
    # Desactivar dropout en inferencia (por si acaso)
    # ¡¡¡ESTA ES LA LÍNEA QUE LO ARREGLA TODO!!!
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0  # ← fuerza dropout a 0 en inferencia
    
    return model

# =============================================================================
# PROCESAMIENTO DE PACIENTE (con ST0)
# =============================================================================
def final_evaluation(patient_dir: str):
    patient_path = Path(patient_dir)
    patient_id = patient_path.name
    logger.info(f"INFERENCIA EN PACIENTE: {patient_id}")
    
    # Crear carpeta de resultados
    output_dir = Path(config.INFERENCE_RESULTS_DIR) / patient_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model = load_best_model()
    
    # BUSCAR ST0
    st0_path = patient_path / "ST0"
    if not st0_path.exists():
        logger.error(f"NO existe {st0_path}")
        return
    
    logger.info(f"Carpeta ST0 encontrada: {st0_path}")
    
    # Recolectar TODOS los .dcm dentro de ST0 (incluyendo subcarpetas)
    #dicom_files = discover_patient_series(patient_path)
    dicom_files = list(st0_path.rglob("*"))
    if len(dicom_files) == 0:
        logger.error(f"No se encontraron archivos DICOM en {st0_path}")
        return
    
    # BUSCAR TODOS LOS ARCHIVOS (rglob funciona según tu prueba)
    all_files = list(st0_path.rglob("*"))
    dicom_files = [f for f in all_files if f.is_file()]
    
    logger.info(f"{len(dicom_files)} archivos encontrados en ST0 y subcarpetas")

    if len(dicom_files) == 0:
        logger.error("No hay archivos → algo raro")
        return
    
    # Procesar imágenes
    valid_images = []
    for dcm_file in dicom_files:
        try:
            ds = pydicom.dcmread(dcm_file, force=True, stop_before_pixels=True)
            # Solo procesar si es un DICOM real con pixel data
            full_ds = pydicom.dcmread(dcm_file, force=True)
            img = process_dicom_image(full_ds)
            if img is not None:
                valid_images.append(img)
        except:
            continue
    
    if len(valid_images) < 20:
        logger.error(f"Menos de 20 cortes válidos → no se puede procesar")
        return
    
    logger.info(f"{len(valid_images)} cortes procesados correctamente")
    
    # Tomar hasta TARGET_DEPTH
    volume = np.stack(valid_images[:config.TARGET_DEPTH])
    volume = resize_volume_depth(volume, config.TARGET_DEPTH)
    volume_tensor = torch.from_numpy(volume).permute(3, 0, 1, 2).float().to(device)

    # DIAGNÓSTICO DE ENTRADA
    logger.info(f"Volumen shape: {volume_tensor.shape}")
    logger.info(f"Volumen min: {volume_tensor.min().item():.6f}")
    logger.info(f"Volumen max: {volume_tensor.max().item():.6f}")
    logger.info(f"Volumen mean: {volume_tensor.mean().item():.6f}")
    logger.info(f"Volumen std: {volume_tensor.std().item():.6f}")

    volume = np.stack(valid_images[:config.TARGET_DEPTH])
    volume = resize_volume_depth(volume, config.TARGET_DEPTH)
    
    # TENSOR 5D CORRECTO
    volume_tensor = torch.from_numpy(volume).permute(3, 0, 1, 2).unsqueeze(0).float().to(device)
    # → shape: [1, 1, 94, 192, 192] → B, C, T, H, W

    # PREDICCIÓN
    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda', enabled=True):
            logit = model(volume_tensor)
            logit_value = logit.item()
            prob = torch.sigmoid(logit).item()
            pred = "POSITIVO" if prob > 0.5 else "NEGATIVO"
    
    logger.info(f"LOGIT CRUDO: {logit_value:.6f}")
    logger.info(f"PROBABILIDAD (sigmoid): {prob:.6f}")
    logger.info(f"RESULTADO FINAL → TEP: {pred}")
    
    # GUARDAR TODO EN inferences/111111111/
    mid_slice = volume[config.TARGET_DEPTH//2, :, :, 0]
    
    plt.figure(figsize=(10, 8))
    plt.imshow(mid_slice, cmap='gray')
    color = 'red' if pred == "POSITIVO" else 'green'
    plt.title(f"PACIENTE {patient_id}\nTEP: {pred}\nProbabilidad: {prob:.4f}", 
              fontsize=18, color=color, fontweight='bold')
    plt.axis('off')
    plt.savefig(output_dir / "resultado_clinico.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Reporte clínico
    report = f"""
        REPORTE AUTOMÁTICO DE DETECCIÓN DE TEP
        Paciente: {patient_id}
        Fecha: {time.strftime('%Y-%m-%d %H:%M')}

        RESULTADO: {pred}
        Probabilidad de TEP: {prob:.4f}
        Umbral utilizado: 0.50

        Modelo: 3D-CNN (r3d_18) entrenado en 938 series HUCSR
        Validación 5-fold: AUC medio 0.6355 | Mejor fold: 0.6779

        Este resultado es auxiliar diagnóstico.
        Requiere confirmación por radiólogo.
        """
    with open(output_dir / "reporte_clinico.txt", "w") as f:
        f.write(report)
    
    # JSON con datos estructurados
    import json
    result_json = {
        "patient_id": patient_id,
        "tep_prediction": pred,
        "tep_probability": round(prob, 4),
        "threshold": 0.5,
        "total_slices_processed": len(valid_images),
        "model_auc": 0.6779
    }
    with open(output_dir / "resultado.json", "w") as f:
        json.dump(result_json, f, indent=2)
    
    logger.info(f"INFERENCIA COMPLETADA → Resultados en: {output_dir.resolve()}")

    # DIAGNÓSTICO FINAL - AÑADE ESTO
    logger.info("=== DIAGNÓSTICO DE PESOS ===")
    for name, param in model.named_parameters():
        if 'fc' in name:
            mean_abs = param.data.abs().mean().item()
            logger.info(f"{name}: mean_abs = {mean_abs:.6f}")
            if mean_abs < 1e-6:
                logger.error(f"PESO CASI CERO → {name}")