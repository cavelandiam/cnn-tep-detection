"""
INFERENCIA COMPLETA EN 1 COMANDO
1. Preprocesa paciente nuevo (DICOM → .npy)
2. Predice TEP por serie
3. Agrega a nivel paciente (max prob)
4. Genera Grad-CAM 3D en la serie más sospechosa
5. Guarda imagen explicativa + CSV de resultados
"""
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import polars as pl
from typing import List, Dict
import gc

from scripts.s1_improved_3dcnn_tep import build_resnet3d_model, device
from scripts.s2_load_images_hucsr import (
    HUCSRConfig, discover_patient_series, parse_series_to_volume, process_single_series
)
from utils import logger, config

# =============================================================================
# GRAD-CAM 3D
# =============================================================================
class GradCAM3D:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]
        def forward_hook(module, input, output):
            self.activations = output
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def __call__(self, x):
        self.model.eval()
        logit = self.model(x)
        score = logit[:, 0]
        self.model.zero_grad()
        score.backward()

        grads = self.gradients.cpu().data.numpy().squeeze()
        acts = self.activations.cpu().data.numpy().squeeze()
        
        # Asegurar que tenemos dimensiones correctas
        if grads.ndim == 4:  # (C, D, H, W)
            weights = np.mean(grads, axis=(1, 2, 3))
        elif grads.ndim == 3:  # (D, H, W) - un solo canal
            grads = grads[np.newaxis, ...]
            weights = np.mean(grads, axis=(1, 2, 3))
        else:
            raise ValueError(f"Dimensiones inesperadas en gradientes: {grads.shape}")
        
        # Generar CAM
        if acts.ndim == 4:  # (C, D, H, W)
            cam = np.zeros(acts.shape[1:], dtype=np.float32)
            for i, w in enumerate(weights):
                cam += w * acts[i]
        elif acts.ndim == 3:  # (D, H, W)
            cam = acts * weights[0] if len(weights) > 0 else acts
        else:
            raise ValueError(f"Dimensiones inesperadas en activaciones: {acts.shape}")
        
        cam = np.maximum(cam, 0)
        
        # Tomar corte medio si es 3D
        if cam.ndim == 3:
            cam = cam[cam.shape[0] // 2]  # Corte medio
        
        # Asegurar que sea 2D
        if cam.ndim != 2:
            raise ValueError(f"CAM debe ser 2D después del procesamiento, pero es {cam.ndim}D: {cam.shape}")
        
        # Redimensionar a tamaño de imagen original
        cam = cv2.resize(cam, (256, 256))
        
        # Normalizar
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)
        
        return cam

# =============================================================================
# HELPER PARA ENCONTRAR CAPA OBJETIVO
# =============================================================================
def get_target_layer(model):
    """
    Encuentra automáticamente la última capa convolucional del modelo.
    Prioriza layer3, luego layer2, luego la última conv3d encontrada.
    """
    # Intentar acceder a layer3
    if hasattr(model, 'layer3'):
        layer3 = model.layer3
        # Si layer3 es un Sequential/ModuleList
        if isinstance(layer3, (torch.nn.Sequential, torch.nn.ModuleList)):
            last_block = layer3[-1]
        else:
            # Si es un bloque directo
            last_block = layer3
        
        # Buscar conv2 o la última convolución en el bloque
        if hasattr(last_block, 'conv2'):
            return last_block.conv2
        elif hasattr(last_block, 'conv1'):
            return last_block.conv1
    
    # Fallback: buscar layer2
    if hasattr(model, 'layer2'):
        layer2 = model.layer2
        if isinstance(layer2, (torch.nn.Sequential, torch.nn.ModuleList)):
            last_block = layer2[-1]
        else:
            last_block = layer2
        
        if hasattr(last_block, 'conv2'):
            return last_block.conv2
        elif hasattr(last_block, 'conv1'):
            return last_block.conv1
    
    # Último recurso: encontrar la última Conv3d en todo el modelo
    last_conv = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv3d):
            last_conv = module
    
    if last_conv is not None:
        return last_conv
    
    raise ValueError("No se pudo encontrar una capa convolucional para Grad-CAM")

# =============================================================================
# INFERENCIA COMPLETA
# =============================================================================
def inference_tep_patient(patient_dir: str) -> Dict:
    """
    Ejecuta inferencia completa en un paciente nuevo sin diagnóstico previo.
    
    Args:
        patient_dir: Ruta a carpeta DICOM del paciente
    
    Returns:
        Dict con resultados
    """
    patient_path = Path(patient_dir)
    if not patient_path.exists():
        raise FileNotFoundError(f"No existe: {patient_dir}")

    output_path = Path(config.INFERENCE_RESULTS_DIR) / patient_path.name
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"INICIANDO INFERENCIA PARA: {patient_path.name}")
    
    # 1. CARGAR MODELO
    model_path = config.HUCSR_FINETUNED_MODEL
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
    
    model = build_resnet3d_model().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Encontrar capa objetivo automáticamente
    target_layer = get_target_layer(model)
    logger.info(f"Modelo cargado: {model_path}")
    logger.info(f"Capa objetivo para Grad-CAM: {target_layer.__class__.__name__}")
    
    # 2. PREPROCESAR PACIENTE
    logger.info("Preprocesando series DICOM...")
    patient_data = discover_patient_series(patient_path)
    if patient_data.num_series == 0:
        raise ValueError("No se encontraron series válidas")
    
    npy_paths = []
    for series in patient_data.series_list:
        result = process_single_series(series, output_path)
        if result:
            npy_paths.append(result["preprocessed_path"])
    
    logger.info(f"{len(npy_paths)} series preprocesadas")
    
    # 3. PREDICCIÓN POR SERIE
    logger.info("Prediciendo por serie...")
    series_results = []
    volumes = []
    
    with torch.no_grad():
        for path in npy_paths:
            volume = np.load(path).astype(np.float32)
            volumes.append(volume)
            x = torch.from_numpy(volume).permute(3, 0, 1, 2).unsqueeze(0).to(device)
            logit = model(x)
            prob = torch.sigmoid(logit).cpu().item()
            series_results.append({
                "path": path,
                "prob": prob,
                "pred": int(prob > 0.5)
            })
    
    # 4. AGREGACIÓN A NIVEL PACIENTE
    probs = [r["prob"] for r in series_results]
    final_prob = max(probs) if probs else 0.0
    final_pred = int(final_prob > 0.5)
    best_idx = np.argmax(probs) if probs else 0
    
    result = {
        "patient_name": patient_path.name,
        "num_series": len(npy_paths),
        "series_probs": probs,
        "final_prob": final_prob,
        "final_pred": final_pred,
        "best_series_path": npy_paths[best_idx] if npy_paths else None
    }
    
    # 5. GRAD-CAM EN MEJOR SERIE
    logger.info("Generando Grad-CAM 3D...")
    best_volume = volumes[best_idx] if volumes else np.zeros((64, 256, 256, 1))
    x = torch.from_numpy(best_volume).permute(3, 0, 1, 2).unsqueeze(0).float().to(device)
    x.requires_grad = True  # Necesario para gradientes
    
    gradcam = GradCAM3D(model, target_layer)
    heatmap = gradcam(x)
    
    # Verificar que heatmap es 2D
    logger.info(f"Heatmap shape: {heatmap.shape}, dtype: {heatmap.dtype}")
    if heatmap.ndim != 2:
        raise ValueError(f"Heatmap debe ser 2D, pero es {heatmap.ndim}D: {heatmap.shape}")
    
    mid_slice = best_volume[32, :, :, 0]  # Corte medio
    
    # Normalizar slice para visualización
    mid_slice = (mid_slice - mid_slice.min()) / (mid_slice.max() - mid_slice.min() + 1e-8)
    
    # Asegurar que mid_slice tenga el mismo tamaño que heatmap
    if mid_slice.shape != heatmap.shape:
        logger.info(f"Redimensionando mid_slice de {mid_slice.shape} a {heatmap.shape}")
        mid_slice = cv2.resize(mid_slice, (heatmap.shape[1], heatmap.shape[0]))
    
    # Convertir ambas imágenes a uint8
    mid_slice_uint8 = (mid_slice * 255).astype(np.uint8)
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    
    # Verificar dimensiones antes de aplicar colormap
    logger.info(f"mid_slice_uint8 shape: {mid_slice_uint8.shape}, heatmap_uint8 shape: {heatmap_uint8.shape}")
    
    # Crear overlay
    mid_slice_bgr = cv2.cvtColor(mid_slice_uint8, cv2.COLOR_GRAY2BGR)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    
    # Verificar dimensiones finales antes de mezclar
    logger.info(f"mid_slice_bgr shape: {mid_slice_bgr.shape}, heatmap_colored shape: {heatmap_colored.shape}")
    
    # Asegurar que ambas imágenes tengan exactamente el mismo tamaño
    if mid_slice_bgr.shape != heatmap_colored.shape:
        logger.warning(f"Ajustando tamaños: {mid_slice_bgr.shape} vs {heatmap_colored.shape}")
        heatmap_colored = cv2.resize(heatmap_colored, (mid_slice_bgr.shape[1], mid_slice_bgr.shape[0]))
    
    overlay = cv2.addWeighted(mid_slice_bgr, 0.6, heatmap_colored, 0.4, 0)
    
    # Guardar imagen
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(mid_slice, cmap='gray')
    axes[0].set_title("Corte Medio")
    axes[0].axis('off')
    
    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title("Grad-CAM 3D")
    axes[1].axis('off')
    
    axes[2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f"Predicción: {'TEP' if final_pred else 'No TEP'} ({final_prob:.1%})")
    axes[2].axis('off')
    
    plt.suptitle(f"Paciente: {patient_path.name} | {len(npy_paths)} series")
    plt.tight_layout()
    
    img_path = output_path / f"prediction_result.png"
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    result["image_path"] = str(img_path)
    logger.info(f"Imagen guardada: {img_path}")
    
    # 6. GUARDAR CSV
    df_results = pl.DataFrame([{
        "patient": result["patient_name"],
        "num_series": result["num_series"],
        "final_prob": result["final_prob"],
        "prediction": "TEP" if result["final_pred"] else "No TEP",
        "image_path": result["image_path"]
    }])
    csv_path = output_path / "result_summary.csv"
    df_results.write_csv(csv_path)
    result["csv_path"] = str(csv_path)
    logger.info(f"CSV guardado: {csv_path}")
    
    # Limpiar
    del model, x, volumes, gradcam
    gc.collect()
    torch.cuda.empty_cache()
    
    logger.info(f"INFERENCIA COMPLETA: {'TEP' if final_pred else 'No TEP'} (prob={final_prob:.1%})")
    return result

# =============================================================================
# MAIN
# =============================================================================
def run_inference_tep(patient_dir: str = None):
    
    logger.init_logger("log_inference_tep")
    
    result = inference_tep_patient(patient_dir=patient_dir)
    
    print("\n" + "="*60)
    print(f"PACIENTE: {result['patient_name']}")
    print(f"SERIES: {result['num_series']}")
    print(f"RESULTADO: {'TEP' if result['final_pred'] else 'No TEP'}")
    print(f"PROBABILIDAD: {result['final_prob']:.1%}")
    print(f"IMAGEN: {result['image_path']}")
    print(f"CSV: {result['csv_path']}")
    print("="*60)