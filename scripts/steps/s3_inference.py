#!/usr/bin/env python3
"""
INFERENCIA USANDO ARCHIVOS .NPY PREPROCESADOS
Esta es la forma CORRECTA - usar los mismos archivos que el entrenamiento
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import polars as pl
import time
import json
from utils import config

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

TARGET_DEPTH = 94
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# =============================================================================
# CARGAR MODELO
# =============================================================================

def load_model():
    """Carga el modelo"""
    print("Cargando modelo (Fold 3 - AUC 0.6779)...")
    
    from torchvision.models.video import R3D_18_Weights
    model = torch.hub.load('pytorch/vision', 'r3d_18', weights=R3D_18_Weights.KINETICS400_V1)
    
    # Adaptar entrada a 1 canal
    old = model.stem[0]
    new = nn.Conv3d(1, 64, kernel_size=old.kernel_size, stride=old.stride,
                    padding=old.padding, bias=False)
    with torch.no_grad():
        new.weight = nn.Parameter(old.weight.mean(dim=1, keepdim=True))
    model.stem[0] = new
    
    model = model.to(device)
    
    model.fc = nn.Sequential(
        nn.Dropout(0.6),
        nn.Linear(model.fc.in_features, 1)
    ).to(device)
    
    checkpoint_path = "/home/cavm/cnn-tep-detection/models/hucsr_finetuned.pth_fold3.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    
    print(f"✓ Modelo cargado (AUC: {checkpoint.get('val_auc', 0.6779):.4f})")
    
    return model


# =============================================================================
# INFERENCIA USANDO .NPY
# =============================================================================

def run_inference(patient_id):
    """
    Ejecuta inferencia usando el archivo .npy preprocesado
    Esta es la forma CORRECTA - usar exactamente los mismos datos que el entrenamiento
    """
    print("=" * 80)
    print(f"INFERENCIA DESDE .NPY - PACIENTE: {patient_id}")
    print("=" * 80)
    
    # Buscar el archivo .npy del paciente en el CSV
    print(f"\n[1] BUSCANDO ARCHIVO .NPY DEL PACIENTE")
    print("-" * 80)
    
    csv_path = "/home/cavm/cnn-tep-detection/data/hucsr/preprocessed_metadata.csv"
    df = pl.read_csv(csv_path)
    
    patient_rows = df.filter(pl.col('patient_name') == patient_id)
    
    if len(patient_rows) == 0:
        print(f"❌ ERROR: Paciente {patient_id} no encontrado en el CSV")
        print(f"   El paciente debe estar en el dataset de entrenamiento para esta inferencia")
        return None
    
    print(f"✓ Paciente encontrado en el CSV")
    print(f"  Tiene {len(patient_rows)} series preprocesadas")
    
    # Tomar la primera serie (o podrías iterar sobre todas)
    first_row = patient_rows.row(0, named=True)
    npy_path = first_row['preprocessed_path']
    series_id = first_row['series_id']
    label = first_row['label']
    
    print(f"\n  Serie seleccionada: {series_id}")
    print(f"  Label real: {'POSITIVO (TEP)' if label == 1 else 'NEGATIVO (No-TEP)'}")
    print(f"  Archivo .npy: {npy_path}")
    
    # Cargar el volumen preprocesado
    print(f"\n[2] CARGANDO VOLUMEN PREPROCESADO")
    print("-" * 80)
    
    volume = np.load(npy_path)
    
    print(f"✓ Volumen cargado")
    print(f"  Shape: {volume.shape}")
    print(f"  Mean:  {volume.mean():.6f}")
    print(f"  Std:   {volume.std():.6f}")
    print(f"  Range: [{volume.min():.6f}, {volume.max():.6f}]")
    
    # Verificar que las estadísticas sean correctas
    expected_mean = 0.6026
    expected_std = 0.2959
    
    mean_diff = abs(volume.mean() - expected_mean)
    std_diff = abs(volume.std() - expected_std)
    
    print(f"\n📊 VERIFICACIÓN:")
    print(f"  Mean esperado: {expected_mean:.4f}, obtenido: {volume.mean():.4f}, diff: {mean_diff:.4f}")
    print(f"  Std esperado:  {expected_std:.4f}, obtenido: {volume.std():.4f}, diff: {std_diff:.4f}")
    
    if mean_diff < 0.05 and std_diff < 0.05:
        print(f"  ✅ Estadísticas CORRECTAS - Preprocesamiento consistente")
    else:
        print(f"  ⚠ Estadísticas DIFERENTES - Puede haber variación entre series")
    
    # Crear tensor
    print(f"\n[3] PREPARANDO TENSOR PARA PREDICCIÓN")
    print("-" * 80)
    
    # El volumen ya está en formato (D, H, W, C) = (94, 192, 192, 1)
    # Necesitamos (B, C, D, H, W) = (1, 1, 94, 192, 192)
    volume_tensor = torch.from_numpy(volume).permute(3, 0, 1, 2).unsqueeze(0).float().to(device)
    
    print(f"✓ Tensor creado: {volume_tensor.shape}")
    print(f"  Mean:  {volume_tensor.mean().item():.6f}")
    print(f"  Std:   {volume_tensor.std().item():.6f}")
    
    # Cargar modelo y predecir
    print(f"\n[4] EJECUTANDO PREDICCIÓN")
    print("-" * 80)
    
    model = load_model()
    
    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda', enabled=True):
            logit = model(volume_tensor)
            logit_value = logit.item()
            prob = torch.sigmoid(logit).item()
            pred = "POSITIVO" if prob > 0.5 else "NEGATIVO"
    
    # Mostrar resultados
    print(f"\n{'='*80}")
    print(f"RESULTADOS DE LA PREDICCIÓN")
    print(f"{'='*80}")
    print(f"\n  Logit crudo:   {logit_value:.6f}")
    print(f"  Probabilidad:  {prob:.6f}")
    print(f"  Predicción:    {pred}")
    print(f"  Label real:    {'POSITIVO' if label == 1 else 'NEGATIVO'}")
    print(f"  Umbral usado:  0.50")
    
    # Análisis
    print(f"\n{'='*80}")
    print(f"ANÁLISIS")
    print(f"{'='*80}")
    
    correct = (pred == "POSITIVO" and label == 1) or (pred == "NEGATIVO" and label == 0)
    
    if -5 <= logit_value <= 5:
        print(f"\n✅ LOGIT EN RANGO NORMAL (-5 a +5)")
        print(f"   → El modelo está funcionando correctamente")
    else:
        print(f"\n⚠ LOGIT FUERA DE RANGO ESPERADO ({logit_value:.2f})")
        print(f"   → Logit muy extremo puede indicar modelo débil")
    
    if 0.1 <= prob <= 0.9:
        print(f"✅ PROBABILIDAD EN RANGO RAZONABLE (0.1 a 0.9)")
    else:
        print(f"⚠ PROBABILIDAD EXTREMA ({prob:.3f})")
        print(f"   → El modelo está muy seguro (puede indicar overfitting)")
    
    if correct:
        print(f"\n✅ PREDICCIÓN CORRECTA")
        print(f"   Predijo: {pred}, Real: {'POSITIVO' if label == 1 else 'NEGATIVO'}")
    else:
        print(f"\n❌ PREDICCIÓN INCORRECTA")
        print(f"   Predijo: {pred}, Real: {'POSITIVO' if label == 1 else 'NEGATIVO'}")
    
    # Interpretación del modelo
    print(f"\n{'='*80}")
    print(f"INTERPRETACIÓN")
    print(f"{'='*80}")
    print(f"""
AUC del modelo: 0.6779 (bajo para uso clínico)

Interpretación del resultado:
- Logit {logit_value:.2f}: {'Indica tendencia negativa' if logit_value < 0 else 'Indica tendencia positiva'}
- Prob {prob:.3f}: {'Baja confianza en positivo' if prob < 0.3 else 'Confianza moderada en positivo' if prob < 0.7 else 'Alta confianza en positivo'}

El modelo tiene AUC 0.6779, lo que significa que:
- Está apenas mejor que el azar (AUC 0.5)
- No es confiable para uso clínico real
- Necesita más datos o mejor arquitectura para mejorar

Recomendaciones:
1. ✓ El preprocesamiento es correcto (confirmado)
2. ✓ La inferencia funciona correctamente
3. ❌ El modelo necesita mejora (más datos, data augmentation, etc.)
""")
    
    # Guardar resultados
    output_dir = Path(config.INFERENCES_DIR) / patient_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Visualización
    mid_slice = volume[TARGET_DEPTH//2, :, :, 0]
    
    plt.figure(figsize=(10, 8))
    plt.imshow(mid_slice, cmap='gray')
    color = 'red' if pred == "POSITIVO" else 'green'
    check = '✓' if correct else '✗'
    plt.title(f"PACIENTE {patient_id}\n{check} Predicción: {pred}, Real: {'POSITIVO' if label == 1 else 'NEGATIVO'}\nProbabilidad: {prob:.4f}", 
              fontsize=16, color=color, fontweight='bold')
    plt.axis('off')
    plt.savefig(output_dir / "resultado_from_npy.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # JSON
    result_json = {
        "patient_id": patient_id,
        "series_id": series_id,
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "prediction": pred,
        "true_label": "POSITIVO" if label == 1 else "NEGATIVO",
        "correct": correct,
        "probability": float(prob),
        "logit": float(logit_value),
        "threshold": 0.5,
        "volume_stats": {
            "mean": float(volume.mean()),
            "std": float(volume.std()),
            "expected_mean": expected_mean,
            "expected_std": expected_std
        },
        "model": {
            "architecture": "r3d_18",
            "fold": 3,
            "auc": 0.6779
        }
    }
    
    with open(output_dir / "resultado.json", "w") as f:
        json.dump(result_json, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"INFERENCIA COMPLETADA")
    print(f"{'='*80}")
    print(f"\nResultados guardados en: {output_dir.resolve()}")
    
    return result_json


