#!/usr/bin/env python3
"""
COMPARACIÓN DIRECTA: Entrenamiento vs Inferencia
Este script compara el preprocesamiento exacto usado en ambos lados
"""

import torch
import numpy as np
import pydicom
from pathlib import Path
import polars as pl

print("=" * 80)
print("COMPARACIÓN DIRECTA: PREPROCESAMIENTO ENTRENAMIENTO VS INFERENCIA")
print("=" * 80)

# =============================================================================
# CARGAR UN DICOM DEL DATASET DE ENTRENAMIENTO
# =============================================================================

print("\n[1] CARGANDO SAMPLE DEL DATASET DE ENTRENAMIENTO")
print("-" * 80)

try:
    # Cargar el CSV de HUCSR
    csv_path = "/home/cavm/cnn-tep-detection/data/hucsr/preprocessed_metadata.csv"
    df = pl.read_csv(csv_path)
    
    # Tomar el primer archivo
    first_row = df.row(0, named=True)
    train_dicom_path = first_row['preprocessed_path']
    
    print(f"✓ Archivo de entrenamiento: {train_dicom_path}")
    
    # Cargar el .npy procesado
    train_processed = np.load(train_dicom_path)
    
    print(f"\n📊 ESTADÍSTICAS DEL ENTRENAMIENTO:")
    print(f"  Shape: {train_processed.shape}")
    print(f"  Dtype: {train_processed.dtype}")
    print(f"  Range: [{train_processed.min():.6f}, {train_processed.max():.6f}]")
    print(f"  Mean: {train_processed.mean():.6f}")
    print(f"  Std: {train_processed.std():.6f}")
    
    # Buscar el DICOM original
    original_path = first_row['preprocessed_path']
    print(f"\n  DICOM original: {original_path}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    train_processed = None

# =============================================================================
# PROCESAR EL MISMO DICOM CON LA FUNCIÓN DE INFERENCIA
# =============================================================================

print("\n[2] PROCESANDO EL MISMO DICOM CON CÓDIGO DE INFERENCIA")
print("-" * 80)

if train_processed is not None:
    try:
        # Importar la función de inferencia
        from scripts.s2_load_images_hucsr import process_dicom_image
        
        # Cargar el DICOM original
        ds = pydicom.dcmread(original_path, force=True)
        
        # Procesar con función de inferencia
        inference_processed = process_dicom_image(ds)
        
        if inference_processed is not None:
            print(f"\n📊 ESTADÍSTICAS DE LA INFERENCIA:")
            print(f"  Shape: {inference_processed.shape}")
            print(f"  Dtype: {inference_processed.dtype}")
            print(f"  Range: [{inference_processed.min():.6f}, {inference_processed.max():.6f}]")
            print(f"  Mean: {inference_processed.mean():.6f}")
            print(f"  Std: {inference_processed.std():.6f}")
            
            # COMPARAR
            print("\n" + "=" * 80)
            print("🔍 COMPARACIÓN CRÍTICA")
            print("=" * 80)
            
            # Diferencia absoluta
            if train_processed.shape == inference_processed.shape:
                diff = np.abs(train_processed - inference_processed)
                max_diff = diff.max()
                mean_diff = diff.mean()
                
                print(f"\n  Diferencia máxima: {max_diff:.6f}")
                print(f"  Diferencia media: {mean_diff:.6f}")
                
                if max_diff < 0.001:
                    print("\n  ✅ IDÉNTICOS - Preprocesamiento es consistente")
                elif max_diff < 0.01:
                    print("\n  ⚠ CASI IDÉNTICOS - Diferencias mínimas (probablemente por float32 vs float64)")
                elif max_diff < 0.1:
                    print("\n  ⚠ SIMILARES - Hay diferencias pequeñas pero significativas")
                else:
                    print("\n  ❌ COMPLETAMENTE DIFERENTES - ¡ESTE ES TU PROBLEMA!")
                    print("\n  → El preprocesamiento es INCONSISTENTE entre entrenamiento e inferencia")
                    print("  → El modelo ve imágenes COMPLETAMENTE DIFERENTES en inferencia")
                    print("  → Por eso predice todo como NEGATIVO")
            else:
                print(f"\n  ❌ SHAPES DIFERENTES:")
                print(f"     Entrenamiento: {train_processed.shape}")
                print(f"     Inferencia: {inference_processed.shape}")
                
        else:
            print("✗ Error al procesar con función de inferencia")
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

# =============================================================================
# PROBAR CON DIFERENTES WINDOWINGS
# =============================================================================

print("\n[3] PROBANDO DIFERENTES WINDOWINGS EN EL MISMO DICOM")
print("-" * 80)

def apply_windowing_test(pixel_array, center, width, name):
    """Test windowing con parámetros específicos"""
    lower = center - width // 2
    upper = center + width // 2
    windowed = np.clip(pixel_array, lower, upper)
    normalized = (windowed - lower) / (upper - lower)
    
    print(f"\n  {name}:")
    print(f"    Center={center}, Width={width}")
    print(f"    → Mean: {normalized.mean():.6f}, Std: {normalized.std():.6f}")
    
    return normalized

try:
    ds = pydicom.dcmread(original_path, force=True)
    pixel_array = ds.pixel_array.astype(np.float32)
    
    if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
        slope = float(ds.RescaleSlope)
        intercept = float(ds.RescaleIntercept)
        hu_array = pixel_array * slope + intercept
        
        print(f"  Hounsfield range: [{hu_array.min():.1f}, {hu_array.max():.1f}]")
        
        # Probar windowings
        lung = apply_windowing_test(hu_array, -600, 1500, "LUNG (incorrecto)")
        vascular = apply_windowing_test(hu_array, 200, 700, "VASCULAR (correcto)")
        pe = apply_windowing_test(hu_array, 100, 700, "PE (alternativo)")
        
        print("\n  📍 COMPARACIÓN CON TU INFERENCIA ACTUAL:")
        print(f"     Tu inferencia mostró: Mean=0.078182, Std=0.156218")
        
        # Calcular distancias
        dist_lung = abs(lung.mean() - 0.078182)
        dist_vasc = abs(vascular.mean() - 0.078182)
        dist_pe = abs(pe.mean() - 0.078182)
        
        print(f"\n  🎯 DISTANCIA A TU INFERENCIA:")
        print(f"     LUNG:     {dist_lung:.6f}")
        print(f"     VASCULAR: {dist_vasc:.6f}")
        print(f"     PE:       {dist_pe:.6f}")
        
        # Identificar el más cercano
        min_dist = min(dist_lung, dist_vasc, dist_pe)
        if min_dist == dist_lung:
            print("\n  → TU INFERENCIA USA WINDOWING LUNG (incorrecto para TEP)")
        elif min_dist == dist_vasc:
            print("\n  → Tu inferencia usa windowing VASCULAR (correcto)")
        else:
            print("\n  → Tu inferencia usa windowing PE")
            
except Exception as e:
    print(f"✗ Error: {e}")

# =============================================================================
# RESUMEN FINAL
# =============================================================================

print("\n" + "=" * 80)
print("RESUMEN Y DIAGNÓSTICO")
print("=" * 80)

print("""
INTERPRETACIÓN DE RESULTADOS:

1. Si la diferencia es < 0.001:
   ✅ El preprocesamiento es consistente
   ❓ El problema es el modelo débil (AUC 0.6779)
   → Necesitas más datos o mejor arquitectura

2. Si la diferencia es > 0.1:
   ❌ El preprocesamiento es INCONSISTENTE
   ❌ ESTE ES TU PROBLEMA PRINCIPAL
   → Corrige process_dicom_image() para usar los MISMOS parámetros
   → Re-ejecuta inferencia

3. Si tu inferencia usa LUNG windowing:
   ❌ Windowing INCORRECTO para detección de TEP
   ❌ El modelo se entrenó con un windowing y estás usando otro
   → Cambia a VASCULAR windowing (center=200, width=700)

PRÓXIMOS PASOS:

1. Ejecuta este script: python diagnostico_preprocessing.py
2. Verifica los resultados de la comparación
3. Si hay inconsistencias, corrige process_dicom_image()
4. Re-entrena el modelo si es necesario (si usaste windowing incorrecto)
5. Re-ejecuta inferencia y verifica que el logit esté en rango normal (-5 a +5)
""")