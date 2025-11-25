# test_data.py
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from pathlib import Path

# Cargar metadata
df = pl.read_csv("data/rsna/preprocessed_metadata.csv")
print(f"Total estudios: {len(df)}")

# Verificar 5 archivos .npy
for i in range(5):
    npy_path = df["preprocessed_path"][i]
    label = df["label"][i]
    
    print(f"\n{'='*60}")
    print(f"Archivo {i+1}: {Path(npy_path).name}")
    print(f"Label: {label}")
    
    # Cargar
    volume = np.load(npy_path)
    
    # Estadísticas
    print(f"Shape: {volume.shape}")
    print(f"Dtype: {volume.dtype}")
    print(f"Min: {volume.min():.6f}")
    print(f"Max: {volume.max():.6f}")
    print(f"Mean: {volume.mean():.6f}")
    print(f"Std: {volume.std():.6f}")
    print(f"Zeros: {(volume == 0).sum()} ({(volume == 0).sum() / volume.size * 100:.1f}%)")
    print(f"NaN/Inf: {np.isnan(volume).sum()}, {np.isinf(volume).sum()}")
    
    # Verificar si tiene datos reales
    if volume.max() - volume.min() < 0.01:
        print("⚠️ ADVERTENCIA: Rango de valores muy pequeño!")
    
    if (volume == 0).sum() / volume.size > 0.5:
        print("⚠️ ADVERTENCIA: Más del 50% son ceros!")
    
    # Visualizar slice central
    if i == 0:
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(volume[0, :, :, 0], cmap='gray')
        plt.title('Primer slice')
        plt.colorbar()
        
        plt.subplot(1, 3, 2)
        plt.imshow(volume[32, :, :, 0], cmap='gray')
        plt.title('Slice central')
        plt.colorbar()
        
        plt.subplot(1, 3, 3)
        plt.hist(volume.flatten(), bins=100)
        plt.title('Histograma de valores')
        plt.xlabel('Valor del pixel')
        plt.ylabel('Frecuencia')
        
        plt.tight_layout()
        plt.savefig('data_verification.png', dpi=150)
        print("\n📊 Visualización guardada en: data_verification.png")

print(f"\n{'='*60}")
print("VERIFICACIÓN COMPLETADA")
print(f"{'='*60}")