import numpy as np
from scripts.model import build_3d_cnn
from scripts.dataset_loader import load_dataset
from utils.config import TRAINED_MODEL_PATH, DICOM_TEP_TRUE_DIR, DICOM_TEP_FALSE_DIR, RSNA_DATASET_DIR

# Cargar datasets
X_train, y_train = load_dataset(DICOM_TEP_TRUE_DIR, 1)
X_val, y_val = load_dataset(DICOM_TEP_FALSE_DIR, 0)

# Fine-Tuning con RSNA PE
X_rsna, y_rsna = load_dataset(RSNA_DATASET_DIR, 1)
X_train = np.vstack([X_train, X_rsna])
y_train = np.hstack([y_train, y_rsna])

# Construir y entrenar modelo
model = build_3d_cnn()
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=8)

# Guardar modelo en formato `.keras`
model.save(TRAINED_MODEL_PATH)
print(f"Modelo guardado en {TRAINED_MODEL_PATH}")
