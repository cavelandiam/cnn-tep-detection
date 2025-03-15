from tensorflow.keras.models import load_model
from scripts.dataset_loader import load_dataset
from utils.config import TRAINED_MODEL_PATH, DICOM_TEP_FALSE_DIR

# Cargar modelo en formato `.keras`
model = load_model(TRAINED_MODEL_PATH)

# Evaluar en validación
X_val, y_val = load_dataset(DICOM_TEP_FALSE_DIR, 0)
val_loss, val_acc = model.evaluate(X_val, y_val)
print(f"Precisión en validación: {val_acc:.2f}")
