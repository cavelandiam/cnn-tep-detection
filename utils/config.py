import os

# 📂 **Rutas del Proyecto**
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# 📂 **Rutas de Datos**
DICOM_TEP_TRUE_DIR = os.path.join(DATA_DIR, "patients_tep_true")
DICOM_TEP_FALSE_DIR = os.path.join(DATA_DIR, "patients_tep_false")
RSNA_DATASET_TRAIN_DIR = os.path.join(DATA_DIR, "rsna_dataset", "rsna-str-pulmonary-embolism-detection", "train")
RSNA_DATASET_TEST_DIR = os.path.join(DATA_DIR, "rsna_dataset", "rsna-str-pulmonary-embolism-detection", "test")
RSNA_CSV_TEST_DIR = os.path.join(DATA_DIR, "rsna_dataset", "rsna-str-pulmonary-embolism-detection", "test.csv")
RSNA_CSV_TRAIN_DIR = os.path.join(DATA_DIR, "rsna_dataset", "rsna-str-pulmonary-embolism-detection", "train.csv")

# 📂 **Rutas de Modelos**
TRAINED_MODEL_PATH = os.path.join(MODEL_DIR, "trained_model.keras")
RADIMAGENET_WEIGHTS = os.path.join(MODEL_DIR, "efficientnet_radimagenet.keras")
RSNA_PRETRAINED_MODEL = os.path.join(MODEL_DIR, "rsna_pretrained_model.keras")

# 📂 **Logs**
LOG_FILE = os.path.join(LOGS_DIR, "training.log")

# 📌 **Parámetros de Preprocesamiento**
IMAGE_DICOM_RESIZE = (512, 512)
IMAGE_SIZE = (224, 224)
TARGET_DEPTH = 512

# 📌 **Hiperparámetros del Modelo**
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 0.0001

# 📌 **Mensajes del Sistema**
MESSAGES = {
    "data_loading": "📡 Cargando datos desde {}...",
    "model_training": "🚀 Entrenando modelo con {} epochs...",
    "model_saved": "✅ Modelo guardado en {}",
    "model_loaded": "🔄 Cargando modelo desde {}",
    "prediction_success": "🎯 Predicción completada con éxito",
    "error_loading_dicom": "❌ Error al cargar archivo DICOM: {}",
}

# 📌 **Constantes**
X_TRAIN_TEP = 'X_tep'
Y_TRAIN_TEP = 'y_tep'
X_TRAIN_NO_TEP = 'X_no_tep'
Y_TRAIN_NO_TEP = 'y_no_tep'
