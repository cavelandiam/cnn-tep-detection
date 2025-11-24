import os

# 📂 **Rutas del Proyecto**
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
GRAPHS_DIR = os.path.join(BASE_DIR, "graphs")
INFERENCES_DIR = os.path.join(BASE_DIR, "inferences")

# 📂 **Rutas de Datos**
HUCSR_DATASET_TEP_TRUE_DIR = os.path.join("/mnt/c/Users/velan/Documents/cnn-tep-detection/data/hucsr/positive")
HUCSR_DATASET_TEP_FALSE_DIR = os.path.join("/mnt/c/Users/velan/Documents/cnn-tep-detection/data/hucsr/negative")
HUCSR_CSV_PREPROCESSED_DATA_DIR = os.path.join(DATA_DIR, "hucsr", "preprocessed_metadata.csv")
HUCSR_PREPROCESSED_DATA_DIR = os.path.join(DATA_DIR, "hucsr", "preprocessed_data")

RSNA_DATASET_TRAIN_DIR = os.path.join("/mnt/c/Users/velan/Documents/cnn-tep-detection/data/rsna/train")
RSNA_CSV_TRAIN_DIR = os.path.join(DATA_DIR, "rsna", "train.csv")
RSNA_CSV_PREPROCESSED_DATA_TRAIN_DIR = os.path.join(DATA_DIR, "rsna", "preprocessed_metadata.csv")
RSNA_PREPROCESSED_DATA_TRAIN_DIR = os.path.join(DATA_DIR, "rsna", "preprocessed_data")

# 📂 **Rutas de Modelos**
RSNA_BEST_MODEL_AUC = os.path.join(MODEL_DIR, "rsna_best_model_auc.pth")
RSNA_PRETRAINED_MODEL = os.path.join(MODEL_DIR, "rsna_pretrained.pth")
HUCSR_FINETUNED_MODEL = os.path.join(MODEL_DIR, "hucsr_finetuned.pth")

# 📂 **Rutas de Gráficas**
HUCSR_GRAPHS_DIR = os.path.join(GRAPHS_DIR, "hucsr")
HUCSR_GRAPHS_METRICS_DIR = os.path.join(HUCSR_GRAPHS_DIR, "metrics")
HUCSR_GRAPHS_CONFUSION_MATRIX_DIR = os.path.join(HUCSR_GRAPHS_METRICS_DIR, "hucsr_confusion_matrix.png")
HUCSR_GRAPHS_MODEL_NAME = "resnet3d_finetuned"
RSNA_GRAPHS_DIR = os.path.join(GRAPHS_DIR, "rsna")
RSNA_GRAPHS_METRICS_DIR = os.path.join(RSNA_GRAPHS_DIR, "metrics")
RSNA_GRAPHS_MODEL_NAME = "resnet3d_pretrained"
RSNA_GRAPHS_CONFUSION_MATRIX_DIR = os.path.join(RSNA_GRAPHS_METRICS_DIR, "rsna_confusion_matrix.png")

# 📂 **Rutas de Inferencias**
INFERENCE_NEW_PATIENTS_DIR = os.path.join("/mnt/c/Users/velan/Documents/cnn-tep-detection/inferences/patients")
INFERENCE_RESULTS_DIR = os.path.join(INFERENCES_DIR, "results")

# 📌 **Parámetros de Preprocesamiento**
IMAGE_SIZE = (128, 128) #(224, 224)
TARGET_DEPTH = 64 #128 #512

# 📌 **Hiperparámetros del Modelo**
BATCH_SIZE = 8
LEARNING_RATE = 2e-4
NUM_WORKERS = PREFETCH_FACTOR = 4
NUM_PROCESSES = 25
EPOCHS = 50
PATIENCE_LEARNING_RATE = 5
PATIENCE_EARLY_STOPPING = PATIENCE_LEARNING_RATE * 3


# 📌 **Constantes**
X_TRAIN_TEP = 'X_tep'
Y_TRAIN_TEP = 'y_tep'
X_TRAIN_NO_TEP = 'X_no_tep'
Y_TRAIN_NO_TEP = 'y_no_tep'
