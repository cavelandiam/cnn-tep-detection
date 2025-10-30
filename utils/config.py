import os

# 📂 **Rutas del Proyecto**
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
GRAPHS_DIR = os.path.join(BASE_DIR, "graphs")

# 📂 **Rutas de Datos**
HUCSR_DATASET_TEP_TRUE_DIR = os.path.join("/mnt/c/Users/velan/Documents/cnn-tep-detection/data/hucsr/positive")
HUCSR_DATASET_TEP_FALSE_DIR = os.path.join("/mnt/c/Users/velan/Documents/cnn-tep-detection/data/hucsr/negative")
HUCSR_CSV_PREPROCESSED_DATA_DIR = os.path.join(DATA_DIR, "hucsr", "preprocessed_metadata.csv")
HUCSR_PREPROCESSED_DATA_DIR = os.path.join(DATA_DIR, "hucsr", "preprocessed_data")
HUCSR_VISUALIZATIONS_DIR = os.path.join(DATA_DIR, "hucsr", "visualizations")
HUCSR_GRAPHS_DIR = os.path.join(GRAPHS_DIR, "hucsr")

RSNA_DATASET_TRAIN_DIR = os.path.join("/mnt/c/Users/velan/Documents/cnn-tep-detection/data/rsna/train")
RSNA_CSV_TRAIN_DIR = os.path.join(DATA_DIR, "rsna", "train.csv")
RSNA_CSV_PREPROCESSED_DATA_TRAIN_DIR = os.path.join(DATA_DIR, "rsna", "preprocessed_metadata.csv")
RSNA_PREPROCESSED_DATA_TRAIN_DIR = os.path.join(DATA_DIR, "rsna", "preprocessed_data")
RSNA_GRAPHS_DIR = os.path.join(GRAPHS_DIR, "rsna")

# 📂 **Rutas de Modelos**
RSNA_BEST_MODEL_AUC = os.path.join(MODEL_DIR, "best_model_auc.pth")
RSNA_PRETRAINED_MODEL = os.path.join(MODEL_DIR, "pretrained_rsna_final.pth")

# 📌 **Parámetros de Preprocesamiento**
IMAGE_SIZE = (128, 128) #(224, 224)
TARGET_DEPTH = 64 #128 #512

# 📌 **Hiperparámetros del Modelo**
BATCH_SIZE = 8
LEARNING_RATE = 2e-4
NUM_WORKERS = PREFETCH_FACTOR = 4
EPOCHS = 2
NUM_PROCESSES = 25

# 📌 **Constantes**
X_TRAIN_TEP = 'X_tep'
Y_TRAIN_TEP = 'y_tep'
X_TRAIN_NO_TEP = 'X_no_tep'
Y_TRAIN_NO_TEP = 'y_no_tep'
