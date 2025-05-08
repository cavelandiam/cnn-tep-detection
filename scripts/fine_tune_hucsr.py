import tensorflow as tf
import h5py
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils.config import TARGET_DEPTH, IMAGE_DICOM_RESIZE
import logging

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/fine_tune_hucsr.log'),
        logging.StreamHandler()
    ]
)

def load_data(hdf5_file):
    """Carga datos desde un archivo HDF5."""
    with h5py.File(hdf5_file, 'r') as h5f:
        X = h5f['X_train'][:]
        y = h5f['y_train'][:]
    return X, y

def main():
    """Ajusta el modelo preentrenado con HUCSR."""
    hdf5_file = 'K:/data_dicom_processed_train.h5'
    pretrained_model_path = 'models/pretrained_rsna.keras'
    
    # Cargar datos
    X, y = load_data(hdf5_file)
    
    # Cargar modelo preentrenado
    model = load_model(pretrained_model_path)
    
    # Congelar capas convolucionales iniciales
    for layer in model.layers[:-3]:
        layer.trainable = False
    
    # Compilar modelo con learning rate bajo
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=10,
        horizontal_flip=True,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1
    )
    
    # Entrenar modelo
    model.fit(
        datagen.flow(X, y, batch_size=2),
        epochs=5,
        validation_split=0.2,
        verbose=1
    )
    
    # Guardar modelo ajustado en formato .keras
    model.save('models/finetuned_hucsr.keras')
    logging.info("Modelo ajustado guardado en models/finetuned_hucsr.keras")

if __name__ == "__main__":
    main()