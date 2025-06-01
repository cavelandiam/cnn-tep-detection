import os
import numpy as np
import pydicom
import pandas as pd
from utils.config import IMAGE_SIZE, TARGET_DEPTH, RSNA_CSV_TRAIN_DIR, RSNA_DATASET_TRAIN_DIR
from pathlib import Path
import logging
from skimage.transform import resize
from concurrent.futures import ThreadPoolExecutor
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv3D, MaxPooling3D, GlobalAveragePooling3D, Dense, Dropout
import warnings

# Suprimir advertencias específicas de pydicom para VR UI
warnings.filterwarnings('ignore', category=UserWarning, message='Invalid value for VR UI')

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/train_rsna.log'),
        logging.StreamHandler()
    ]
)

# Entrenar el modelo
def pretrain_model():
    train_csv = pd.read_csv(RSNA_CSV_TRAIN_DIR)
    model = create_model()
    
    batch_size = 1  # Ajusta según tu RAM
    steps_per_epoch = len([p for p in Path(RSNA_DATASET_TRAIN_DIR).iterdir() if p.is_dir()]) // batch_size
    
    model.fit(
        rsna_data_generator(RSNA_DATASET_TRAIN_DIR, train_csv, batch_size=batch_size),
        steps_per_epoch=steps_per_epoch,
        epochs=1  # Ajusta según tus necesidades
    )
    
    # Guardar el modelo en formato .keras
    modelSaved = "K:/pretrained_rsna.keras"
    #modelSaved = "models/pretrained_rsna.keras"
    model.save(modelSaved)
    logging.info("Modelo preentrenado guardado en models/pretrained_rsna.keras")

# Definir la arquitectura del modelo
def create_model():    
    inputs = Input(shape=(TARGET_DEPTH, *IMAGE_SIZE, 1))
    x = Conv3D(32, (3, 3, 3), activation='relu')(inputs)
    x = MaxPooling3D((2, 2, 2))(x)  # reduce a (128, 112, 112, 32)
    
    x = Conv3D(64, (3, 3, 3), activation='relu')(x)
    x = MaxPooling3D((2, 2, 2))(x)  # reduce a (64, 56, 56, 64)
    
    x = Conv3D(128, (3, 3, 3), activation='relu')(x)
    x = MaxPooling3D((2, 2, 2))(x)  # reduce a (32, 28, 28, 128)

    x = Conv3D(256, (3, 3, 3), activation='relu')(x)
    x = MaxPooling3D((2, 2, 2))(x)  # reduce a (16, 14, 14, 256)

    x = GlobalAveragePooling3D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Generador de datos para RSNA
def rsna_data_generator(directory, train_csv, batch_size):
    studies = [p for p in Path(directory).iterdir() if p.is_dir()]
    for i in range(0, len(studies), batch_size):
        batch_studies = studies[i:i + batch_size]
        X_batch = []
        y_batch = []
        
        with ThreadPoolExecutor(max_workers=1) as executor:
            futures = [executor.submit(process_study, study, train_csv) for study in batch_studies]
            for future in futures:
                volume, label = future.result()
                if volume is not None:
                    X_batch.append(volume)
                    y_batch.append(label)
        
        if X_batch:
            yield np.array(X_batch), np.array(y_batch)

# Función para procesar un estudio completo
def process_study(study_path, train_csv):
    study_id = study_path.name
    try:
        label = 1 - int(train_csv[train_csv['StudyInstanceUID'] == study_id]['negative_exam_for_pe'].iloc[0])
    except IndexError:
        logging.warning(f"No se encontró etiqueta para el estudio {study_id}")
        return None, None

    series_dirs = [p for p in study_path.iterdir() if p.is_dir()]
    if len(series_dirs) != 1:
        logging.warning(f"Se esperaba una serie, pero {study_id} tiene {len(series_dirs)}")
        return None, None

    series_dir = series_dirs[0]
    files = list(series_dir.iterdir())
    sorted_files = []
    for f in files:
        try:
            ds = pydicom.dcmread(f, force=True)
            instance_num = ds.get('InstanceNumber', float('inf'))
            sorted_files.append((instance_num, f))
        except Exception as e:
            logging.error(f"Error al leer {f}: {e}")
            continue

    sorted_files = [f for _, f in sorted(sorted_files, key=lambda x: x[0])]
    patient_volume = [load_dicom_image(f, IMAGE_SIZE) for f in sorted_files]
    patient_volume = [img for img in patient_volume if img is not None]

    if not patient_volume:
        logging.warning(f"No se generó volumen para {study_id}")
        return None, None

    volume = np.array(patient_volume)
    if volume.shape[0] != TARGET_DEPTH:
        volume = pad_or_trim_volume(volume, TARGET_DEPTH)

    return volume, label

# Función para cargar y preprocesar una imagen DICOM
def load_dicom_image(dicom_path, target_size):
    if os.path.getsize(dicom_path) < 10 * 1024:
        logging.warning(f"Archivo {dicom_path} ignorado (muy pequeño).")
        return None
    try:
        dicom_data = pydicom.dcmread(dicom_path, force=True)
        if not hasattr(dicom_data, 'pixel_array'):
            logging.warning(f"Archivo {dicom_path} ignorado (sin datos de imagen).")
            return None
        if dicom_data.file_meta.TransferSyntaxUID.is_compressed:
            dicom_data.decompress()
        img = dicom_data.pixel_array.astype(np.float32)
        if img.ndim == 3 and img.shape[-1] == 3:
            logging.warning(f"Archivo {dicom_path} ignorado (es un resumen diagnóstico).")
            return None
        img = np.clip(img, -1000, 1000) #limita los valores de los píxeles a un rango de -1000 a 1000 Hounsfield Units (HU)
        img = (img + 1000) / 2000
        img = resize(img, target_size, anti_aliasing=True, preserve_range=True)
        return np.expand_dims(img, axis=-1)
    except Exception as e:
        logging.error(f"Error al procesar {dicom_path}: {e}")
        return None

# Función para ajustar la profundidad del volumen
def pad_or_trim_volume(volume, target_depth):
    D, H, W, C = volume.shape
    if D == target_depth:
        return volume
    if D < target_depth:
        pad_size = target_depth - D
        pad_before = pad_size // 2
        pad_after = pad_size - pad_before
        return np.pad(volume, ((pad_before, pad_after), (0, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)
    else:
        # Recorte simétrico
        trim_size = D - target_depth
        trim_before = trim_size // 2
        trim_after = trim_size - trim_before
        return volume[trim_before:D - trim_after, :, :, :]