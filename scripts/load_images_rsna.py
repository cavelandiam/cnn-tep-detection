import os
import numpy as np
import pydicom
import pandas as pd
from scripts.process_rsna import load_dicom_image
from utils.config import IMAGE_DICOM_RESIZE, MESSAGES, TARGET_DEPTH, RSNA_CSV_TRAIN_DIR, RSNA_DATASET_TRAIN_DIR

from pathlib import Path
import logging
from skimage.transform import resize
from concurrent.futures import ThreadPoolExecutor
import h5py
from collections import defaultdict

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_loading_rsna.log'),
        logging.StreamHandler()
    ]
)

def load_all_datasets():
    """Carga y preprocesa el dataset RSNA."""    
    output_file = 'K:/data_dicom_processed_rsna.h5'
    RSNA_DATASET_TRAIN_DIR = "D:/Trabajos Maestría/Trabajo de grado/CNN_TEP_DETECTION/data/test_code_load_images_rsna/train";
    load_dataset_rsna(RSNA_DATASET_TRAIN_DIR, RSNA_CSV_TRAIN_DIR, output_file)

def load_dataset_rsna(directory, train_csv_path, output_file):
    """
    Carga imágenes DICOM de RSNA y las guarda en HDF5.
    """
    logging.info(MESSAGES["data_loading"].format(directory))
    directory = Path(directory)
    train_csv = pd.read_csv(train_csv_path)
    
    studies = [p for p in directory.iterdir() if p.is_dir()]
    
    with ThreadPoolExecutor(max_workers=1) as executor, h5py.File(output_file, 'w') as h5f:  # Ajusta max_workers según tu hardware
        volumes_dset = h5f.create_dataset('X_train', shape=(0, TARGET_DEPTH, *IMAGE_DICOM_RESIZE, 1), 
                                         maxshape=(None, TARGET_DEPTH, *IMAGE_DICOM_RESIZE, 1), 
                                         dtype=np.float32)
        labels_dset = h5f.create_dataset('y_train', shape=(0,), maxshape=(None,), dtype=np.int32)
        
        for i, study in enumerate(studies):
            volume, label = process_study(study, train_csv)
            if volume is None:
                continue
            
            logging.info(f"Estudio: {study.name}, Cortes: {volume.shape[0]}, Shape: {volume.shape}")
            
            volumes_dset.resize((i + 1), axis=0)
            labels_dset.resize((i + 1), axis=0)
            volumes_dset[i] = volume
            labels_dset[i] = label

def process_study(study, train_csv):
    """
    Procesa un estudio con una única serie DICOM.
    """
    study_path = Path(study)
    study_id = study_path.name
    
    # Obtener la etiqueta desde el CSV
    try:
        label = 1 - int(train_csv[train_csv['StudyInstanceUID'] == study_id]['negative_exam_for_pe'].iloc[0])
    except IndexError:
        logging.warning(f"No se encontró etiqueta para el estudio {study_id} en train.csv")
        return None, None
    
    # Obtener la única serie
    series_dirs = [p for p in study_path.iterdir() if p.is_dir()]
    if len(series_dirs) != 1:
        logging.warning(f"Se esperaba una serie, pero {study_id} tiene {len(series_dirs)}")
        return None, None
    
    series_dir = series_dirs[0]
    files = list(series_dir.iterdir())
    
    # Ordenar archivos por InstanceNumber
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
    
    # Cargar las imágenes
    patient_volume = [load_dicom_image(f, IMAGE_DICOM_RESIZE) for f in sorted_files]
    patient_volume = [img for img in patient_volume if img is not None]
    
    if not patient_volume:
        logging.warning(f"No se generó volumen para {study_id}")
        return None, None
    
    volume = np.array(patient_volume)
    
    # Ajustar a TARGET_DEPTH
    if volume.shape[0] != TARGET_DEPTH:
        volume = pad_or_trim_volume(volume, TARGET_DEPTH)
    
    return volume, label

def pad_or_trim_volume(volume, target_depth):
    """
    Ajusta la profundidad de un volumen 3D a `target_depth` con padding o trimming centrado.
    """
    D, H, W, C = volume.shape
    if D == target_depth:
        return volume
    if D < target_depth:
        pad_size = target_depth - D
        pad_before = pad_size // 2
        pad_after = pad_size - pad_before
        return np.pad(volume, ((pad_before, pad_after), (0, 0), (0, 0), (0, 0)), 
                      mode='constant', constant_values=0)
    else:
        start = (D - target_depth) // 2
        return volume[start:start + target_depth, :, :, :]

def load_dicom_image(dicom_path, target_size):
    """
    Carga y preprocesa una imagen DICOM con normalización basada en unidades Hounsfield.
    """
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
        img = np.clip(img, -1000, 1000)
        img = (img + 1000) / 2000
        img = resize(img, target_size, anti_aliasing=True, preserve_range=True)
        return np.expand_dims(img, axis=-1)
    except Exception as e:
        logging.error(f"Error al procesar {dicom_path}: {e}")
        return None