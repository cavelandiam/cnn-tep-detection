import os
import numpy as np
import pydicom
from scripts.preprocess import load_dicom_image
from utils.config import DICOM_TEP_TRUE_DIR, DICOM_TEP_FALSE_DIR, RSNA_DATASET_DIR, IMAGE_DICOM_RESIZE, MESSAGES, MODEL_DIR, TARGET_DEPTH, X_TRAIN_NO_TEP, X_TRAIN_TEP, Y_TRAIN_NO_TEP, Y_TRAIN_TEP
import cv2
from pathlib import Path
import logging
from skimage.transform import resize
from concurrent.futures import ThreadPoolExecutor
import h5py
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def analizar_num_cortes(directory):
    """
    Analiza la cantidad de cortes en cada tomografía de un conjunto de datos DICOM.
    
    Args:
        directory (str): Ruta de la carpeta con estudios DICOM.
    
    Returns:
        dict: Estadísticas de cortes (máximo, mínimo, promedio, lista).
    """
    num_cortes_lista = []
    directory = Path(directory)

    for patient in directory.iterdir():
        if not patient.is_dir():
            continue
        st0_path = patient / "ST0"
        if st0_path.exists() and st0_path.is_dir():
            num_cortes = len(list(st0_path.iterdir()))
            num_cortes_lista.append(num_cortes)

    if not num_cortes_lista:
        print(f"⚠️ No se encontraron estudios válidos en {directory}")
        return {"max": 0, "min": 0, "mean": 0, "lista": []}

    stats = {
        "max": np.max(num_cortes_lista),
        "min": np.min(num_cortes_lista),
        "mean": np.mean(num_cortes_lista),
        "lista": num_cortes_lista
    }
    print(f"📊 Estadísticas de cortes: {stats}")
    return stats

def pad_or_trim_volume(volume, target_depth):
    """
    Ajusta la profundidad de un volumen 3D a `target_depth` con padding o trimming centrado.

    Args:
        volume (np.array): Volumen con forma (D, H, W, C).
        target_depth (int): Profundidad deseada.

    Returns:
        np.array: Volumen ajustado con forma (target_depth, H, W, C).
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
    
    Args:
        dicom_path (str): Ruta del archivo DICOM.
        target_size (tuple): Dimensión final (height, width).
    
    Returns:
        np.array: Imagen normalizada con forma (height, width, 1).
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

        # Normalización basada en HU
        img = dicom_data.pixel_array.astype(np.float32)

        # Verificación de imágenes a color (3 canales)
        if img.ndim == 3 and img.shape[-1] == 3:            
            logging.warning(f"⚠️ Archivo {dicom_path} ignorado (Es un resumen diagnóstico).")            

            return None

        img = np.clip(img, -1000, 1000)  # Rango típico para tejidos
        img = (img + 1000) / 2000  # Normalizar a [0, 1]

        # Redimensionar con antialiasing
        img = resize(img, target_size, anti_aliasing=True, preserve_range=True)
        return np.expand_dims(img, axis=-1)  # (H, W, 1)
    except Exception as e:
        logging.error(f"Error al procesar {dicom_path}: {e}")
        return None

def load_dataset_hucsr(directory, label, ruta_guardar_temporal):
    """
    Carga imágenes DICOM como generador y las guarda en HDF5 por batches.
    
    Args:
        directory (str): Ruta del directorio de pacientes.
        label (int): 1 para TEP, 0 para No-TEP.
        ruta_guardar_temporal (str): Ruta para guardar el archivo HDF5.
    """
    logging.info(MESSAGES["data_loading"].format(directory))
    directory = Path(directory)

    def process_patient(patient):
        st0_path = patient / "ST0"
        if not st0_path.exists():
            logging.warning(f"No se encontró ST0 en {patient}")
            return None, None
        
        # Ordenar archivos DICOM con manejo de excepciones
        dicom_files = list(st0_path.iterdir())
        sorted_files = []
        for f in dicom_files:
            try:
                ds = pydicom.dcmread(f)
                instance_num = ds.get('InstanceNumber', None)
                if instance_num is None:
                    logging.warning(f"Archivo {f} sin InstanceNumber, usando orden por nombre.")
                    instance_num = int(''.join(filter(str.isdigit, f.name))) if any(c.isdigit() for c in f.name) else float('inf')
                sorted_files.append((instance_num, f))
            except Exception as e:
                logging.error(f"Error al leer {f}: {e}")
                continue        

        # Ordenar por InstanceNumber y desempatar por nombre si es necesario
        dicom_files = [f for _, f in sorted(sorted_files, key=lambda x: (x[0], x[1].name))]

        patient_volume = [load_dicom_image(f, IMAGE_DICOM_RESIZE) for f in dicom_files]
        patient_volume = [img for img in patient_volume if img is not None]

        if not patient_volume:
            return None, None

        volume = np.array(patient_volume)
        volume = pad_or_trim_volume(volume, TARGET_DEPTH)
        return volume, label

    patients = [p for p in directory.iterdir() if p.is_dir()]

    # Asigna nombres a los datasets según el tipo de datos (TEP o no-TEP)
    if label == 1:
        tag_name_X = X_TRAIN_TEP
        tag_name_y = Y_TRAIN_TEP
    else:
        tag_name_X = X_TRAIN_NO_TEP
        tag_name_y = Y_TRAIN_NO_TEP


    with ThreadPoolExecutor(max_workers=4) as executor, h5py.File(ruta_guardar_temporal, 'w') as h5f:
        volumes_dset = h5f.create_dataset(tag_name_X, shape=(0, TARGET_DEPTH, *IMAGE_DICOM_RESIZE, 1), 
                                         maxshape=(None, TARGET_DEPTH, *IMAGE_DICOM_RESIZE, 1), 
                                         dtype=np.float32)
        labels_dset = h5f.create_dataset(tag_name_y, shape=(0,), maxshape=(None,), dtype=np.int32)

        for i, (volume, label) in enumerate(executor.map(process_patient, patients)):
            if volume is None:
                continue
            volumes_dset.resize((i + 1), axis=0)
            labels_dset.resize((i + 1), axis=0)
            volumes_dset[i] = volume
            labels_dset[i] = label

def load_all_datasets():
    """
    Carga y combina datasets de TEP y No-TEP en un archivo HDF5 final.
    
    Returns:
        tuple: Rutas a los archivos HDF5 de train y val.
    """
    analizar_num_cortes(DICOM_TEP_TRUE_DIR)
    analizar_num_cortes(DICOM_TEP_FALSE_DIR)

    tep_file = "C:/Users/velan/OneDrive/Documentos/tep_data.h5"
    no_tep_file = "C:/Users/velan/OneDrive/Documentos/no_tep_data.h5"
    load_dataset_hucsr(DICOM_TEP_TRUE_DIR, 1, tep_file)
    load_dataset_hucsr(DICOM_TEP_FALSE_DIR, 0, no_tep_file)

    # Combinar en un archivo final sin cargar todo en memoria
    output_file = "C:/Users/velan/OneDrive/Documentos/train_data.h5"
    with h5py.File(tep_file, 'r') as tep_h5, h5py.File(no_tep_file, 'r') as no_tep_h5:
        # Obtener tamaños de los datasets
        n_tep = tep_h5[X_TRAIN_TEP].shape[0]  # Número de volúmenes TEP
        n_no_tep = no_tep_h5[X_TRAIN_NO_TEP].shape[0]  # Número de volúmenes No-TEP
        total_volumes = n_tep + n_no_tep

        # Crear el archivo final con datasets vacíos pero con forma máxima definida
        with h5py.File(output_file, 'w') as final_h5f:
            # Crear datasets redimensionables
            volume_shape = (TARGET_DEPTH, *IMAGE_DICOM_RESIZE, 1)  # Forma de un volumen individual
            X_train_dset = final_h5f.create_dataset('X_train', 
                                                    shape=(0, *volume_shape), 
                                                    maxshape=(None, *volume_shape), 
                                                    dtype=np.float32)
            y_train_dset = final_h5f.create_dataset('y_train', 
                                                    shape=(0,), 
                                                    maxshape=(None,), 
                                                    dtype=np.int32)

            # Copiar datos de TEP en bloques
            for i in range(n_tep):
                X_train_dset.resize((i + 1), axis=0)
                y_train_dset.resize((i + 1), axis=0)
                X_train_dset[i] = tep_h5[X_TRAIN_TEP][i]
                y_train_dset[i] = tep_h5[Y_TRAIN_TEP][i]

            # Copiar datos de No-TEP en bloques, continuando desde el índice anterior
            offset = n_tep
            for i in range(n_no_tep):
                X_train_dset.resize((offset + i + 1), axis=0)
                y_train_dset.resize((offset + i + 1), axis=0)
                X_train_dset[offset + i] = no_tep_h5[X_TRAIN_NO_TEP][i]
                y_train_dset[offset + i] = no_tep_h5[Y_TRAIN_NO_TEP][i]

    return output_file

