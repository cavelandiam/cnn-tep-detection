import os
import numpy as np
import pydicom
from scripts.process_rsna import load_dicom_image  # Asumo que esta función está definida
from utils.config import (
    DICOM_TEP_TRUE_DIR, DICOM_TEP_FALSE_DIR, RSNA_DATASET_DIR,
    IMAGE_DICOM_RESIZE, MESSAGES, MODEL_DIR, TARGET_DEPTH,
    X_TRAIN_NO_TEP, X_TRAIN_TEP, Y_TRAIN_NO_TEP, Y_TRAIN_TEP
)
from pathlib import Path
import logging
from skimage.transform import resize
from concurrent.futures import ThreadPoolExecutor
import h5py
from collections import defaultdict
from scipy.interpolate import RegularGridInterpolator
import shutil
import matplotlib.pyplot as plt

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_loading.log'),
        logging.StreamHandler()
    ]
)

def load_all_datasets():
    """
    Carga y combina datasets de TEP y No-TEP en un archivo HDF5 final.
    
    Returns:
        tuple: Rutas a los archivos HDF5 de train y val.
    """

    DICOM_TEP_TRUE_DIR = "D:/Trabajos Maestría/Trabajo de grado/CNN_TEP_DETECTION/data/test_code_load_images_hucsr/1/"
    DICOM_TEP_FALSE_DIR = "D:/Trabajos Maestría/Trabajo de grado/CNN_TEP_DETECTION/data/test_code_load_images_hucsr/0/"

    tep_file = "K:/data_dicom_processed_tep.h5"
    no_tep_file = "K:/data_dicom_processed_no_tep.h5"
    
    load_dataset_hucsr(DICOM_TEP_TRUE_DIR, 1, tep_file)
    load_dataset_hucsr(DICOM_TEP_FALSE_DIR, 0, no_tep_file)

    output_file = "K:/data_dicom_processed_train.h5"

    with h5py.File(tep_file, 'r') as tep_h5, h5py.File(no_tep_file, 'r') as no_tep_h5:
        n_tep = tep_h5[X_TRAIN_TEP].shape[0]
        n_no_tep = no_tep_h5[X_TRAIN_NO_TEP].shape[0]
        total_volumes = n_tep + n_no_tep

        with h5py.File(output_file, 'w') as final_h5f:
            volume_shape = (TARGET_DEPTH, *IMAGE_DICOM_RESIZE, 1)
            X_train_dset = final_h5f.create_dataset('X_train', 
                                                    shape=(0, *volume_shape), 
                                                    maxshape=(None, *volume_shape), 
                                                    dtype=np.float32)
            y_train_dset = final_h5f.create_dataset('y_train', 
                                                    shape=(0,), 
                                                    maxshape=(None,), 
                                                    dtype=np.int32)

            for i in range(n_tep):
                X_train_dset.resize((i + 1), axis=0)
                y_train_dset.resize((i + 1), axis=0)
                X_train_dset[i] = tep_h5[X_TRAIN_TEP][i]
                y_train_dset[i] = tep_h5[Y_TRAIN_TEP][i]

            offset = n_tep
            for i in range(n_no_tep):
                X_train_dset.resize((offset + i + 1), axis=0)
                y_train_dset.resize((offset + i + 1), axis=0)
                X_train_dset[offset + i] = no_tep_h5[X_TRAIN_NO_TEP][i]
                y_train_dset[offset + i] = no_tep_h5[Y_TRAIN_NO_TEP][i]

    return output_file

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
    patients = [p for p in directory.iterdir() if p.is_dir()]
    
    tag_name_X = X_TRAIN_TEP if label == 1 else X_TRAIN_NO_TEP
    tag_name_y = Y_TRAIN_TEP if label == 1 else Y_TRAIN_NO_TEP

    with ThreadPoolExecutor(max_workers=4) as executor, h5py.File(ruta_guardar_temporal, 'w') as h5f:
        volumes_dset = h5f.create_dataset(tag_name_X, shape=(0, TARGET_DEPTH, *IMAGE_DICOM_RESIZE, 1), 
                                         maxshape=(None, TARGET_DEPTH, *IMAGE_DICOM_RESIZE, 1), 
                                         dtype=np.float32)
        labels_dset = h5f.create_dataset(tag_name_y, shape=(0,), maxshape=(None,), dtype=np.int32)

        for i, patient in enumerate(patients):
            series_dict, patient_name, lbl = process_patient_series(patient)
            if not series_dict:
                continue
            
            sorted_files, series_uid, n_series = select_relevant_series(series_dict, patient_name)
            volume, lbl = process_series(sorted_files, patient_name, lbl)
            if volume is None:
                continue
            
            save_series_to_folders(series_dict, patient_name)
            
            logging.info(
                f"Paciente: {patient_name}, Series: {n_series}, Serie seleccionada: {series_uid}, "
                f"Cortes: {len(sorted_files)}, Shape: {volume.shape}"
            )
            
            volumes_dset.resize((i + 1), axis=0)
            labels_dset.resize((i + 1), axis=0)
            volumes_dset[i] = volume
            labels_dset[i] = lbl

def process_patient_series(patient):
    """
    Procesa un paciente y agrupa sus archivos DICOM por SeriesInstanceUID.

    Args:
        patient (Path): Directorio del paciente.

    Returns:
        tuple: (series_dict, patient_name, label).
    """
    st0_path = patient / "ST0"
    if not st0_path.exists():
        logging.warning(f"No se encontró ST0 en {patient}")
        return None, patient.name, None

    series_dict = defaultdict(list)
    for f in st0_path.iterdir():
        try:
            ds = pydicom.dcmread(f, force=True)
            series_uid = ds.SeriesInstanceUID
            series_dict[series_uid].append(f)
        except Exception as e:
            logging.error(f"Error al leer {f}: {e}")
            continue

    if not series_dict:
        logging.warning(f"No se encontraron series válidas en {patient}")
        return None, patient.name, None

    label = 1 if 'TTEP' in patient.name else 0
    return series_dict, patient.name, label

def select_relevant_series(series_dict, patient_name):
    """
    Selecciona la serie DICOM más relevante (con más cortes).

    Args:
        series_dict (dict): Diccionario con SeriesInstanceUID como claves y listas de archivos.
        patient_name (str): Nombre del paciente.

    Returns:
        tuple: (sorted_files, selected_series_uid, num_series).
    """
    num_series = len(series_dict)
    if num_series > 1:
        logging.info(f"Paciente {patient_name}: {num_series} series detectadas")
    
    # Seleccionar la serie con más cortes
    selected_series_uid = max(series_dict, key=lambda k: len(series_dict[k]))
    selected_files = series_dict[selected_series_uid]

    # Ordenar por InstanceNumber y posición espacial
    sorted_files = []
    for f in selected_files:
        try:
            ds = pydicom.dcmread(f)
            instance_num = ds.get('InstanceNumber', float('inf'))
            position = ds.get('ImagePositionPatient', [float('inf')] * 3)[2]  # Eje Z
            sorted_files.append((instance_num, position, f))
        except Exception as e:
            logging.error(f"Error al leer {f}: {e}")
            continue
    
    sorted_files = [f for _, _, f in sorted(sorted_files, key=lambda x: (x[0], x[1]))]
    return sorted_files, selected_series_uid, num_series

def process_series(sorted_files, patient_name, label):
    """
    Procesa una serie DICOM y genera un volumen 3D.

    Args:
        sorted_files (list): Lista de archivos DICOM ordenados.
        patient_name (str): Nombre del paciente.
        label (int): Etiqueta (1 para TEP, 0 para No-TEP).

    Returns:
        tuple: (volume, label).
    """
    patient_volume = [load_dicom_image(f, IMAGE_DICOM_RESIZE) for f in sorted_files]
    patient_volume = [img for img in patient_volume if img is not None]
    
    if not patient_volume:
        logging.warning(f"No se generó volumen para {patient_name}")
        return None, None

    volume = np.array(patient_volume)
    if volume.shape[0] != TARGET_DEPTH:
        volume = pad_or_trim_volume(volume, TARGET_DEPTH)  # O resample_volume si es necesario
        # Para resampling, descomentar:
        # volume = resample_volume(volume, volume.shape[0], TARGET_DEPTH)
    
    return volume, label

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

        img = dicom_data.pixel_array.astype(np.float32)
        if img.ndim == 3 and img.shape[-1] == 3:
            logging.warning(f"Archivo {dicom_path} ignorado (es un resumen diagnóstico).")
            return None

        img = np.clip(img, -1000, 1000)  # Rango típico para tejidos
        img = (img + 1000) / 2000  # Normalizar a [0, 1]
        img = resize(img, target_size, anti_aliasing=True, preserve_range=True)
        return np.expand_dims(img, axis=-1)  # (H, W, 1)
    except Exception as e:
        logging.error(f"Error al procesar {dicom_path}: {e}")
        return None

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

def resample_volume(volume, original_depth, target_depth):
    """
    Resamplea un volumen 3D para ajustar la profundidad a `target_depth`.

    Args:
        volume (np.array): Volumen con forma (D, H, W, C).
        original_depth (int): Profundidad original.
        target_depth (int): Profundidad deseada.

    Returns:
        np.array: Volumen resampleado con forma (target_depth, H, W, C).
    """
    D, H, W, C = volume.shape
    z_original = np.linspace(0, 1, D)
    z_new = np.linspace(0, 1, target_depth)
    interpolator = RegularGridInterpolator((z_original, np.arange(H), np.arange(W)), volume[:, :, :, 0])
    new_grid = np.array(np.meshgrid(z_new, np.arange(H), np.arange(W), indexing='ij')).T.reshape(-1, 3)
    resampled = interpolator(new_grid).reshape(target_depth, H, W)
    return np.expand_dims(resampled, axis=-1)


def save_series_to_folders(series_dict, patient_name, output_dir='D:/Trabajos Maestría/Trabajo de grado/CNN_TEP_DETECTION/data/test_code_load_images_hucsr/processed_series'):
    """
    Guarda cada serie DICOM en una carpeta separada por paciente.

    Args:
        series_dict (dict): Diccionario con SeriesInstanceUID como claves y listas de archivos.
        patient_name (str): Nombre del paciente.
        output_dir (str): Directorio base para guardar las series.
    """
    output_dir = Path(output_dir) / patient_name
    output_dir.mkdir(parents=True, exist_ok=True)

    for series_uid, files in series_dict.items():
        # Simplificar SeriesInstanceUID para el nombre de la carpeta (usar los últimos 8 caracteres)
        series_folder = output_dir / f"series_{series_uid[-8:]}"
        series_folder.mkdir(exist_ok=True)

        # Copiar archivos DICOM
        for f in files:
            shutil.copy(f, series_folder / f.name)

        # Generar una visualización del corte medio (opcional)
        try:
            middle_file = files[len(files) // 2]
            img = load_dicom_image(middle_file, IMAGE_DICOM_RESIZE)
            if img is not None:
                plt.imshow(img[:, :, 0], cmap='gray')
                plt.title(f"{patient_name} - Serie {series_uid[-8:]} - Corte medio")
                plt.axis('off')
                plt.savefig(series_folder / 'middle_slice.png', bbox_inches='tight')
                plt.close()
        except Exception as e:
            logging.error(f"Error al generar visualización para {series_folder}: {e}")

        logging.info(f"Serie {series_uid[-8:]} guardada en {series_folder} con {len(files)} cortes")