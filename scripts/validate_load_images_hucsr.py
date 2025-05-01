import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
from utils.config import (
    DICOM_TEP_TRUE_DIR, DICOM_TEP_FALSE_DIR, 
    IMAGE_DICOM_RESIZE, TARGET_DEPTH, 
    X_TRAIN_NO_TEP, X_TRAIN_TEP, Y_TRAIN_NO_TEP, Y_TRAIN_TEP
)

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/hdf5_validation.log'),
        logging.StreamHandler()
    ]
)

def validate():
    """Función principal para validar los archivos HDF5."""
    # Contar pacientes en los directorios originales

    #DICOM_TEP_TRUE_DIR = "D:/Trabajos Maestría/Trabajo de grado/CNN_TEP_DETECTION/data/test_code_load_images_hucsr/1/"
    #DICOM_TEP_FALSE_DIR = "D:/Trabajos Maestría/Trabajo de grado/CNN_TEP_DETECTION/data/test_code_load_images_hucsr/0/"

    n_tep_expected = count_patients(DICOM_TEP_TRUE_DIR)  # 45 para HUCSR
    n_no_tep_expected = count_patients(DICOM_TEP_FALSE_DIR)  # 45 para HUCSR
    logging.info(f"Pacientes esperados: TEP={n_tep_expected}, No-TEP={n_no_tep_expected}")

    # Validar archivo HDF5 combinado
    hdf5_file = "K:/data_dicom_processed_hucsr.h5"
    is_valid = validate_hdf5(hdf5_file, n_tep_expected, n_no_tep_expected)
    
    if is_valid:
        logging.info(f"Validación exitosa para {hdf5_file}")
    else:
        logging.error(f"Validación fallida para {hdf5_file}")

def count_patients(directory):
    """Cuenta el número de pacientes en un directorio."""
    directory = Path(directory)
    return len([p for p in directory.iterdir() if p.is_dir()])

def validate_hdf5(hdf5_file, expected_tep, expected_no_tep):
    """
    Valida el contenido de un archivo HDF5 con datos de TEP.

    Args:
        hdf5_file (str): Ruta al archivo HDF5.
        expected_tep (int): Número esperado de pacientes con TEP.
        expected_no_tep (int): Número esperado de pacientes sin TEP.
    """
    logging.info(f"Validando archivo HDF5: {hdf5_file}")
    
    try:
        with h5py.File(hdf5_file, 'r') as h5f:
            # 1. Verificar datasets
            expected_datasets = ['X_train', 'y_train']
            for dataset in expected_datasets:
                if dataset not in h5f:
                    logging.error(f"Dataset {dataset} no encontrado en {hdf5_file}")
                    return False
                logging.info(f"Dataset {dataset} encontrado")

            X_train = h5f['X_train']
            y_train = h5f['y_train']
            
            # 2. Verificar número de muestras
            n_samples = X_train.shape[0]
            if n_samples != y_train.shape[0]:
                logging.error(f"Inconsistencia: X_train ({n_samples}) y y_train ({y_train.shape[0]}) tienen diferente número de muestras")
                return False
            logging.info(f"Número total de muestras: {n_samples}")

            # 3. Verificar dimensiones de los volúmenes
            expected_shape = (TARGET_DEPTH, IMAGE_DICOM_RESIZE[0], IMAGE_DICOM_RESIZE[1], 1)
            for i in range(min(n_samples, 5)):  # Verificar solo las primeras 5 muestras
                if X_train[i].shape != expected_shape:
                    logging.error(f"Volumen {i} tiene forma incorrecta: {X_train[i].shape}, esperado: {expected_shape}")
                    return False
            logging.info(f"Forma de los volúmenes correcta: {expected_shape}")

            # 4. Verificar normalización de valores
            sample_volume = X_train[0]
            min_val, max_val = sample_volume.min(), sample_volume.max()
            if not (0 <= min_val <= max_val <= 1):
                logging.warning(f"Valores fuera del rango [0, 1]: min={min_val}, max={max_val}")
            logging.info(f"Rango de valores en volumen de muestra: min={min_val}, max={max_val}")

            # 5. Verificar etiquetas
            labels = np.array(y_train)
            unique_labels, counts = np.unique(labels, return_counts=True)
            label_counts = dict(zip(unique_labels, counts))
            logging.info(f"Distribución de etiquetas: {label_counts}")
            
            n_tep = label_counts.get(1, 0)
            n_no_tep = label_counts.get(0, 0)
            if n_tep != expected_tep or n_no_tep != expected_no_tep:
                logging.warning(f"Discrepancia en etiquetas: esperado TEP={expected_tep}, No-TEP={expected_no_tep}, "
                                f"encontrado TEP={n_tep}, No-TEP={n_no_tep}")

            # 6. Visualizar cortes de un volumen
            visualize_volume(X_train, y_train, sample_idx=0, output_dir='logs/visualizations')

        return True

    except Exception as e:
        logging.error(f"Error al validar {hdf5_file}: {e}")
        return False

def visualize_volume(X_train, y_train, sample_idx, output_dir='logs/visualizations'):
    """
    Genera visualizaciones de cortes seleccionados de un volumen.

    Args:
        X_train (h5py.Dataset): Dataset de volúmenes.
        y_train (h5py.Dataset): Dataset de etiquetas.
        sample_idx (int): Índice del volumen a visualizar.
        output_dir (str): Directorio para guardar las imágenes.
    """
    Path(output_dir).mkdir(exist_ok=True)
    volume = X_train[sample_idx]
    label = y_train[sample_idx]
    label_str = "TEP" if label == 1 else "No-TEP"

    # Seleccionar cortes representativos (inicio, medio, final)
    slices = [0, volume.shape[0] // 2, volume.shape[0] - 1]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, slice_idx in enumerate(slices):
        axes[i].imshow(volume[slice_idx, :, :, 0], cmap='gray')
        axes[i].set_title(f"Corte {slice_idx} (Z={slice_idx})")
        axes[i].axis('off')
    
    fig.suptitle(f"Volumen {sample_idx} - Etiqueta: {label_str}")
    output_path = f"{output_dir}/volume_{sample_idx}_{label_str}.png"
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    logging.info(f"Visualización guardada en {output_path}")

