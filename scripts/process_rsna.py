import os
import h5py
import pydicom
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.transform import resize
from utils.config import MODEL_DIR, RSNA_DATASET_TRAIN_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

def load_rsna_metadata(csv_path, needed_cols=None):
    """
    Lee el CSV principal que contiene los metadatos de la competencia de RSNA.
    Filtra las columnas relevantes (StudyInstanceUID, SOPInstanceUID, etc.).
    
    :param csv_path: Ruta al archivo CSV (ej: train.csv).
    :param needed_cols: Lista de columnas que necesitas conservar.
    :return: DataFrame con las columnas filtradas.
    """
    if needed_cols is None:
        needed_cols = [
            'StudyInstanceUID',
            'SeriesInstanceUID',
            'SOPInstanceUID',
            'InstanceNumber',
            'pe_present_on_image'
        ]
    logging.info(f"Leyendo CSV de metadatos: {csv_path}")
    df = pd.read_csv(csv_path)
    df = df[needed_cols]
    return df

def load_dicom_image(dicom_path, skip_color_images=True):
    """
    Carga un archivo DICOM desde `dicom_path` y retorna un numpy array float32 en rango [0,1].
    - skip_color_images: Si es True y la imagen es a color, la ignora o lanza excepción (según tu criterio).
    """
    dicom_data = pydicom.dcmread(dicom_path)
    
    # Verificar si es imagen a color (ej: PhotometricInterpretation = 'RGB')
    if skip_color_images and hasattr(dicom_data, 'PhotometricInterpretation'):
        if 'RGB' in dicom_data.PhotometricInterpretation:
            raise ValueError("Imagen DICOM a color detectada. Se ignora según configuración.")
    
    # Extraer pixel_array
    img = dicom_data.pixel_array.astype(np.float32)
    
    # Normalizar a [0,1]
    min_val, max_val = np.min(img), np.max(img)
    if max_val - min_val < 1e-8:
        # Evita divisiones por cero si la imagen está en blanco o negro total
        img = np.zeros_like(img, dtype=np.float32)
    else:
        img = (img - min_val) / (max_val - min_val)
    
    return img


def resize_image(img, target_size=(512, 512)):
    """
    Redimensiona la imagen 2D a la resolución deseada.
    Ajusta el modo de interpolación si lo requieres.
    """
    resized = resize(img, output_shape=target_size, mode='constant', preserve_range=True)
    return resized.astype(np.float32)


def process_study(study_id, df_study, base_dir, target_size=(512, 512)):
    """
    Procesa un único estudio agrupado por StudyInstanceUID:
    1. Ordena los cortes por InstanceNumber (o la columna que tengas).
    2. Carga y redimensiona cada DICOM, formando un volumen 3D.
    3. Asigna etiqueta a nivel de estudio (1 si hay cortes con PE, de lo contrario 0).
    
    :param study_id: Identificador del estudio (StudyInstanceUID).
    :param df_study: Sub-DataFrame que contiene todas las filas (cortes) de ese estudio.
    :param base_dir: Carpeta raíz con subcarpetas de SeriesInstanceUID, etc.
    :param target_size: Dimensiones a las que se redimensionará cada corte (H, W).
    :return: (study_id, volume_3d, label)
    """
    # Ordenar por InstanceNumber
    df_study = df_study.sort_values(by='InstanceNumber')
    
    slice_paths = []
    for _, row in df_study.iterrows():
        sop = row['SOPInstanceUID']
        series = row['SeriesInstanceUID']
        dicom_path = os.path.join(base_dir, series, f"{sop}.dcm")
        slice_paths.append(dicom_path)
    
    slices_3d = []
    for path in slice_paths:
        # Carga el corte (2D)
        try:
            img = load_dicom_image(path, skip_color_images=True)
            img = resize_image(img, target_size)
            slices_3d.append(img)
        except Exception as e:
            logging.warning(f"Omitiendo corte {path} por error: {e}")
            continue
    
    if len(slices_3d) == 0:
        raise ValueError(f"Estudio {study_id} no tiene cortes válidos o todos fallaron al cargar.")
    
    # Volumen 3D con ejes (num_slices, H, W)
    volume_3d = np.stack(slices_3d, axis=0)
    
    # Etiqueta a nivel de estudio:
    has_pe = df_study['pe_present_on_image'].any()
    label = 1 if has_pe else 0
    
    return study_id, volume_3d, label


def store_study_in_hdf5(hdf5_group, study_id, volume_3d, label):
    """
    Crea un subgrupo dentro del HDF5 para almacenar un volumen 3D y su etiqueta.
    Usa compresión GZIP para ahorrar espacio.
    
    :param hdf5_group: Grupo principal en el archivo HDF5 (ej: "studies").
    :param study_id: Identificador único del estudio.
    :param volume_3d: Numpy array con el volumen (num_slices, H, W).
    :param label: Etiqueta (0 o 1).
    """
    grp = hdf5_group.create_group(str(study_id))
    grp.create_dataset("volume", data=volume_3d, compression="gzip")
    grp.attrs["label"] = label


def load_rsna_dataset_to_hdf5(
    train_csv_path,
    base_dicom_dir,
    output_hdf5_path="rsna_train_data.h5",
    target_size=(512, 512),
    max_studies=None,
    parallel=False,
    num_workers=4
):
    """
    Función principal que:
    1. Lee el CSV de metadatos (train.csv).
    2. Agrupa por StudyInstanceUID.
    3. Procesa cada estudio -> volumen 3D.
    4. Almacena en HDF5 con compresión.
    
    :param train_csv_path: Ruta al CSV con metadatos.
    :param base_dicom_dir: Carpeta base donde están las subcarpetas de SeriesInstanceUID.
    :param output_hdf5_path: Nombre del archivo HDF5 de salida.
    :param target_size: Resolución final para cada corte (ej: 512x512).
    :param max_studies: Número máximo de estudios a procesar (para debug). Si None, procesa todos.
    :param parallel: Si True, usa multiprocessing.Pool para procesar en paralelo.
    :param num_workers: Número de procesos en paralelo (si parallel=True).
    """
    logging.info("[1/4] Cargando metadatos del CSV...")
    df = load_rsna_metadata(train_csv_path)
    
    logging.info("[2/4] Agrupando por StudyInstanceUID...")
    grouped = df.groupby('StudyInstanceUID')
    study_ids = list(grouped.groups.keys())
    logging.info(f"Total de estudios en CSV: {len(study_ids)}")
    
    if max_studies is not None:
        study_ids = study_ids[:max_studies]
        logging.info(f"Procesaremos únicamente {len(study_ids)} estudios (max_studies={max_studies}).")
    
    # Prepara archivo HDF5
    logging.info(f"[3/4] Creando archivo HDF5: {output_hdf5_path}")
    hdf5_file = h5py.File(output_hdf5_path, "w")
    studies_group = hdf5_file.create_group("studies")
    
    def process_wrapper(study_id):
        # Función que se usará en paralelo si parallel=True
        df_study = grouped.get_group(study_id)
        return process_study(study_id, df_study, base_dicom_dir, target_size)
    
    # Iteración secuencial o en paralelo
    if parallel:
        logging.info(f"[INFO] Ejecutando en paralelo con {num_workers} procesos.")
        with Pool(processes=num_workers) as pool:
            results = []
            for sid in tqdm(study_ids, desc="Procesando estudios"):
                results.append(pool.apply_async(process_wrapper, (sid,)))
            
            for r in tqdm(results, desc="Guardando en HDF5"):
                try:
                    study_id, volume_3d, label = r.get()  # Recolecta resultados
                    store_study_in_hdf5(studies_group, study_id, volume_3d, label)
                except Exception as e:
                    logging.error(f"Error en estudio: {e}")
    else:
        logging.info("[INFO] Ejecutando procesamiento en modo secuencial.")
        for sid in tqdm(study_ids, desc="Procesando estudios"):
            df_study = grouped.get_group(sid)
            try:
                study_id, volume_3d, label = process_study(sid, df_study, base_dicom_dir, target_size)
                store_study_in_hdf5(studies_group, study_id, volume_3d, label)
            except Exception as e:
                logging.error(f"Error al procesar estudio {sid}: {e}")
                continue
    
    # Cerrar archivo
    hdf5_file.close()
    logging.info(f"[4/4] Proceso finalizado. Archivo guardado en: {output_hdf5_path}")


def load_data_rsna():
    """
    Ejemplo de uso directo desde CLI o punto de entrada.
    Ajusta las rutas a tu entorno.
    """
    # Ajusta estos paths y parámetros:
    train_csv_path = os.path.join(MODEL_DIR,"rsna-str-pulmonary-embolism-detection", "train.csv")
    rsna_train_data = os.path.join(RSNA_DATASET_DIR, "rsna-str-pulmonary-embolism-detection", "train")
    output_hdf5_path = "rsna_train_data.h5"
    
    # Llamada principal
    load_rsna_dataset_to_hdf5(
        train_csv_path=train_csv_path,
        base_dicom_dir=rsna_train_data,
        output_hdf5_path=output_hdf5_path,
        target_size=(512, 512),
        max_studies=None,      # O un número para test
        parallel=True,         # Si quieres usar multiprocessing
        num_workers=8          # Ajusta según tu CPU/GPU
    )