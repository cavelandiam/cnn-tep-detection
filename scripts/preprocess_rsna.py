# scripts/preprocess_rsna_improved.py
"""
Script de preprocesamiento mejorado para RSNA dataset
Genera archivos .npy con:
- Nuevas dimensiones: (96, 192, 192, 1)
- Ventana Hounsfield adaptativa según scanner
- Selección inteligente de series axiales
- Mejor manejo de errores y logging

USO:
    # Procesar solo estudios nuevos
    python -m scripts.preprocess_rsna_improved
    
    # Reprocesar TODOS (sobrescribe existentes)
    python -m scripts.preprocess_rsna_improved --force
"""

import gc
import os
from pathlib import Path
from typing import Optional, List, Union
import multiprocessing as mp

import numpy as np
import polars as pl
import pydicom
from scipy.ndimage import zoom
from skimage.transform import resize
from tqdm import tqdm

from utils import logger, config

# =============================================================================
# FUNCIONES DE PREPROCESAMIENTO MEJORADO
# =============================================================================

def adaptive_windowing(img: np.ndarray, ds: pydicom.FileDataset) -> np.ndarray:
    """Ventana Hounsfield CORRECTA para PE detection"""
    
    # VALORES CORRECTOS
    window_center = 200
    window_width = 700
    
    # Ajustes según scanner (mantener)
    if hasattr(ds, 'KVP'):
        try:
            kvp = float(ds.KVP)
            if kvp < 100:
                window_width = 800
            elif kvp > 140:
                window_width = 600
        except (ValueError, TypeError):
            pass
    
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img = np.clip(img, img_min, img_max)
    img = (img - img_min) / (img_max - img_min)
    
    return img.astype(np.float32)


def is_axial_orientation(image_orientation: List[float]) -> bool:
    """Verifica si una serie DICOM tiene orientación axial"""
    if len(image_orientation) != 6:
        return False
    
    row = np.array(image_orientation[:3])
    col = np.array(image_orientation[3:])
    normal = np.cross(row, col)
    
    return abs(normal[2]) > 0.9


def select_best_series(study_path: Path) -> Optional[Path]:
    """Selecciona la mejor serie para análisis de TEP (prioriza axiales)"""
    series_dirs = [p for p in study_path.iterdir() if p.is_dir()]
    if not series_dirs:
        return None
    
    best_series = None
    max_axial_slices = 0
    
    for series_dir in series_dirs:
        dicom_files = list(series_dir.glob('*.dcm'))
        if len(dicom_files) < 10:
            continue
        
        try:
            sample_ds = pydicom.dcmread(str(dicom_files[0]), stop_before_pixels=True)
            image_orientation = sample_ds.get('ImageOrientationPatient', None)
            
            if image_orientation and is_axial_orientation(image_orientation):
                if len(dicom_files) > max_axial_slices:
                    max_axial_slices = len(dicom_files)
                    best_series = series_dir
        except Exception:
            continue
    
    # Fallback: serie con más slices
    if best_series is None:
        best_series = max(series_dirs, key=lambda p: len(list(p.glob('*.dcm'))))
    
    return best_series


def process_dicom_image_improved(ds: pydicom.FileDataset) -> Optional[np.ndarray]:
    """Procesa una imagen DICOM con preprocesamiento mejorado"""
    if not hasattr(ds, 'pixel_array') or ds.pixel_array is None:
        return None
    
    if ds.pixel_array.ndim != 2:
        return None
    
    img = ds.pixel_array.astype(np.float32)
    
    # Convertir a HU
    if 'RescaleSlope' in ds and 'RescaleIntercept' in ds:
        try:
            slope = float(ds.RescaleSlope)
            intercept = float(ds.RescaleIntercept)
            img = img * slope + intercept
        except (ValueError, TypeError):
            pass
    
    # Windowing adaptativo
    img = adaptive_windowing(img, ds)
    
    # Resize
    if img.shape != config.IMAGE_SIZE:
        try:
            img = resize(img, config.IMAGE_SIZE, anti_aliasing=True, preserve_range=True)
        except Exception:
            return None
    
    img = np.clip(img, 0.0, 1.0)
    return np.expand_dims(img, axis=-1)


def resize_volume_depth(volume: np.ndarray, target_depth: int) -> np.ndarray:
    """Redimensiona el volumen a la profundidad objetivo"""
    current_depth = volume.shape[0]
    if current_depth == target_depth:
        return volume
    
    if current_depth < 1:
        return create_zero_volume()
    
    scale = target_depth / current_depth
    try:
        resized = zoom(volume, (scale, 1, 1, 1), order=1)
        if resized.shape[0] == target_depth:
            return resized.astype(np.float32)
        else:
            return create_zero_volume()
    except Exception:
        return create_zero_volume()


def create_zero_volume() -> np.ndarray:
    """Crea volumen vacío con forma esperada"""
    return np.zeros((config.TARGET_DEPTH, *config.IMAGE_SIZE, 1), dtype=np.float32)


def parse_dicom_study_improved(study_path: Union[str, Path]) -> np.ndarray:
    """Parsea un estudio DICOM completo con selección inteligente de series"""
    try:
        study_path_obj = Path(study_path)
        
        if not study_path_obj.exists():
            return create_zero_volume()
        
        # Seleccionar mejor serie
        best_series = select_best_series(study_path_obj)
        if best_series is None:
            return create_zero_volume()
        
        # Cargar archivos DICOM
        dicom_files = list(best_series.glob('*.dcm'))
        
        # Ordenar por SliceLocation
        try:
            dicom_files_sorted = sorted(
                dicom_files,
                key=lambda f: float(pydicom.dcmread(str(f), stop_before_pixels=True).get('SliceLocation', 0))
            )
        except Exception:
            dicom_files_sorted = sorted(dicom_files)
        
        if len(dicom_files_sorted) < 10:
            return create_zero_volume()
        
        # Procesar imágenes
        valid_images = []
        for dcm_file in dicom_files_sorted:
            try:
                ds = pydicom.dcmread(str(dcm_file), force=True)
                processed_img = process_dicom_image_improved(ds)
                if processed_img is not None:
                    valid_images.append(processed_img)
                del ds
                gc.collect()
            except Exception:
                continue
        
        if len(valid_images) < 10:
            return create_zero_volume()
        
        # Crear volumen
        volume = np.stack(valid_images[:config.TARGET_DEPTH], axis=0)
        resized_volume = resize_volume_depth(volume, config.TARGET_DEPTH)
        
        if resized_volume.shape == (config.TARGET_DEPTH, *config.IMAGE_SIZE, 1):
            return resized_volume
        else:
            return create_zero_volume()
            
    except Exception:
        return create_zero_volume()


def preprocess_single_study(args):
    """Procesa un estudio individual y guarda como .npy"""
    study_id, study_path, output_dir = args
    
    npy_path = Path(output_dir) / f"{study_id}.npy"
    
    # Procesar estudio
    volume = parse_dicom_study_improved(study_path)
    
    # Guardar
    np.save(npy_path, volume)
    
    return str(npy_path)


# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

def preprocess_rsna_dataset(force_reprocess: bool = False):
    """
    Preprocesa todo el dataset RSNA
    
    Args:
        force_reprocess: Si True, reprocesa incluso si los archivos existen
    """
    # Inicializar logger
    logger.init_logger("log_preprocess_rsna_improved", metrics_file="metrics_preprocess.json")
    
    logger.info("="*80)
    logger.info("PREPROCESAMIENTO MEJORADO DE DATASET RSNA")
    logger.info("="*80)
    logger.info(f"Dimensiones objetivo: {config.IMAGE_SIZE} x {config.TARGET_DEPTH}")
    logger.info(f"Directorio de salida: {config.RSNA_PREPROCESSED_DATA_TRAIN_DIR}")
    
    # 1. Cargar CSV original
    logger.info("\n1. Cargando metadatos originales...")
    df = pl.read_csv(config.RSNA_CSV_TRAIN_DIR)
    logger.info(f"   Total de filas en CSV: {len(df)}")
    
    # 2. Agrupar por estudio
    logger.info("2. Agrupando por estudio...")
    studies_df = df.group_by("StudyInstanceUID").agg(
        pl.col("negative_exam_for_pe").first().alias("negative_exam")
    ).with_columns([
        pl.col("StudyInstanceUID").map_elements(
            lambda uid: str(Path(config.RSNA_DATASET_TRAIN_DIR) / uid), 
            return_dtype=pl.String
        ).alias("study_path"),
        (1 - pl.col("negative_exam")).alias("label")
    ]).drop("negative_exam")
    
    # 3. Verificar que estudios existen
    logger.info("3. Verificando existencia de directorios...")
    studies_df = studies_df.filter(
        pl.col("study_path").map_elements(
            lambda p: Path(p).exists() and Path(p).is_dir(), 
            return_dtype=pl.Boolean
        )
    )
    
    total_studies = len(studies_df)
    logger.info(f"   ✅ {total_studies} estudios válidos encontrados")
    
    if total_studies == 0:
        raise ValueError("No se encontraron estudios válidos")
    
    # 4. Crear directorio de salida
    output_dir = Path(config.RSNA_PREPROCESSED_DATA_TRAIN_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 5. Determinar qué estudios procesar
    if force_reprocess:
        logger.info("\n4. MODO FORCE: Reprocesando TODOS los estudios")
        studies_to_process = studies_df
    else:
        existing_npy = set(p.stem for p in output_dir.glob('*.npy'))
        studies_to_process = studies_df.filter(
            ~pl.col("StudyInstanceUID").is_in(existing_npy)
        )
        logger.info(f"\n4. Ya existen: {len(existing_npy)} archivos .npy")
        logger.info(f"   Por procesar: {len(studies_to_process)} estudios")
    
    if len(studies_to_process) == 0:
        logger.info("   ✅ Todos los estudios ya están preprocesados")
    else:
        # 6. Preprocesar en paralelo
        logger.info(f"\n5. Preprocesando {len(studies_to_process)} estudios...")
        logger.info(f"   Usando {config.NUM_PROCESSES} procesos")
        
        # Preparar argumentos
        args_list = [
            (row['StudyInstanceUID'], row['study_path'], str(output_dir))
            for row in studies_to_process.iter_rows(named=True)
        ]
        
        # Procesar con barra de progreso
        with mp.Pool(processes=config.NUM_PROCESSES) as pool:
            preprocessed_paths = list(
                tqdm(
                    pool.imap(preprocess_single_study, args_list),
                    total=len(args_list),
                    desc="Preprocesando",
                    unit="estudio"
                )
            )
        
        logger.info(f"   ✅ {len(preprocessed_paths)} archivos generados")
    
    # 7. Generar CSV de metadatos
    logger.info("\n6. Generando CSV de metadatos preprocesados...")
    
    studies_df = studies_df.with_columns(
        pl.col("StudyInstanceUID").map_elements(
            lambda uid: str(output_dir / f"{uid}.npy"),
            return_dtype=pl.String
        ).alias("preprocessed_path")
    )
    
    # Verificar que todos los .npy existen
    studies_df = studies_df.filter(
        pl.col("preprocessed_path").map_elements(
            lambda p: Path(p).exists(),
            return_dtype=pl.Boolean
        )
    )
    
    output_csv = config.RSNA_CSV_PREPROCESSED_DATA_TRAIN_DIR
    studies_df.write_csv(output_csv)
    logger.info(f"   ✅ Metadatos guardados: {output_csv}")
    
    # 8. Estadísticas finales
    logger.info("\n" + "="*80)
    logger.info("RESUMEN")
    logger.info("="*80)
    logger.info(f"Estudios procesados: {len(studies_df)}")
    logger.info(f"Positivos (TEP): {len(studies_df.filter(pl.col('label') == 1))}")
    logger.info(f"Negativos: {len(studies_df.filter(pl.col('label') == 0))}")
    logger.info(f"Dimensiones volumen: ({config.TARGET_DEPTH}, {config.IMAGE_SIZE[0]}, {config.IMAGE_SIZE[1]}, 1)")
    
    total_size_gb = sum(p.stat().st_size for p in output_dir.glob('*.npy')) / 1e9
    logger.info(f"Espacio usado: {total_size_gb:.2f} GB")
    logger.info(f"Promedio/estudio: {total_size_gb / len(studies_df):.3f} GB")
    
    logger.info("\n✅ PREPROCESAMIENTO COMPLETADO")
    
    return studies_df


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocesamiento mejorado RSNA')
    parser.add_argument('--force', action='store_true', 
                       help='Reprocesar todos los estudios')
    
    args = parser.parse_args()
    
    if args.force:
        logger.warning("⚠️ MODO FORCE: Se reprocesarán TODOS los estudios")
        response = input("¿Continuar? (y/n): ")
        if response.lower() != 'y':
            print("Cancelado")
            exit(0)
    
    try:
        preprocess_rsna_dataset(force_reprocess=args.force)
    except Exception as e:
        print(f"ERROR: {e}")
        raise