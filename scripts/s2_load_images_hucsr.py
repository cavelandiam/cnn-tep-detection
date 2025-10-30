"""
Carga y preprocesamiento optimizado de datos HUCSR
Procesa TODAS las series de cada paciente como instancias separadas
Versión mejorada con mejores prácticas de ML/DL
"""
import gc
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import polars as pl
import pydicom
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from skimage.transform import resize
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from utils import logger, config


# =============================================================================
# CONFIGURACIÓN
# =============================================================================

class HUCSRConfig:
    """Configuración específica para HUCSR"""
    
    # Directorios
    POSITIVOS_DIR = Path(config.HUCSR_DATASET_TEP_TRUE_DIR)
    NEGATIVOS_DIR = Path(config.HUCSR_DATASET_TEP_FALSE_DIR)

    # Output
    OUTPUT_DIR = Path(config.HUCSR_PREPROCESSED_DATA_DIR)
    METADATA_CSV = Path(config.HUCSR_CSV_PREPROCESSED_DATA_DIR)
    #VISUALIZATIONS_DIR = Path(config.HUCSR_VISUALIZATIONS_DIR)
    
    # Parámetros de imagen
    IMAGE_SIZE = config.IMAGE_SIZE  # (256, 256)
    TARGET_DEPTH = config.TARGET_DEPTH  # 64
    
    # Ventana Hounsfield para angiotacs (ajustado para TEP)
    WINDOW_CENTER = -600
    WINDOW_WIDTH = 1600
    
    # Procesamiento paralelo
    MAX_WORKERS = 4
    
    # Filtros de calidad
    MIN_SLICES_PER_SERIES = 10  # Mínimo de cortes para considerar una serie
    MIN_FILE_SIZE_KB = 10  # Archivos DICOM muy pequeños se ignoran
    
    # Estrategia para múltiples series
    PROCESS_ALL_SERIES = True  # True: procesar todas, False: solo la principal


# =============================================================================
# CLASES DE DATOS
# =============================================================================

class SeriesInfo:
    """Información de una serie DICOM"""
    
    def __init__(self, series_uid: str, files: List[Path], patient_name: str, label: int):
        self.series_uid = series_uid
        self.files = files
        self.patient_name = patient_name
        self.label = label
        self.num_slices = len(files)
        self.is_valid = self.num_slices >= HUCSRConfig.MIN_SLICES_PER_SERIES
        
    def __repr__(self):
        return (f"SeriesInfo(patient={self.patient_name}, "
                f"uid={self.series_uid[-8:]}, slices={self.num_slices})")


class PatientData:
    """Datos completos de un paciente"""
    
    def __init__(self, patient_dir: Path):
        self.patient_dir = patient_dir
        self.patient_name = patient_dir.name
        self.label = 1 if 'TTEP' in self.patient_name else 0
        self.series_list: List[SeriesInfo] = []
        
    def add_series(self, series_info: SeriesInfo):
        """Agrega una serie al paciente"""
        if series_info.is_valid:
            self.series_list.append(series_info)
    
    @property
    def num_series(self) -> int:
        return len(self.series_list)
    
    @property
    def total_slices(self) -> int:
        return sum(s.num_slices for s in self.series_list)
    
    def get_main_series(self) -> Optional[SeriesInfo]:
        """Retorna la serie con más cortes"""
        if not self.series_list:
            return None
        return max(self.series_list, key=lambda s: s.num_slices)
    
    def __repr__(self):
        return (f"PatientData(name={self.patient_name}, label={self.label}, "
                f"series={self.num_series}, total_slices={self.total_slices})")


# =============================================================================
# UTILIDADES DICOM
# =============================================================================

def read_dicom_safe(dicom_path: Path) -> Optional[pydicom.FileDataset]:
    """Lee archivo DICOM con manejo de errores"""
    try:
        # Verificar tamaño mínimo
        if dicom_path.stat().st_size < HUCSRConfig.MIN_FILE_SIZE_KB * 1024:
            return None
        
        ds = pydicom.dcmread(dicom_path, force=True)
        
        # Verificar que tenga pixel_array
        if not hasattr(ds, 'pixel_array'):
            return None
        
        return ds
        
    except Exception as e:
        logger.debug(f"Error leyendo {dicom_path.name}: {e}")
        return None


def get_slice_location(ds: pydicom.FileDataset) -> float:
    """Obtiene la posición espacial del corte"""
    # Prioridad: SliceLocation > ImagePositionPatient[2] > InstanceNumber
    if hasattr(ds, 'SliceLocation') and ds.SliceLocation is not None:
        return float(ds.SliceLocation)
    
    if hasattr(ds, 'ImagePositionPatient') and ds.ImagePositionPatient is not None:
        return float(ds.ImagePositionPatient[2])
    
    if hasattr(ds, 'InstanceNumber') and ds.InstanceNumber is not None:
        return float(ds.InstanceNumber)
    
    return 0.0


def sort_dicom_files(files: List[Path]) -> List[Path]:
    """Ordena archivos DICOM por posición espacial"""
    sorted_files = []
    
    for f in files:
        ds = read_dicom_safe(f)
        if ds is None:
            continue
        
        location = get_slice_location(ds)
        instance_num = getattr(ds, 'InstanceNumber', float('inf'))
        sorted_files.append((location, instance_num, f))
        
        del ds
    
    # Ordenar por location primero, luego por instance number
    sorted_files.sort(key=lambda x: (x[0], x[1]))
    
    return [f for _, _, f in sorted_files]


# =============================================================================
# PROCESAMIENTO DE IMÁGENES
# =============================================================================

def process_dicom_image(ds: pydicom.FileDataset) -> Optional[np.ndarray]:
    """
    Procesa una imagen DICOM individual con windowing para angiotacs
    Idéntico al pipeline de RSNA para consistencia
    """
    if not hasattr(ds, 'pixel_array') or ds.pixel_array is None:
        return None
    
    if ds.pixel_array.ndim != 2:
        # Ignorar imágenes RGB (resúmenes diagnósticos)
        if ds.pixel_array.ndim == 3 and ds.pixel_array.shape[-1] == 3:
            return None
        return None
    
    img = ds.pixel_array.astype(np.float32)
    
    # Aplicar RescaleSlope y RescaleIntercept (convertir a HU)
    if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
        try:
            slope = float(ds.RescaleSlope)
            intercept = float(ds.RescaleIntercept)
            img = img * slope + intercept
        except (ValueError, TypeError):
            logger.debug(f"No se pudo aplicar rescale")
    
    # Windowing para angiotacs de tórax
    window_center = HUCSRConfig.WINDOW_CENTER
    window_width = HUCSRConfig.WINDOW_WIDTH
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    
    img = np.clip(img, img_min, img_max)
    
    # Normalizar a [0, 1]
    img = (img - img_min) / (img_max - img_min)
    
    # Resize a tamaño objetivo
    if img.shape != HUCSRConfig.IMAGE_SIZE:
        try:
            img_resized = resize(
                img,
                HUCSRConfig.IMAGE_SIZE,
                anti_aliasing=True,
                preserve_range=True
            )
            img = img_resized.astype(np.float32)
        except Exception as e:
            logger.warning(f"Error en resize: {e}")
            return None
    
    # Clip final y agregar canal
    img = np.clip(img, 0.0, 1.0)
    return np.expand_dims(img, axis=-1)


def resize_volume_depth(volume: np.ndarray, target_depth: int) -> np.ndarray:
    """
    Redimensiona volumen a profundidad objetivo usando interpolación
    Idéntico al usado en RSNA
    """
    current_depth = volume.shape[0]
    
    if current_depth < 1:
        logger.warning(f"Volumen inválido con profundidad {current_depth}")
        return create_zero_volume()
    
    if current_depth == target_depth:
        return volume
    
    scale = target_depth / current_depth
    try:
        resized = zoom(volume, (scale, 1, 1, 1), order=1)
        if resized.shape[0] == target_depth:
            return resized.astype(np.float32)
        else:
            logger.warning(f"Redimensionado falló: shape={resized.shape}")
            return create_zero_volume()
    except Exception as e:
        logger.warning(f"Error en zoom: {e}")
        return create_zero_volume()


def create_zero_volume() -> np.ndarray:
    """Crea volumen vacío con forma esperada"""
    return np.zeros(
        (HUCSRConfig.TARGET_DEPTH, *HUCSRConfig.IMAGE_SIZE, 1),
        dtype=np.float32
    )


# =============================================================================
# PROCESAMIENTO DE SERIES Y PACIENTES
# =============================================================================

def parse_series_to_volume(series_info: SeriesInfo) -> Optional[np.ndarray]:
    """
    Convierte una serie DICOM en un volumen 3D normalizado
    
    Returns:
        np.ndarray: Volumen con shape (TARGET_DEPTH, H, W, 1) o None si falla
    """
    try:
        # Ordenar archivos por posición espacial
        sorted_files = sort_dicom_files(series_info.files)
        
        if not sorted_files:
            logger.warning(f"No hay archivos válidos en {series_info}")
            return None
        
        # Procesar cada slice
        valid_images = []
        for dicom_file in sorted_files:
            ds = read_dicom_safe(dicom_file)
            if ds is None:
                continue
            
            processed_img = process_dicom_image(ds)
            if processed_img is not None:
                valid_images.append(processed_img)
            
            del ds
        
        if not valid_images:
            logger.warning(f"No se procesaron imágenes válidas para {series_info}")
            return None
        
        # Limitar a TARGET_DEPTH slices
        if len(valid_images) > HUCSRConfig.TARGET_DEPTH:
            valid_images = valid_images[:HUCSRConfig.TARGET_DEPTH]
        
        # Stack y redimensionar
        volume = np.stack(valid_images, axis=0)
        resized_volume = resize_volume_depth(volume, HUCSRConfig.TARGET_DEPTH)
        
        # Validar forma final
        expected_shape = (HUCSRConfig.TARGET_DEPTH, *HUCSRConfig.IMAGE_SIZE, 1)
        if resized_volume.shape != expected_shape:
            logger.error(f"Forma inválida: {resized_volume.shape}, esperado: {expected_shape}")
            return None
        
        logger.info(f"✓ Serie procesada: {series_info.patient_name}/{series_info.series_uid[-8:]} "
                   f"({len(valid_images)} → {HUCSRConfig.TARGET_DEPTH} slices)")
        
        return resized_volume.astype(np.float32)
        
    except Exception as e:
        logger.error(f"Error procesando serie {series_info}: {e}")
        return None


def discover_patient_series(patient_dir: Path) -> PatientData:
    """
    Descubre todas las series de un paciente
    
    Args:
        patient_dir: Directorio del paciente (ej: TTEP_001)
    
    Returns:
        PatientData con todas las series encontradas
    """
    patient_data = PatientData(patient_dir)
    
    # Buscar carpeta ST0 (estructura típica de HUCSR)
    st0_path = patient_dir / "ST0"
    if not st0_path.exists():
        logger.warning(f"No se encontró ST0 en {patient_dir}")
        return patient_data
    
    # Agrupar archivos por SeriesInstanceUID
    series_dict = defaultdict(list)
    
    for dicom_file in st0_path.iterdir():
        if not dicom_file.suffix.lower() in ['.dcm', '.dicom', '']:
            continue
        
        ds = read_dicom_safe(dicom_file)
        if ds is None:
            continue
        
        # Obtener SeriesInstanceUID
        if not hasattr(ds, 'SeriesInstanceUID'):
            logger.debug(f"Archivo sin SeriesInstanceUID: {dicom_file.name}")
            continue
        
        series_uid = ds.SeriesInstanceUID
        series_dict[series_uid].append(dicom_file)
        
        del ds
    
    # Crear SeriesInfo para cada serie encontrada
    for series_uid, files in series_dict.items():
        series_info = SeriesInfo(
            series_uid=series_uid,
            files=files,
            patient_name=patient_data.patient_name,
            label=patient_data.label
        )
        patient_data.add_series(series_info)
    
    if patient_data.num_series == 0:
        logger.warning(f"No se encontraron series válidas en {patient_dir}")
    else:
        logger.info(f"Paciente {patient_data.patient_name}: "
                   f"{patient_data.num_series} series, "
                   f"{patient_data.total_slices} slices totales")
    
    return patient_data


def process_single_series(series_info: SeriesInfo, output_dir: Path) -> Optional[Dict]:
    """
    Procesa una serie individual y guarda como .npy
    
    Returns:
        Dict con metadata de la serie procesada
    """
    # Crear nombre único para la serie
    series_id = f"{series_info.patient_name}_{series_info.series_uid[-8:]}"
    npy_path = output_dir / f"{series_id}.npy"
    
    # Skip si ya existe
    if npy_path.exists():
        logger.debug(f"Ya procesado: {series_id}")
        return {
            'series_id': series_id,
            'patient_name': series_info.patient_name,
            'series_uid': series_info.series_uid,
            'label': series_info.label,
            'num_slices_original': series_info.num_slices,
            'preprocessed_path': str(npy_path)
        }
    
    # Procesar serie a volumen
    volume = parse_series_to_volume(series_info)
    
    if volume is None:
        logger.error(f"Fallo al procesar: {series_id}")
        return None
    
    # Guardar volumen
    np.save(npy_path, volume)
    logger.info(f"💾 Guardado: {npy_path.name}")
    
    # Limpiar memoria
    del volume
    gc.collect()
    
    return {
        'series_id': series_id,
        'patient_name': series_info.patient_name,
        'series_uid': series_info.series_uid,
        'label': series_info.label,
        'num_slices_original': series_info.num_slices,
        'preprocessed_path': str(npy_path)
    }


def visualize_series(series_info: SeriesInfo, volume: np.ndarray, output_dir: Path):
    """
    Genera visualización del corte medio de una serie
    """
    try:
        vis_dir = output_dir / series_info.patient_name
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        # Corte medio
        mid_slice = volume[HUCSRConfig.TARGET_DEPTH // 2, :, :, 0]
        
        plt.figure(figsize=(8, 8))
        plt.imshow(mid_slice, cmap='gray')
        plt.title(f"{series_info.patient_name} - Serie {series_info.series_uid[-8:]}\n"
                 f"Label: {'TEP' if series_info.label == 1 else 'No-TEP'}")
        plt.axis('off')
        
        save_path = vis_dir / f"serie_{series_info.series_uid[-8:]}_mid.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.debug(f"Visualización guardada: {save_path}")
        
    except Exception as e:
        logger.error(f"Error generando visualización: {e}")


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

class HUCSRPreprocessor:
    """Preprocesador principal para datos HUCSR"""
    
    def __init__(self):
        self.config = HUCSRConfig()
        self.config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        #self.config.VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)
        
    def discover_all_patients(self) -> List[PatientData]:
        """Descubre todos los pacientes en HUCSR"""
        logger.info("🔍 Descubriendo pacientes HUCSR...")
        
        all_patients = []
        
        # Procesar positivos
        if self.config.POSITIVOS_DIR.exists():
            for patient_dir in self.config.POSITIVOS_DIR.iterdir():
                if patient_dir.is_dir():
                    patient_data = discover_patient_series(patient_dir)
                    if patient_data.num_series > 0:
                        all_patients.append(patient_data)
        
        # Procesar negativos
        if self.config.NEGATIVOS_DIR.exists():
            for patient_dir in self.config.NEGATIVOS_DIR.iterdir():
                if patient_dir.is_dir():
                    patient_data = discover_patient_series(patient_dir)
                    if patient_data.num_series > 0:
                        all_patients.append(patient_data)
        
        logger.info(f"✅ Descubiertos {len(all_patients)} pacientes")
        
        # Estadísticas
        total_series = sum(p.num_series for p in all_patients)
        n_positivos = sum(1 for p in all_patients if p.label == 1)
        n_negativos = len(all_patients) - n_positivos
        
        logger.info(f"   Positivos: {n_positivos}, Negativos: {n_negativos}")
        logger.info(f"   Total series: {total_series}")
        logger.info(f"   Promedio series/paciente: {total_series/len(all_patients):.1f}")
        
        return all_patients
    
    def process_all_series(self, patients: List[PatientData]) -> pl.DataFrame:
        """
        Procesa todas las series de todos los pacientes
        
        Args:
            patients: Lista de PatientData
        
        Returns:
            DataFrame con metadata de todas las series procesadas
        """
        logger.info("🚀 Iniciando procesamiento de todas las series...")
        
        # Recopilar todas las series
        all_series = []
        for patient in patients:
            if self.config.PROCESS_ALL_SERIES:
                # Procesar TODAS las series
                all_series.extend(patient.series_list)
            else:
                # Solo serie principal
                main_series = patient.get_main_series()
                if main_series:
                    all_series.append(main_series)
        
        logger.info(f"Total series a procesar: {len(all_series)}")
        
        # Procesar en paralelo con barra de progreso
        metadata_list = []
        
        with ThreadPoolExecutor(max_workers=self.config.MAX_WORKERS) as executor:
            futures = {
                executor.submit(
                    process_single_series,
                    series,
                    self.config.OUTPUT_DIR
                ): series for series in all_series
            }
            
            with tqdm(total=len(futures), desc="Procesando series") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        metadata_list.append(result)
                    pbar.update(1)
        
        # Crear DataFrame
        if not metadata_list:
            raise ValueError("No se procesaron series válidas")
        
        df = pl.DataFrame(metadata_list)
        
        # Guardar metadata
        df.write_csv(self.config.METADATA_CSV)
        logger.info(f"💾 Metadata guardada: {self.config.METADATA_CSV}")
        
        # Estadísticas finales
        logger.info(f"\n📊 ESTADÍSTICAS FINALES:")
        logger.info(f"   Series procesadas: {len(df)}")
        logger.info(f"   Positivas: {len(df.filter(pl.col('label') == 1))}")
        logger.info(f"   Negativas: {len(df.filter(pl.col('label') == 0))}")
        logger.info(f"   Pacientes únicos: {df['patient_name'].n_unique()}")
        
        return df


def load_images_hucsr():
    """Función principal"""
    logger.init_logger("log_process_data_hucsr", metrics_file="log_metrics_hucsr.json")
    
    logger.info("="*80)
    logger.info("🚀 PREPROCESAMIENTO HUCSR - TODAS LAS SERIES")
    logger.info("="*80)
    
    # Crear preprocesador
    preprocessor = HUCSRPreprocessor()
    
    # Descubrir pacientes
    patients = preprocessor.discover_all_patients()
    
    if not patients:
        logger.error("❌ No se encontraron pacientes válidos")
        return
    
    # Procesar todas las series
    metadata_df = preprocessor.process_all_series(patients)
    
    logger.info("\n✅ Preprocesamiento completado exitosamente")
    logger.info(f"   Archivo metadata: {preprocessor.config.METADATA_CSV}")
    logger.info(f"   Volúmenes .npy: {preprocessor.config.OUTPUT_DIR}")
