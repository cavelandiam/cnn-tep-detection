import numpy as np
from pathlib import Path
import logging
from collections import defaultdict
import pydicom
from utils.config import DICOM_TEP_TRUE_DIR, DICOM_TEP_FALSE_DIR, IMAGE_SIZE

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/target_depth_calculation_by_series.log'),
        logging.StreamHandler()
    ]
)

def analizar_num_cortes(directory):
    """
    Analiza la cantidad de cortes en todas las series DICOM de un conjunto de datos.
    
    Args:
        directory (str): Ruta de la carpeta con estudios DICOM.
        filter_series (bool): Si True, filtra series con 'CTPA' o 'Contrast' en SeriesDescription.
    
    Returns:
        dict: Estadísticas de cortes por serie (máximo, mínimo, promedio, lista, mediana, percentiles).
    """
    num_cortes_lista = []
    slice_thickness_lista = []
    directory = Path(directory)

    for patient in directory.iterdir():
        if not patient.is_dir():
            continue
        st0_path = patient / "ST0"
        if st0_path.exists() and st0_path.is_dir():
            # Agrupar por SeriesInstanceUID
            series_dict = defaultdict(list)
            for f in st0_path.iterdir():
                try:
                    ds = pydicom.dcmread(f, force=True)
                    series_dict[ds.SeriesInstanceUID].append(f)
                except Exception as e:
                    logging.error(f"Error al leer {f}: {e}")
                    continue
            
            # Procesar todas las series
            for series_uid, files in series_dict.items():
                try:
                    ds = pydicom.dcmread(files[0])
                    series_desc = ds.get('SeriesDescription', '').lower()
                                        
                    num_cortes = len(files)
                    num_cortes_lista.append(num_cortes)
                    
                    # Obtener SliceThickness
                    slice_thickness = float(ds.get('SliceThickness', 1.0))
                    slice_thickness_lista.append(slice_thickness)
                    
                    logging.info(f"Paciente {patient.name}, Serie {series_uid[-8:]}: {num_cortes} cortes, "
                                 f"SeriesDescription={series_desc}, SliceThickness={slice_thickness} mm")
                    
                except Exception as e:
                    logging.warning(f"No se pudo procesar serie {series_uid} de {patient.name}: {e}")
                    continue

    if not num_cortes_lista:
        logging.warning(f"No se encontraron series válidas en {directory}")
        return {
            "max": 0, "min": 0, "mean": 0, "median": 0, "p75": 0, "p90": 0,
            "lista": [], "slice_thickness_mean": 0
        }

    stats = {
        "max": np.max(num_cortes_lista),
        "min": np.min(num_cortes_lista),
        "mean": np.mean(num_cortes_lista),
        "median": np.median(num_cortes_lista),
        "p75": np.percentile(num_cortes_lista, 75),
        "p90": np.percentile(num_cortes_lista, 90),
        "lista": num_cortes_lista,
        "slice_thickness_mean": np.mean(slice_thickness_lista),
        "num_series": len(num_cortes_lista)
    }
    logging.info(f"Estadísticas de cortes en {directory}: {stats}")
    return stats

def estimate_memory_usage(target_depth, image_size, n_samples):
    """
    Estima el uso de memoria para un volumen 3D.

    Args:
        target_depth (int): Profundidad del volumen.
        image_size (tuple): Tamaño de la imagen (height, width).
        n_samples (int): Número de volúmenes.

    Returns:
        float: Tamaño estimado en MB.
    """
    volume_size_bytes = target_depth * image_size[0] * image_size[1] * 4
    total_size_mb = (volume_size_bytes * n_samples) / (1024 ** 2)
    return total_size_mb

def suggest_target_depth(stats_tep, stats_no_tep, image_size, max_memory_mb=8000):
    """
    Sugiere un TARGET_DEPTH óptimo basado en estadísticas de cortes por serie.

    Args:
        stats_tep (dict): Estadísticas de cortes para TEP.
        stats_no_tep (dict): Estadísticas de cortes para No-TEP.
        image_size (tuple): Tamaño de la imagen (height, width).
        max_memory_mb (float): Límite de memoria en MB (8 GB para datos).

    Returns:
        int: TARGET_DEPTH sugerido.
    """
    # Combinar estadísticas
    all_cuts = stats_tep['lista'] + stats_no_tep['lista']
    median_cuts = np.median(all_cuts)
    p75_cuts = np.percentile(all_cuts, 75)
    slice_thickness = np.mean([stats_tep['slice_thickness_mean'], stats_no_tep['slice_thickness_mean']])

    # Cobertura anatómica (30 cm)
    thorax_coverage_mm = 300
    min_depth_anatomic = int(thorax_coverage_mm / slice_thickness)

    # Posibles valores de TARGET_DEPTH
    possible_depths = [128, 256, 512]
    candidates = [d for d in possible_depths if d >= min_depth_anatomic]

    # Seleccionar el depth más cercano a la mediana
    n_samples = stats_tep['num_series'] + stats_no_tep['num_series']
    selected_depth = None
    for depth in candidates:
        if depth >= median_cuts:
            memory_mb = estimate_memory_usage(depth, image_size, n_samples)
            if memory_mb <= max_memory_mb:
                selected_depth = depth
                break

    if selected_depth is None:
        selected_depth = min(candidates, key=lambda x: abs(x - median_cuts))

    logging.info(f"Sugerencia de TARGET_DEPTH: {selected_depth}")
    logging.info(f" - Cubre mediana: {median_cuts:.0f} cortes")
    logging.info(f" - Cubre percentil 75: {p75_cuts:.0f} cortes")
    logging.info(f" - Cobertura anatómica mínima: {min_depth_anatomic} cortes (para {thorax_coverage_mm} mm)")
    logging.info(f" - Uso estimado de memoria: {estimate_memory_usage(selected_depth, image_size, n_samples):.2f} MB "
                 f"para {n_samples} series")
    return selected_depth

def calculate():
    """Calcula y sugiere un TARGET_DEPTH óptimo basado en cortes por serie."""
    stats_tep = analizar_num_cortes(DICOM_TEP_TRUE_DIR)
    stats_no_tep = analizar_num_cortes(DICOM_TEP_FALSE_DIR)
    target_depth = suggest_target_depth(stats_tep, stats_no_tep, IMAGE_SIZE)
    logging.info(f"TARGET_DEPTH óptimo recomendado: {target_depth}")
