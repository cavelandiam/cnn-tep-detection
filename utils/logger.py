import logging
import sys
from pathlib import Path
import multiprocessing as mp
import json

# --- CONFIGURACIÓN GLOBAL DEL LOGGER ---
_logger = None
_log_file = None
_metrics_file = None

def init_logger(name: str, log_dir: str = "logs", metrics_file: str = "metrics_log.json"):
    """
    Inicializa el logger con un nombre y archivo de log específicos.
    Solo se inicializa en el proceso principal.

    Args:
        name (str): Nombre del logger y base del archivo de log (ej: "train_rsna")
        log_dir (str): Directorio donde se guardarán los logs (por defecto: "logs")
        metrics_file (str): Nombre del archivo JSON para métricas (por defecto: "metrics_log.json")
    """

    global _logger, _log_file, _metrics_file

    # Solo inicializar en el proceso principal
    if mp.current_process().name != 'MainProcess':
        return

    # Crear directorio de logs si no existe
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Definir ruta del archivo de log
    _log_file = f"{log_dir}/{name}.log"
    _metrics_file = f"{log_dir}/{metrics_file}"

    # Si ya existe un logger anterior, limpiarlo
    if _logger is not None:
        for handler in _logger.handlers[:]:
            _logger.removeHandler(handler)
        _logger.handlers.clear()

    # Crear nuevo logger
    _logger = logging.getLogger(name)
    _logger.setLevel(logging.INFO)
    _logger.propagate = False  # Evita propagación a root logger

    # Formato común
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Handler de archivo
    file_handler = logging.FileHandler(_log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)

    # Handler de consola
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Añadir handlers
    _logger.addHandler(file_handler)
    _logger.addHandler(console_handler)

    # Filtro para pydicom
    class IgnoreInvalidVRUIFilter(logging.Filter):
        def filter(self, record):
            msg = record.getMessage()
            return "Invalid value for VR UI" not in msg

    pydicom_logger = logging.getLogger("pydicom")
    pydicom_logger.addFilter(IgnoreInvalidVRUIFilter())

    # Ignorar advertencias de Python
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, message='.*Invalid value for VR UI.*')

    # Mensaje de inicio
    info(f"✅ Logger inicializado: {_log_file}")
    info(f"✅ Registro de métricas inicializado: {_metrics_file}")

def _get_logger():
    """Devuelve el logger actual. Crea uno dummy si no se inicializó."""
    global _logger
    if _logger is None:
        # Crear logger dummy para procesos hijos
        _logger = logging.getLogger("dummy")
        _logger.setLevel(logging.CRITICAL)  # Silenciar en procesos hijos
        _logger.addHandler(logging.NullHandler())
    return _logger

# --- FUNCIONES DE LOGGING SIMPLES ---

def info(message: str):
    _get_logger().info(message)

def warning(message: str):
    _get_logger().warning(message)

def error(message: str):
    _get_logger().error(message)

def debug(message: str):
    _get_logger().debug(message)

def critical(message: str):
    _get_logger().critical(message)

# --- FUNCIONES ESPECIALIZADAS PARA TU PROYECTO ---

def log_step(step: str, status: str = "OK", details: str = ""):
    msg = f"[{step}] {status}"
    if details:
        msg += f" → {details}"
    info(msg)

def start_training(model_name: str, epochs: int, batch_size: int):
    info(f"🚀 Iniciando entrenamiento de '{model_name}' | Epochs: {epochs} | Batch: {batch_size}")

def end_training(elapsed_time: float, final_loss: float, final_acc: float):
    info(f"✅ Entrenamiento completado en {elapsed_time:.2f}s | Loss: {final_loss:.4f} | Acc: {final_acc:.4f}")

def checkpoint_saved(epoch: int, loss: float, filepath: str):
    info(f"💾 Checkpoint guardado (Epoch {epoch}) | Loss: {loss:.4f} | Path: {filepath}")

def gpu_info(gpus: list):
    if gpus:
        names = [gpu.name for gpu in gpus]
        info(f"🖥️  GPU(s) detectada(s): {len(gpus)} → {names}")
    else:
        warning("⚠️  No se detectaron GPUs. Usando CPU.")

def model_summary(model):
    """Log resumen del modelo Keras"""
    from io import StringIO
    import sys
    old_stdout = sys.stdout
    sys.stdout = buffer = StringIO()
    model.summary()
    sys.stdout = old_stdout
    summary_str = buffer.getvalue()
    info("\n" + "="*60 + "\n" + "MODEL SUMMARY:\n" + summary_str + "="*60)

# --- NUEVA FUNCIÓN PARA MÉTRICAS JSON ---
def log_metrics(epoch: int, train_metrics: dict, val_metrics: dict):
    """
    Escribe las métricas de entrenamiento y validación en un archivo JSON.
    """
    global _metrics_file

    # Cargar datos existentes si el archivo existe
    metrics_data = []
    if Path(_metrics_file).exists():
        try:
            with open(_metrics_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    metrics_data = json.loads(content)
        except json.JSONDecodeError as e:
            warning(f"Error decodificando { _metrics_file}: {str(e)}. Iniciando nuevo array.")
            metrics_data = []

    # Formatear métricas con 4 decimales
    train_formatted = {k: f"{v:.4f}" for k, v in train_metrics.items()}
    val_formatted = {k: f"{v:.4f}" for k, v in val_metrics.items()}

    # Crear nuevo objeto de época
    epoch_data = {
        "EPOCH": epoch,
        "TRAIN": train_formatted,
        "VALIDATION": val_formatted
    }

    # Añadir al array
    metrics_data.append(epoch_data)

    # Escribir el array completo al archivo
    try:
        with open(_metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics_data, f, ensure_ascii=False, indent=2)
        info(f"📊 Métricas de época {epoch} guardadas en {_metrics_file}")
    except Exception as e:
        error(f"Error al guardar métricas en {_metrics_file}: {str(e)}")