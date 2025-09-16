import logging
import sys
from pathlib import Path

# --- CONFIGURACIÓN GLOBAL DEL LOGGER ---
# Variables globales para manejar el logger por script
_logger = None
_log_file = None

def init_logger(name: str, log_dir: str = "logs"):
    """
    Inicializa el logger con un nombre y archivo de log específicos.
    Debe llamarse AL INICIO de cada script (.py) que use logging.

    Args:
        name (str): Nombre del logger y base del archivo de log (ej: "train_rsna")
        log_dir (str): Directorio donde se guardarán los logs (por defecto: "logs")
    """
    global _logger, _log_file

    # Crear directorio de logs si no existe
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Definir ruta del archivo de log
    _log_file = f"{log_dir}/{name}.log"

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



    # Creamos un filtro personalizado
    class IgnoreInvalidVRUIFilter(logging.Filter):
        def filter(self, record):
            msg = record.getMessage()
            return (
                "Invalid value for VR UI" not in msg 
                #and "Unsupported VR" not in msg
                #and "Missing required tag" not in msg
            )

    # Aplicamos el filtro al logger de pydicom (no al nuestro)
    pydicom_logger = logging.getLogger("pydicom")
    pydicom_logger.addFilter(IgnoreInvalidVRUIFilter())

    # También ignoramos las warnings de Python (opcional, pero redundante)
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, message='.*Invalid value for VR UI.*')



    # Mensaje de inicio
    info(f"✅ Logger inicializado: {_log_file}")


def _get_logger():
    """Devuelve el logger actual. Lanza error si no se inicializó."""
    if _logger is None:
        raise RuntimeError(
            "Logger no inicializado. Llama a init_logger(name) al inicio de tu script."
        )
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