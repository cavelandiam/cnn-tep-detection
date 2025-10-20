import gc
import os
import random
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, List

import numpy as np
import polars as pl
import pydicom
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.checkpoint import checkpoint
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from monai.transforms import Compose, RandFlipD, RandRotateD, RandAdjustContrastD, RandGaussianNoiseD
from torchmetrics.classification import BinaryAUROC, BinaryPrecision, BinaryRecall, BinaryAccuracy, BinaryF1Score, BinarySpecificity, MatthewsCorrCoef, PrecisionRecallCurve
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, auc
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import zoom
from skimage.transform import resize
import psutil
import torch.multiprocessing as mp  # Importar torch.multiprocessing

from utils import logger, config
import multiprocessing as mp

# --- CONFIGURACIÓN PARA REPRODUCIBILIDAD ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# --- CONFIGURACIÓN DE LOGGING Y DISPOSITIVO ---
if mp.current_process().name == 'MainProcess':
    logger.init_logger("log_process_data_rsna", metrics_file="log_metrics_rsna.json")

try:
    mp.set_start_method('spawn', force=True)
    logger.info("✅ Método de inicio de multiprocessing configurado a 'spawn'")
except RuntimeError as e:
    logger.warning(f"⚠️ No se pudo configurar el método 'spawn': {str(e)}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info(f"Usando dispositivo: {device}")
logger.info(f"¿CUDA disponible?: {torch.cuda.is_available()}")
if device.type == "cuda":
    try:
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}, Memoria total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        logger.info(f"Memoria GPU disponible: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)) / 1e9:.2f} GB")
    except Exception as e:
        logger.error(f"Error al obtener información de GPU: {str(e)}")
else:
    logger.info("Usando CPU para entrenamiento")
    logger.info(f"RAM disponible: {psutil.virtual_memory().available / 1e9:.2f} GB")

logger.info(f"PyTorch version: {torch.__version__}")
logger.info(f"CUDA version: {torch.version.cuda}")
logger.info(f"Device capability: {torch.cuda.get_device_capability() if torch.cuda.is_available() else 'N/A'}")

# =============================================================================
# PREPROCESAMIENTO DICOM
# =============================================================================

def process_dicom_image(ds: pydicom.FileDataset) -> Optional[np.ndarray]:
    """Procesa una imagen DICOM individual"""
    if not hasattr(ds, 'pixel_array') or ds.pixel_array is None:
        return None
    
    if ds.pixel_array.ndim != 2:
        return None
    
    img = ds.pixel_array.astype(np.float32)
    
    if 'RescaleSlope' in ds and 'RescaleIntercept' in ds:
        try:
            slope = float(ds.RescaleSlope)
            intercept = float(ds.RescaleIntercept)
            img = img * slope + intercept
        except (ValueError, TypeError):
            pass
    
    window_center, window_width = -600, 1600 if ds.Modality == 'CT' else (40, 400)
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img = np.clip(img, img_min, img_max)
    img = (img - img_min) / (img_max - img_min)
    
    if img.shape != config.IMAGE_SIZE:
        try:
            img_resized = resize(img, config.IMAGE_SIZE, anti_aliasing=True, preserve_range=True)
            img = img_resized.astype(np.float32)
        except:
            pass
    
    img = np.clip(img, 0.0, 1.0)
    return np.expand_dims(img, axis=-1)

def resize_volume_depth(volume: np.ndarray, target_depth: int) -> np.ndarray:
    """Redimensiona el volumen a la profundidad objetivo"""
    current_depth = volume.shape[0]
    if current_depth < 1:
        logger.warning(f"Volumen inválido con profundidad {current_depth}")
        return np.zeros((target_depth, *volume.shape[1:]), dtype=np.float32)
    
    if current_depth == target_depth:
        return volume
    
    scale = target_depth / current_depth
    try:
        resized = zoom(volume, (scale, 1, 1, 1), order=1)
        if resized.shape[0] == target_depth:
            return resized.astype(np.float32)
        else:
            logger.warning(f"Redimensionado falló: {resized.shape}")
            return np.zeros((target_depth, *volume.shape[1:]), dtype=np.float32)
    except Exception as e:
        logger.warning(f"Error en zoom: {e}")
        return np.zeros((target_depth, *volume.shape[1:]), dtype=np.float32)

def parse_dicom_study(study_path: Union[str, bytes]) -> np.ndarray:
    """Parsea un estudio DICOM completo a volumen 3D"""
    try:
        study_path_str = study_path.decode('utf-8') if isinstance(study_path, bytes) else study_path
        study_path_obj = Path(study_path_str)
        
        if not study_path_obj.exists() or not study_path_obj.is_dir():
            logger.warning(f"Carpeta de estudio no existe: {study_path_str}")
            return create_zero_volume()
        
        series_dirs = [p for p in study_path_obj.iterdir() if p.is_dir()]
        if not series_dirs:
            logger.warning(f"No se encontraron series en {study_path_str}")
            return create_zero_volume()
        
        series_dir = max(series_dirs, key=lambda p: len(list(p.glob('*.dcm'))))
        dicom_files = sorted(series_dir.glob('*.dcm'), 
                             key=lambda f: float(pydicom.dcmread(f, stop_before_pixels=True).get('SliceLocation', 0)))
        
        if not dicom_files:
            logger.warning(f"No se encontraron archivos DICOM en {series_dir}")
            return create_zero_volume()
        
        logger.info(f"Procesando {len(dicom_files)} archivos DICOM en {series_dir.name}")
        
        valid_images = []
        total_files = len(dicom_files)
        compression_errors = 0
        
        for i, dcm_file in enumerate(dicom_files):
            try:
                ds = pydicom.dcmread(str(dcm_file), force=True)
                processed_img = process_dicom_image(ds)
                if processed_img is not None:
                    valid_images.append(processed_img)
                del ds, processed_img
                gc.collect()
                
                if (i + 1) % 50 == 0 or i == total_files - 1:
                    success_rate = len(valid_images) / (i + 1) * 100
                    logger.info(f"Progreso: {i+1}/{total_files} ({success_rate:.1f}% éxito)")
                    
            except Exception as e:
                compression_errors += 1
                error_type = classify_dicom_error(str(e))
                logger.info(f"Error {error_type} en {dcm_file.name}: {e}")
                continue
        
        success_rate = len(valid_images) / total_files * 100 if total_files > 0 else 0
        logger.info(f"Estudio {Path(study_path_str).name}: {len(valid_images)}/{total_files} "
                    f"imágenes válidas ({success_rate:.1f}%)")
        
        volume = np.stack(valid_images[:config.TARGET_DEPTH], axis=0)
        logger.info(f"Volumen inicial: {volume.shape}")
        
        resized_volume = resize_volume_depth(volume, config.TARGET_DEPTH)
        
        if resized_volume.shape == (config.TARGET_DEPTH, *config.IMAGE_SIZE, 1):
            logger.info(f"Volumen final: {resized_volume.shape}")
            return resized_volume.astype(np.float32)
        else:
            logger.warning(f"Forma inválida: {resized_volume.shape}")
            return create_zero_volume()
            
    except Exception as e:
        logger.error(f"Error crítico procesando {study_path_str}: {e}")
        return create_zero_volume()

def create_zero_volume() -> np.ndarray:
    """Crea volumen vacío con forma esperada"""
    return np.zeros((config.TARGET_DEPTH, *config.IMAGE_SIZE, 1), dtype=np.float32)

def classify_dicom_error(error_msg: str) -> str:
    """Clasifica errores de DICOM para mejor logging"""
    error_lower = error_msg.lower()
    if any(word in error_lower for word in ['decompress', 'compression', 'jpeg', 'j2k']):
        return "COMPRESIÓN"
    elif 'pixel_array' in error_lower or 'pixel data' in error_lower:
        return "PIXEL_DATA"
    elif 'file' in error_lower or 'read' in error_lower:
        return "LECTURA"
    else:
        return "DESCONOCIDO"

def preprocess_single_study(row):
    """Procesa un estudio individual y guarda como .npy"""
    study_id = row['StudyInstanceUID']
    study_path = row['study_path']
    output_dir = config.RSNA_PREPROCESSED_DATA_TRAIN_DIR
    Path(output_dir).mkdir(exist_ok=True)
    npy_path = f"{output_dir}/{study_id}.npy"
    
    if Path(npy_path).exists():
        logger.info(f"Volumen ya preprocesado para {study_id}")
        return npy_path
    
    volume = parse_dicom_study(study_path)
    np.save(npy_path, volume)
    logger.info(f"Preprocesado {study_id} guardado en {npy_path}")
    return npy_path

def preprocess_all_studies(df: pl.DataFrame, num_processes: int = 2):
    """Preprocesa todos los estudios en paralelo"""
    logger.info("Iniciando preprocesamiento paralelo de estudios...")
    with mp.Pool(processes=num_processes) as pool:
        preprocessed_paths = pool.map(preprocess_single_study, [row for row in df.iter_rows(named=True)])
    
    df = df.with_columns(pl.Series("preprocessed_path", preprocessed_paths))
    df.write_csv(config.RSNA_CSV_PREPROCESSED_DATA_TRAIN_DIR)
    logger.info(f"Preprocesamiento completado. Metadatos guardados en {config.RSNA_CSV_PREPROCESSED_DATA_TRAIN_DIR}")
    return df


# =============================================================================
# DATASET
# =============================================================================

class RSNADataset(Dataset):
    def __init__(self, data_df, is_train: bool = False):
        self.data_df = data_df
        self.is_train = is_train
        self.train_transforms = create_train_transforms() if is_train else None
        self.val_transforms = create_val_transforms() if not is_train else None
    
    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, idx):
        npy_path = self.data_df["preprocessed_path"][idx]
        label = self.data_df["label"][idx]
        
        volume = np.load(npy_path)
        volume_tensor = torch.from_numpy(volume).permute(3, 0, 1, 2).float()
        
        if self.is_train and self.train_transforms:
            volume_tensor = self.train_transforms({"image": volume_tensor})["image"]
        elif not self.is_train and self.val_transforms:
            volume_tensor = self.val_transforms({"image": volume_tensor})["image"]
        
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return volume_tensor, label_tensor


# =============================================================================
# MODEL COMPONENTS
# =============================================================================

class ResidualBlock(nn.Module):
    """Bloque residual 3D optimizado"""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride > 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
        
        self.relu_final = nn.ReLU(inplace=True)
    
    def forward(self, x):        
        out = self.conv1(x)        
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        shortcut_out = self.shortcut(x)        
        
        if out.shape != shortcut_out.shape:
            logger.error(f"Forma de main path {out.shape} != shortcut {shortcut_out.shape}")
            raise ValueError("Desajuste de formas en ResidualBlock")
        
        out += shortcut_out
        out = self.relu_final(out)        
        return out

class ResNet3D(nn.Module):
    """Modelo ResNet3D optimizado para uso eficiente de memoria"""
    def __init__(self, input_channels: int = 1, num_classes: int = 1):
        super().__init__()
        # Reducir canales iniciales de 16 a 8
        self.initial_block = nn.Sequential(
            nn.Conv3d(input_channels, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)  # Stride=2 para reducir dimensiones
        )
        # Reducir número de canales en cada capa
        self.layer1 = ResidualBlock(8, 16)
        self.layer2 = ResidualBlock(16, 32, stride=2)
        self.layer3 = ResidualBlock(32, 64, stride=2)
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64, 128)  # Reducir de 256 a 128
        self.relu = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):        
        x = self.initial_block(x)                     
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)            
        x = self.global_pool(x)            
        x = x.view(x.size(0), -1)            
        x = self.dropout1(x)
        x = self.fc1(x)            
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)            
        return x

def initialize_model_weights(model: nn.Module):
    """Inicializa pesos del modelo en el dispositivo correcto"""
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            m.weight.data = m.weight.data.to(device)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
            m.weight.data = m.weight.data.to(device)
            m.bias.data = m.bias.data.to(device)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            m.weight.data = m.weight.data.to(device)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                m.bias.data = m.bias.data.to(device)

def build_resnet3d_model(input_channels: int = 1, num_classes: int = 1) -> nn.Module:
    logger.info(f"Creando ResNet3D optimizado: input_channels={input_channels}, num_classes={num_classes}")
    
    model = ResNet3D(input_channels, num_classes).to(device)
    logger.info(f"Modelo creado en dispositivo: {next(model.parameters()).device}")
    
    initialize_model_weights(model)
    
    # Probar modelo con entrada reducida
    test_input = torch.randn(1, input_channels, config.TARGET_DEPTH, *config.IMAGE_SIZE).to(device)
    logger.info(f"Probando modelo con entrada: {test_input.shape}, dispositivo: {test_input.device}")
    
    model.eval()
    with torch.no_grad():
        try:
            test_output = model(test_input)
            logger.info(f"Salida del modelo: {test_output.shape}, dispositivo: {test_output.device}")
            if test_output.shape != torch.Size([1, num_classes]):
                logger.error(f"Forma de salida inesperada: {test_output.shape}, esperado: [1, {num_classes}]")
                raise ValueError("Forma de salida del modelo incorrecta")
        except Exception as e:
            logger.error(f"Error al probar modelo: {str(e)}")
            raise
    
    model.train()
    del test_input, test_output
    torch.cuda.empty_cache()
    
    return model


# =============================================================================
# TRANSFORMS & AUGMENTATIONS
# =============================================================================

def create_train_transforms() -> Compose:
    return Compose([
        RandFlipD(keys="image", spatial_axis=2, prob=0.2),
        RandRotateD(keys="image", range_x=20, prob=0.2),
        RandAdjustContrastD(keys="image", gamma=(0.8, 1.2), prob=0.2),
        RandGaussianNoiseD(keys="image", std=0.02, prob=0.2)
    ])

def create_val_transforms() -> Compose:
    return Compose([])


# =============================================================================
# METRICS AND VISUALIZATIONS
# =============================================================================

def plot_training_curves(history: Dict[str, List[float]]):
    metrics = ['loss', 'accuracy', 'precision', 'recall', 'auc', 'f1']
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    for i, metric in enumerate(metrics):
        ax = axes[i//3, i%3]
        ax.plot(history[metric], label=f'Train {metric}', color='royalblue', linewidth=2)
        ax.plot(history[f'val_{metric}'], label=f'Val {metric}', color='orangered', linestyle='--', linewidth=2)
        ax.set_title(f'{metric.capitalize()}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.capitalize())
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("models/rsna_training_curves.png", dpi=300, bbox_inches='tight')
    #plt.show()

def calculate_confusion_matrix(val_df: pl.DataFrame, model: nn.Module, val_loader: DataLoader, device: torch.device):
    model.eval()
    predictions, labels = [], []
    with torch.no_grad():
        for images, batch_labels in val_loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images).squeeze(-1)
            predictions.extend(torch.sigmoid(outputs).cpu().numpy())
            labels.extend(batch_labels.cpu().numpy())
            
            # Liberar memoria
            del images, batch_labels, outputs
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    
    pred_binary = (np.array(predictions) > 0.5).astype(int)
    true_binary = np.array(labels).astype(int)
    cm = confusion_matrix(true_binary, pred_binary)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusión')
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.savefig("models/rsna_confusion_matrix.png", dpi=300)
    #plt.show()
    f1_val = f1_score(true_binary, pred_binary)
    logger.info(f"F1-Score validación: {f1_val:.4f}")
    return f1_val

def create_metrics(device: torch.device):
    return {
        'accuracy': BinaryAccuracy().to(device),        
        'precision': BinaryPrecision().to(device),
        'recall': BinaryRecall().to(device),        
        'f1': BinaryF1Score().to(device),
        'auc': BinaryAUROC().to(device),
        'specificity': BinarySpecificity().to(device),
        'mcc': MatthewsCorrCoef(task='binary').to(device)
    }

def calculate_metrics(labels, preds, metrics):    
    labels = [int(label) for label in labels]
    labels = torch.tensor(labels, dtype=torch.long).to(metrics['accuracy'].device)
    preds = torch.tensor(preds, dtype=torch.float).to(metrics['accuracy'].device)

    metric_dict = {
        key: metrics[key](preds, labels).item() for key in ['accuracy', 'precision', 'recall', 'f1', 'auc', 'specificity', 'mcc']
    }

    # Calcular PR-AUC
    precision, recall, _ = PrecisionRecallCurve(task='binary')(preds, labels)
    pr_auc = auc(recall.cpu().numpy(), precision.cpu().numpy())
    metric_dict['pr_auc'] = pr_auc
        
    return metric_dict

def log_gpu_memory():
    """Monitorea y registra uso de memoria GPU"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        max_allocated = torch.cuda.max_memory_allocated() / 1e9
        logger.info(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Max: {max_allocated:.2f}GB")


# =============================================================================
# TRAINING
# =============================================================================

def create_optimizer(model: nn.Module, learning_rate: float) -> optim.Optimizer:
    return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

def create_criterion(class_weights: torch.Tensor) -> nn.Module:
    #return nn.BCEWithLogitsLoss(pos_weight=class_weights[1], weight=class_weights[0])
    return nn.BCEWithLogitsLoss(pos_weight=class_weights[1])

def train_epoch(model, data_loader, optimizer, criterion, class_weights, device, metrics, epoch, scaler=None):
    """Entrenamiento optimizado para memoria GPU"""
    model.train()
    running_loss = 0.0
    total_batches = len(data_loader)
    all_preds = []
    all_labels = []
    
    start_time = time.time()
    logger.info(f"Iniciando época {epoch}, total de lotes esperados: {total_batches}")
    log_gpu_memory()
    
    for batch_idx, (images, labels) in enumerate(data_loader):
        #logger.info(f"Procesando lote {batch_idx}: imágenes {images.shape}, etiquetas {labels.tolist()}")
        images, labels = images.to(device, non_blocking=True), labels.to(device)
        
        # Usar set_to_none=True para liberar memoria
        optimizer.zero_grad(set_to_none=True)
        
        with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu', dtype=torch.float16):
            outputs = model(images)
            loss = criterion(outputs, labels.unsqueeze(1).float())
        
        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        running_loss += loss.item()
        #logger.info(f"Epoch {epoch}, Batch {batch_idx}/{total_batches}, Loss: {loss.item():.4f}")
        
        # Calcular predicciones y liberar memoria
        preds = torch.sigmoid(outputs).detach().cpu().numpy().flatten()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy().astype(int).flatten())
        
        # Liberar tensores inmediatamente
        del images, labels, outputs, loss, preds
        
        # Limpiar cache GPU después de cada batch
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # if batch_idx % 50 == 0 and batch_idx > 0:
        #     # Calcular métricas intermedias con subset de datos
        #     subset_size = min(100, len(all_preds))
        #     subset_preds = all_preds[-subset_size:]
        #     subset_labels = all_labels[-subset_size:]
        #     metrics_dict = calculate_metrics(subset_labels, subset_preds, metrics)
        #     logger.info(f"Métricas intermedias (lote {batch_idx}): Loss: {running_loss / (batch_idx + 1):.4f}, AUC: {metrics_dict['auc']:.4f}")
        #     log_gpu_memory()
        
        #logger.info(f"Tiempo para lote {batch_idx}: {time.time() - start_time:.2f} segundos")
        start_time = time.time()
    
    # Limpieza final
    gc.collect()
    torch.cuda.empty_cache()
    
    avg_loss = running_loss / total_batches
    metrics_dict = calculate_metrics(all_labels, all_preds, metrics)
    metrics_dict['loss'] = avg_loss
    
    logger.info(f"Época {epoch} Completada: {total_batches} lotes procesados, pérdida promedio: {avg_loss:.4f}")
    logger.info(f"Métricas Finales Entrenamiento: ACC: {metrics_dict['accuracy']:.4f}, PRECISION: {metrics_dict['precision']:.4f}, RECALL: {metrics_dict['recall']:.4f}, F1: {metrics_dict['f1']:.4f}, AUC: {metrics_dict['auc']:.4f}, SPEC: {metrics_dict['specificity']:.4f}, MCC: {metrics_dict['mcc']:.4f}, PR-AUC: {metrics_dict['pr_auc']:.4f}")
    #log_gpu_memory()
    
    return metrics_dict

def validate_epoch(model, data_loader, criterion, device, metrics):
    """Validación optimizada para memoria GPU"""
    model.eval()
    running_loss = 0.0
    total_batches = len(data_loader)
    all_preds = []
    all_labels = []
    
    start_time = time.time()
    logger.info(f"Iniciando validación, total de lotes esperados: {total_batches}")
    log_gpu_memory()
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device, non_blocking=True), labels.to(device)
            
            with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu', dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, labels.unsqueeze(1).float())
            
            running_loss += loss.item()
            preds = torch.sigmoid(outputs).detach().cpu().numpy().flatten()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy().astype(int).flatten())
            
            # Liberar memoria
            del images, labels, outputs, loss, preds
            
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # if batch_idx % 50 == 0 and batch_idx > 0:
            #     subset_size = min(100, len(all_preds))
            #     subset_preds = all_preds[-subset_size:]
            #     subset_labels = all_labels[-subset_size:]
            #     metrics_dict = calculate_metrics(subset_labels, subset_preds, metrics)
            #     logger.info(f"Métricas intermedias (lote {batch_idx}/{total_batches}): Loss: {running_loss / (batch_idx + 1):.4f}, AUC: {metrics_dict['auc']:.4f}")
            
            #logger.info(f"Tiempo para lote {batch_idx}: {time.time() - start_time:.2f} segundos")
            start_time = time.time()
    
    # Limpieza final
    gc.collect()
    torch.cuda.empty_cache()
    
    avg_loss = running_loss / total_batches
    metrics_dict = calculate_metrics(all_labels, all_preds, metrics)
    metrics_dict['loss'] = avg_loss
    
    logger.info(f"Validación Completada: {total_batches} lotes procesados, pérdida promedio: {avg_loss:.4f}")    
    logger.info(f"Métricas Finales Validación: ACC: {metrics_dict['accuracy']:.4f}, PRECISION: {metrics_dict['precision']:.4f}, RECALL: {metrics_dict['recall']:.4f}, F1: {metrics_dict['f1']:.4f}, AUC: {metrics_dict['auc']:.4f}, SPEC: {metrics_dict['specificity']:.4f}, MCC: {metrics_dict['mcc']:.4f}, PR-AUC: {metrics_dict['pr_auc']:.4f}")
    #log_gpu_memory()
    
    return metrics_dict

def save_model_checkpoint(model: nn.Module, path: str, val_auc: float, epoch: int):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'val_auc': val_auc,
        'epoch': epoch,
    }, path)
    logger.info(f"Checkpoint guardado: {path} (AUC: {val_auc:.4f}, Epoch: {epoch})")

def load_model_checkpoint(model: nn.Module, path: str) -> Optional[float]:
    if not Path(path).exists():
        return None
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Checkpoint cargado: {path} (AUC: {checkpoint.get('val_auc', 0):.4f})")
    return checkpoint.get('val_auc', 0)

def test_model(device: torch.device):
    """Prueba el modelo con datos de ejemplo"""
    logger.info("Probando modelo con lote de ejemplo...")
    test_images = torch.randn(1, 1, config.TARGET_DEPTH, *config.IMAGE_SIZE).to(device)
    test_labels = torch.tensor([1.0]).to(device)
    
    logger.info(f"Lote de prueba: {test_images.shape}, etiquetas {test_labels.tolist()}")
    model = build_resnet3d_model(input_channels=1, num_classes=1).to(device)
    
    with torch.no_grad():
        try:
            outputs = model(test_images)
            logger.info(f"Salida del modelo: {outputs.shape}")
            outputs = outputs.squeeze(-1)
            logger.info(f"Después de squeeze: {outputs.shape}")
            logger.info("✅ Modelo funciona correctamente")
        except Exception as e:
            logger.error(f"Error al probar modelo: {str(e)}")
            raise
    
    del model, test_images, test_labels, outputs
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

def clear_gpu_memory():
    """Limpia memoria GPU antes de empezar entrenamiento"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    logger.info("✅ Memoria GPU limpiada")


# =============================================================================
# INITIATE TRAINING
# =============================================================================

def update_history_and_log(epoch: int, train_metrics: dict, val_metrics: dict, history: dict):
    metrics_keys = ['loss', 'accuracy', 'precision', 'recall', 'auc', 'f1', 'specificity', 'mcc', 'pr_auc']
    for key in metrics_keys:
        history[key].append(train_metrics.get(key, 0.0))  # Default a 0 si no existe
        history[f'val_{key}'].append(val_metrics.get(key, 0.0))
    
    logger.info(f"RESULTADOS ÉPOCA {epoch}:")
    logger.info(f"  Train = ACC: {train_metrics.get('accuracy', 0):.3f}, PRECISION: {train_metrics.get('precision', 0):.4f}, "
                f"RECALL: {train_metrics.get('recall', 0):.4f}, F1: {train_metrics.get('f1', 0):.3f}, "
                f"AUC: {train_metrics.get('auc', 0):.4f}, PR-AUC: {train_metrics.get('pr_auc', 0):.4f}, "
                f"SPEC: {train_metrics.get('specificity', 0):.4f}, MCC: {train_metrics.get('mcc', 0):.4f}, "
                f"Loss: {train_metrics.get('loss', 0):.4f}")
    logger.info(f"  Valid = ACC: {val_metrics.get('accuracy', 0):.3f}, PRECISION: {val_metrics.get('precision', 0):.4f}, "
                f"RECALL: {val_metrics.get('recall', 0):.4f}, F1: {val_metrics.get('f1', 0):.3f}, "
                f"AUC: {val_metrics.get('auc', 0):.4f}, PR-AUC: {val_metrics.get('pr_auc', 0):.4f}, "
                f"SPEC: {val_metrics.get('specificity', 0):.4f}, MCC: {val_metrics.get('mcc', 0):.4f}, "
                f"Loss: {val_metrics.get('loss', 0):.4f}")


# =============================================================================
# INITIATE TRAINING
# =============================================================================

def pretrain_model():
    logger.info("=== INICIANDO PREENTRENAMIENTO 3D-CNN (PyTorch) - OPTIMIZADO PARA MEMORIA ===")

    # Verificar compatibilidad CUDA
    if device.type == 'cuda':
        major, minor = torch.cuda.get_device_capability()
        if major < 7:
            logger.warning(f"⚠️ La GPU tiene arquitectura sm_{major}.{minor}, puede no ser compatible con PyTorch {torch.__version__}")
        else:
            logger.info(f"✅ Arquitectura GPU sm_{major}.{minor} compatible con PyTorch {torch.__version__}")

    # Limpiar memoria GPU al inicio
    clear_gpu_memory()
 
    logger.info("1. Preprocesando datos DICOM...")
    df = pl.read_csv(config.RSNA_CSV_TRAIN_DIR)
    logger.info(f"Total de estudios en CSV: {len(df['StudyInstanceUID'].unique())}")
    
    studies_df = df.group_by("StudyInstanceUID").agg(
        pl.col("negative_exam_for_pe").first().alias("negative_exam")
    ).with_columns([
        pl.col("StudyInstanceUID").map_elements(
            lambda uid: str(Path(config.RSNA_DATASET_TRAIN_DIR) / uid), 
            return_dtype=pl.String
        ).alias("study_path"),
        (1 - pl.col("negative_exam")).alias("label")
    ]).drop("negative_exam")
    
    studies_df = studies_df.filter(
        pl.col("study_path").map_elements(
            lambda p: Path(p).exists() and Path(p).is_dir(), 
            return_dtype=pl.Boolean
        )
    )
    
    total_studies = len(studies_df)
    logger.info(f"✅ Metadatos cargados: {total_studies} estudios válidos")
    
    if total_studies == 0:
        logger.error("No se encontraron estudios válidos. Verifica rutas.")
        raise ValueError("Dataset vacío")
    
    missing_studies = df.filter(
        ~df["StudyInstanceUID"].is_in(studies_df["StudyInstanceUID"])
    )["StudyInstanceUID"].n_unique()
    if missing_studies > 0:
        logger.warning(f"⚠️ {missing_studies} estudios sin carpetas físicas")

    # Generar archivos .npy y preprocessed_metadata.csv
    if not Path(config.RSNA_CSV_PREPROCESSED_DATA_TRAIN_DIR).exists():
        logger.info("Generando preprocessed_metadata.csv...")
        studies_df = preprocess_all_studies(studies_df, num_processes=config.NUM_PROCESSES)
    else:
        logger.info("preprocessed_metadata.csv ya existe, cargando...")
        studies_df = pl.read_csv(config.RSNA_CSV_PREPROCESSED_DATA_TRAIN_DIR)
    
    # Verificar que los archivos .npy existan
    missing_npy = studies_df.filter(
        ~pl.col("preprocessed_path").map_elements(
            lambda p: Path(p).exists(), return_dtype=pl.Boolean
        )
    )
    if len(missing_npy) > 0:
        logger.warning(f"⚠️ {len(missing_npy)} estudios sin archivos .npy preprocesados")



    # Paso 2: Dividir datos
    logger.info("2. Dividiendo datos...")
    train_df_pd, val_df_pd = train_test_split(
        studies_df.to_pandas(),
        test_size=0.2,
        random_state=SEED,
        stratify=studies_df["label"]
    )
    
    train_df = pl.from_pandas(train_df_pd)
    val_df = pl.from_pandas(val_df_pd)
    
    logger.info(f"✅ División: {len(train_df)} entrenamiento, {len(val_df)} validación")    

    # Verificar archivos .npy en train_df y val_df
    logger.info("Verificando archivos .npy en train_df...")
    for path in train_df["preprocessed_path"][:5]:
        if not Path(path).exists():
            logger.error(f"Archivo .npy no encontrado: {path}")
            raise FileNotFoundError(f"Archivo .npy no encontrado: {path}")
    logger.info("Verificando archivos .npy en val_df...")
    for path in val_df["preprocessed_path"][:5]:
        if not Path(path).exists():
            logger.error(f"Archivo .npy no encontrado: {path}")
            raise FileNotFoundError(f"Archivo .npy no encontrado: {path}")




    logger.info("3. Calculando pesos de clase...")
    train_pos = len(train_df.filter(pl.col("label") == 1))
    train_neg = len(train_df) - train_pos
    logger.info(f"   Clases train: {train_pos} positivos ({train_pos/len(train_df)*100:.1f}%), "
               f"{train_neg} negativos ({train_neg/len(train_df)*100:.1f}%)")
    
    n_total, n_neg, n_pos = len(train_df), train_neg, train_pos
    class_weights = torch.tensor([
        (1 / n_neg) * (n_total / 2.0) if n_neg > 0 else 1.0,
        (1 / n_pos) * (n_total / 2.0) if n_pos > 0 else 1.0
    ], device=device)
    logger.info(f"✅ Pesos: clase 0={class_weights[0]:.2f}, clase 1={class_weights[1]:.2f}")





    logger.info("4. Creando datasets...")
    train_dataset = RSNADataset(train_df, is_train=True)
    val_dataset = RSNADataset(val_df, is_train=False)
    
    logger.info(f"Tamaño de train_dataset: {len(train_dataset)}")
    logger.info(f"Tamaño de val_dataset: {len(val_dataset)}")     
    
    logger.info("Creando DataLoaders optimizados...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,  # Desactivar workers para evitar deadlocks
        pin_memory=True if device.type == 'cuda' else False,
        persistent_workers=True if config.NUM_WORKERS > 0 else False,
        prefetch_factor=config.PREFETCH_FACTOR if config.NUM_WORKERS > 0 else None
    )    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,  # Desactivar workers para evitar deadlocks
        pin_memory=True if device.type == 'cuda' else False,
        persistent_workers=True if config.NUM_WORKERS > 0 else False,
        prefetch_factor=config.PREFETCH_FACTOR if config.NUM_WORKERS > 0 else None
    )
    
    logger.info(f"Esperado número de lotes: train={len(train_loader)}, val={len(val_loader)}")
    
    # Probar carga de un lote
    logger.info("Probando carga de un lote de train_loader...")
    try:
        for batch in train_loader:
            volumes, labels = batch
            logger.info(f"Lote cargado: volúmenes {volumes.shape}, etiquetas {labels.shape}")
            del volumes, labels
            break
    except Exception as e:
        logger.error(f"Error al cargar lote de train_loader: {str(e)}")
        raise
    
    logger.info("Probando carga de un lote de val_loader...")
    try:
        for batch in val_loader:
            volumes, labels = batch
            logger.info(f"Lote cargado: volúmenes {volumes.shape}, etiquetas {labels.shape}")
            break
    except Exception as e:
        logger.error(f"Error al cargar lote de val_loader: {str(e)}")
        raise

    clear_gpu_memory()




    logger.info("5. Creando modelo ResNet3D optimizado...")
    model = build_resnet3d_model(input_channels=1, num_classes=1)
    logger.info(f"Modelo en dispositivo: {next(model.parameters()).device}")
    
    test_model(device)
    clear_gpu_memory()









    logger.info("6. Configurando entrenamiento...")
    optimizer = create_optimizer(model, config.LEARNING_RATE)
    criterion = create_criterion(class_weights)
    metrics = create_metrics(device)
    
    use_mixed_precision = True if device.type == 'cuda' else False
    scaler = torch.amp.GradScaler('cuda') if use_mixed_precision and device.type == 'cuda' else None
    logger.info(f"✅ Optimizador: Adam(lr={config.LEARNING_RATE}), Mixed precision: {use_mixed_precision}")
    
    history = {
        'loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'auc': [], 'f1': [],
        'specificity': [], 'mcc': [], 'pr_auc': [],
        'val_loss': [], 'val_accuracy': [], 'val_precision': [], 'val_recall': [], 'val_auc': [], 'val_f1': [],
        'val_specificity': [], 'val_mcc': [], 'val_pr_auc': []
    }
    
    best_val_auc = 0.0
    patience = 5
    no_improve_count = 0
    
    logger.info(f"🎯 Iniciando entrenamiento para {config.EPOCHS} épocas...")
    log_gpu_memory()

    for epoch in range(config.EPOCHS):
        logger.info(f"\n--- ÉPOCA {epoch+1}/{config.EPOCHS} ---")

        train_metrics = train_epoch(model, train_loader, optimizer, criterion, class_weights, device, metrics, epoch+1, scaler)

        val_metrics = validate_epoch(model, val_loader, criterion, device, metrics)
        
        logger.log_metrics(epoch + 1, train_metrics, val_metrics)

        update_history_and_log(epoch + 1, train_metrics, val_metrics, history)

        current_val_auc = val_metrics['auc']
        if current_val_auc > best_val_auc:
            best_val_auc = current_val_auc
            save_model_checkpoint(model, config.RSNA_BEST_MODEL_AUC, best_val_auc, epoch+1)
            no_improve_count = 0
            logger.info(f"🌟 NUEVO MEJOR MODELO: AUC = {best_val_auc:.4f}")
        else:
            no_improve_count += 1
        
        if no_improve_count >= patience:
            for g in optimizer.param_groups:
                g['lr'] *= 0.5
                logger.info(f"📉 LR reducido a {g['lr']:.6f}")
            no_improve_count = 0
        
        if no_improve_count >= 10:
            logger.info(f"🛑 Early stopping en época {epoch+1}")
            break

        # Limpiar memoria después de cada época
        clear_gpu_memory()
    
    logger.info(f"\n=== ENTRENAMIENTO COMPLETADO ===")
    logger.info(f"Mejor AUC validación: {best_val_auc:.4f}")

    final_model_path = config.RSNA_PRETRAINED_MODEL
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"💾 Modelo final guardado: {final_model_path}")
    
    logger.info("📊 Generando visualizaciones...")
    plot_training_curves(history)
    final_f1 = calculate_confusion_matrix(val_df, model, val_loader, device)
    
    final_train_auc = history['auc'][-1]
    final_val_auc = history['val_auc'][-1]
    logger.info(f"📈 RESULTADOS FINALES: Train AUC: {final_train_auc:.4f}, Valid AUC: {final_val_auc:.4f}, Valid F1: {final_f1:.4f}")
    
    del model, train_loader, val_loader, optimizer, criterion
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    logger.info("✅ Preentrenamiento completado exitosamente")
    return history, best_val_auc
