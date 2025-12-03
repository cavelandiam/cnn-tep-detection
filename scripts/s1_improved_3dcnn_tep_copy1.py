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
import torchvision.models as models

from utils import logger, config, visualization
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
    
    window_center, window_width = 200, 700  # Windowing para contraste vascular (TEP)
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

class BasicBlock3D(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class ResNet34_3D(nn.Module):
    def __init__(self, pretrained_2d=True):
        super().__init__()
        
        # Cargar ResNet34 2D
        resnet2d = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=pretrained_2d)

        # STEM CORRECTO: kernel temporal = 1 (solo al inicio)
        self.conv1 = nn.Conv3d(1, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))

        # INFLADO CORRECTO DEL CONV1
        with torch.no_grad():
            w2d = resnet2d.conv1.weight  # [64, 3, 7, 7]
            w_gray = w2d.mean(dim=1, keepdim=True)  # [64, 1, 7, 7]
            w_inflated = w_gray.unsqueeze(2)  # [64, 1, 1, 7, 7] ← kernel temporal = 1
            self.conv1.weight.copy_(w_inflated)

        # Capas residuales (inflado automático)
        self.layer1 = self._inflate_residual_layer(resnet2d.layer1, 64)
        self.layer2 = self._inflate_residual_layer(resnet2d.layer2, 64)
        self.layer3 = self._inflate_residual_layer(resnet2d.layer3, 128)
        self.layer4 = self._inflate_residual_layer(resnet2d.layer4, 256)

        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(512, 1)

    def _inflate_residual_layer(self, layer2d, in_channels):
        blocks = []
        for i, block2d in enumerate(layer2d):
            stride = 2 if i == 0 and block2d.downsample is not None else 1
            downsample = None
            if block2d.downsample is not None:
                downsample = nn.Sequential(
                    nn.Conv3d(in_channels, in_channels * 4, kernel_size=1, stride=(1 if stride == 1 else (2,2,2)), bias=False),
                    nn.BatchNorm3d(in_channels * 4),
                )
            block3d = BasicBlock3D(
                inplanes=in_channels if i == 0 else block2d.conv2.out_channels,
                planes=block2d.conv2.out_channels,
                stride=(1, stride, stride) if stride > 1 else 1,
                downsample=downsample
            )
            # Inflado de pesos
            with torch.no_grad():
                w = block2d.conv1.weight
                block3d.conv1.weight.copy_(w.unsqueeze(2))
                w = block2d.conv2.weight
                block3d.conv2.weight.copy_(w.unsqueeze(2))
                if downsample:
                    w = block2d.downsample[0].weight
                    downsample[0].weight.copy_(w.unsqueeze(2))
            blocks.append(block3d)
            in_channels = block2d.conv2.out_channels
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.conv1(x)      # [B, 64, D, H/2, W/2]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)    # [B, 64, D/2, H/4, W/4]

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
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

def build_resnet3d_model():
    logger.info("Creando Video ResNet-34 3D (oficial torchvision) preentrenado en Kinetics-400...")
    
    # Modelo oficial 3D de PyTorch, preentrenado en videos reales
    model = models.video.r3d_18(pretrained=True, progress=True)  # o r3d_18 si quieres más rápido
    
    # Cambiar entrada: de 3 canales (RGB) → 1 canal (grayscale)
    original_conv1 = model.stem[0]  # Conv3d(3, 64, ...)
    new_conv1 = nn.Conv3d(
        in_channels=1,
        out_channels=64,
        kernel_size=original_conv1.kernel_size,
        stride=original_conv1.stride,
        padding=original_conv1.padding,
        bias=False
    )
    
    # Inicializar con promedio de pesos RGB → gris
    with torch.no_grad():
        new_conv1.weight = nn.Parameter(original_conv1.weight.mean(dim=1, keepdim=True))
    
    model.stem[0] = new_conv1
    
    # Cambiar cabeza: de 400 clases → 1
    model.fc = nn.Linear(model.fc.in_features, 1)
    
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Modelo 3D cargado → Total params: {total_params:,} | Trainable: {trainable_params:,}")
    
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

def plot_training_curves(history: Dict[str, List[float]], output_dir: str = "graphs"):
    metrics_keys = ['loss', 'accuracy', 'precision', 'recall', 'auc', 'f1', 'specificity', 'mcc', 'pr_auc']

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Guardando gráficas de métricas en: {out_path.resolve()}")

    for metric in metrics_keys:
        train_key = metric
        val_key = f'val_{metric}'
        
        if train_key not in history or val_key not in history:
            logger.warning(f"Métrica {metric} no encontrada en history. Saltando...")
            continue
        
        train_vals = history[train_key]
        val_vals = history[val_key]
        
        if not train_vals or not val_vals:
            continue
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_vals, label=f'Train {metric}', color='royalblue', linewidth=2.5)
        plt.plot(val_vals, label=f'Val {metric}', color='orangered', linestyle='--', linewidth=2.5)
        
        plt.title(f'{metric.upper()} vs Epoch', fontsize=16, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel(metric.capitalize(), fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Guardar con nombre de la métrica
        save_path = out_path / f"{metric}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

def calculate_confusion_matrix(val_df: pl.DataFrame, model: nn.Module, val_loader: DataLoader, device: torch.device, route: str = "graphs") -> float:
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
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Real')
    plt.savefig(route, dpi=300)
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
    logger.info("=== INICIANDO ENTRENAMIENTO RSNA-PE 3D-CNN (VÁLIDO) ===")

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
    studies_df = pl.read_csv(config.RSNA_CSV_PREPROCESSED_DATA_TRAIN_DIR)
    logger.info(f"Estudios cargados: {len(studies_df)}")



    logger.info("1.1. Aplicando balanceo 1:1 de clases (undersampling de negativos)...")
    # Separar positivos y negativos
    positive_df = studies_df.filter(pl.col("label") == 1)
    negative_df = studies_df.filter(pl.col("label") == 0)
    
    num_positives = len(positive_df)
    num_negatives = len(negative_df)
    logger.info(f"Antes del balanceo → Positivos: {num_positives}, Negativos: {num_negatives}")
    negative_df_balanced = negative_df.sample(n=num_positives, seed=SEED, shuffle=True)
    studies_df_balanced = pl.concat([positive_df, negative_df_balanced]).sort("StudyInstanceUID")
    logger.info(f"Balanceado → Total: {len(studies_df)} (Pos: {num_positives}, Neg: {num_positives}) → Ratio 1:1")
    studies_df = studies_df_balanced
    



    # Paso 2: Dividir datos
    logger.info("2. Dividiendo datos...")
    train_df_pd, val_df_pd = train_test_split(studies_df.to_pandas(),test_size=0.2,random_state=SEED,stratify=studies_df["label"])
    train_df = pl.from_pandas(train_df_pd)
    val_df = pl.from_pandas(val_df_pd)
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}")   


    logger.info("3. Calculando pesos de clase...")
    pos_weight = torch.tensor([1.0], device=device)
    logger.info(f"Clases balanceadas → pos_weight = {pos_weight.item()}")





    logger.info("4. Creando datasets...")
    train_dataset = RSNADataset(train_df, is_train=True)
    val_dataset = RSNADataset(val_df, is_train=False)
    
    logger.info(f"Tamaño de train_dataset: {len(train_dataset)}")
    logger.info(f"Tamaño de val_dataset: {len(val_dataset)}")     
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
                              num_workers=config.NUM_WORKERS, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                            num_workers=config.NUM_WORKERS, pin_memory=True, persistent_workers=True)
    
    
    logger.info(f"Esperado número de lotes: train={len(train_loader)}, val={len(val_loader)}")

    clear_gpu_memory()




    logger.info("5. Creando modelo ResNet3D optimizado...")
    # Modelo + optimizador
    model = build_resnet3d_model()
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    metrics = create_metrics(device)
    accumulation_steps = getattr(config, 'GRADIENT_ACCUMULATION_STEPS', config.GRADIENT_ACCUMULATION_STEPS)
    logger.info(f"Gradient accumulation: {accumulation_steps} pasos → batch efectivo = {config.BATCH_SIZE * accumulation_steps}")

    visualization.plot_model_architecture(
        model=model,
        save_dir=config.RSNA_GRAPHS_DIR,
        model_name= config.RSNA_GRAPHS_MODEL_NAME,
        input_size=(1, config.TARGET_DEPTH, *config.IMAGE_SIZE)  # (C, D, H, W, 1)
    )


    logger.info("6. Configurando entrenamiento...")

    best_val_auc = 0.0
    no_improve = 0
    
    history = {
        'loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'auc': [], 'f1': [],
        'specificity': [], 'mcc': [], 'pr_auc': [],
        'val_loss': [], 'val_accuracy': [], 'val_precision': [], 'val_recall': [], 'val_auc': [], 'val_f1': [],
        'val_specificity': [], 'val_mcc': [], 'val_pr_auc': []
    }
    

    
    logger.info(f"🎯 Iniciando entrenamiento para {config.EPOCHS} épocas...")
    log_gpu_memory()

    for epoch in range(config.EPOCHS):
        logger.info(f"\n--- ÉPOCA {epoch+1}/{config.EPOCHS} ---")

        # ==================== ENTRENAMIENTO ====================
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_preds, train_labels = [], []
        running_loss = 0.0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).unsqueeze(1).float()

            with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
    
            running_loss += loss.item() * accumulation_steps
            preds = torch.sigmoid(outputs).detach()
            train_preds.extend(preds.cpu().numpy().flatten())
            train_labels.extend(labels.cpu().numpy().flatten())

            del images, labels, outputs, loss, preds
            torch.cuda.empty_cache()

        # Métricas de entrenamiento
        train_metrics = calculate_metrics(train_labels, train_preds, metrics)
        train_metrics['loss'] = running_loss / len(train_loader)

        # ==================== VALIDACIÓN ====================
        val_metrics = validate_epoch(model, val_loader, criterion, device, metrics)

        # ==================== GUARDAR EN HISTORY ====================
        for key in ['loss', 'accuracy', 'precision', 'recall', 'auc', 'f1', 'specificity', 'mcc', 'pr_auc']:
            history[key].append(train_metrics.get(key, 0.0))
            history[f'val_{key}'].append(val_metrics.get(key, 0.0))

        # ==================== LOGGING ====================
        update_history_and_log(epoch, train_metrics, val_metrics, history)

        # Guardar mejor modelo
        current_val_auc = val_metrics['auc']
        if current_val_auc > best_val_auc:
            best_val_auc = current_val_auc
            save_model_checkpoint(model, config.RSNA_BEST_MODEL_AUC, best_val_auc, epoch)
            logger.info(f"NUEVO MEJOR MODELO → AUC: {best_val_auc:.4f} (Época {epoch})")
            no_improve = 0
        else:
            no_improve += 1

        # Early stopping
        if no_improve >= config.PATIENCE_EARLY_STOPPING:
            logger.info(f"Early stopping activado en época {epoch}")
            break

        clear_gpu_memory()        
    
    # ==================== FINAL ====================
    logger.info(f"\nENTRENAMIENTO COMPLETADO → Mejor Val AUC: {best_val_auc:.4f}")

    torch.save(model.state_dict(), config.RSNA_PRETRAINED_MODEL)
    logger.info(f"Modelo final guardado en: {config.RSNA_PRETRAINED_MODEL}")

    logger.info("📊 Generando visualizaciones...")
    plot_training_curves(history, config.RSNA_GRAPHS_METRICS_DIR)
    calculate_confusion_matrix(val_df, model, val_loader, device, config.RSNA_GRAPHS_CONFUSION_MATRIX_DIR)
        
    logger.info("✅ Preentrenamiento completado exitosamente")
    return history, best_val_auc
