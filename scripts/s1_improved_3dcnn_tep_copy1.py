# scripts/s1_improved_3dcnn_tep_v2.py
"""
Sistema MS-ResNet3D-PE Mejorado para Detección de Tromboembolismo Pulmonar
Versión 2.0 - Con arquitectura profunda y optimizaciones avanzadas

Mejoras principales:
1. Arquitectura ResNet3D más profunda (inspirada en ResNet34)
2. Mecanismo de atención espacial
3. Augmentaciones robustas optimizadas para imágenes médicas
4. Estrategia de entrenamiento mejorada con schedulers avanzados
5. Preprocesamiento adaptativo de ventanas Hounsfield
6. Focal Loss para mejor manejo de desbalance
"""

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
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from monai.transforms import (
    Compose, RandFlipD, RandRotateD, RandAdjustContrastD, 
    RandGaussianNoiseD, RandAffineD, RandGaussianSmoothD,
    RandScaleIntensityD, RandShiftIntensityD
)
from torchmetrics.classification import (
    BinaryAUROC, BinaryPrecision, BinaryRecall, BinaryAccuracy, 
    BinaryF1Score, BinarySpecificity, MatthewsCorrCoef, PrecisionRecallCurve
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, auc
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import zoom
from skimage.transform import resize
import psutil
import torch.multiprocessing as mp

from utils import logger, config, visualization

# =============================================================================
# CONFIGURACIÓN PARA REPRODUCIBILIDAD
# =============================================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False  # Cambiar a False para más reproducibilidad

# =============================================================================
# CONFIGURACIÓN DE LOGGING Y DISPOSITIVO
# =============================================================================

if mp.current_process().name == 'MainProcess':
    logger.init_logger("log_improved_rsna_v2", metrics_file="log_metrics_rsna_v2.json")

try:
    mp.set_start_method('spawn', force=True)
    logger.info("✅ Método de inicio de multiprocessing configurado a 'spawn'")
except RuntimeError as e:
    logger.warning(f"⚠️ No se pudo configurar el método 'spawn': {str(e)}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info(f"Usando dispositivo: {device}")

if device.type == "cuda":
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"Memoria total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA version: {torch.version.cuda}")

# =============================================================================
# PREPROCESAMIENTO DICOM MEJORADO
# =============================================================================

def adaptive_windowing(img: np.ndarray, ds: pydicom.FileDataset) -> np.ndarray:
    """
    Ventana Hounsfield adaptativa optimizada para angiotacs pulmonares
    
    Args:
        img: Array de imagen en unidades Hounsfield
        ds: Dataset DICOM con metadata
    
    Returns:
        Imagen normalizada y windowed
    """
    # Ventana base para angiotac pulmonar (optimizada para TEP)
    window_center = -600
    window_width = 1600
    
    # Ajustar según características del scanner (si disponible)
    if hasattr(ds, 'KVP'):
        try:
            kvp = float(ds.KVP)
            if kvp < 100:  # Scanner de baja energía
                window_width = 1800
            elif kvp > 140:  # Scanner de alta energía
                window_width = 1400
        except (ValueError, TypeError):
            pass
    
    # Aplicar windowing
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img = np.clip(img, img_min, img_max)
    
    # Normalización a [0, 1]
    img = (img - img_min) / (img_max - img_min)
    
    return img.astype(np.float32)


def process_dicom_image_improved(ds: pydicom.FileDataset) -> Optional[np.ndarray]:
    """Procesa una imagen DICOM con preprocesamiento mejorado"""
    if not hasattr(ds, 'pixel_array') or ds.pixel_array is None:
        return None
    
    if ds.pixel_array.ndim != 2:
        return None
    
    img = ds.pixel_array.astype(np.float32)
    
    # Aplicar RescaleSlope y RescaleIntercept para convertir a HU
    if 'RescaleSlope' in ds and 'RescaleIntercept' in ds:
        try:
            slope = float(ds.RescaleSlope)
            intercept = float(ds.RescaleIntercept)
            img = img * slope + intercept
        except (ValueError, TypeError):
            pass
    
    # Windowing adaptativo
    img = adaptive_windowing(img, ds)
    
    # Resize si es necesario
    if img.shape != config.IMAGE_SIZE:
        try:
            img = resize(img, config.IMAGE_SIZE, anti_aliasing=True, preserve_range=True)
        except Exception as e:
            logger.warning(f"Error en resize: {e}")
            return None
    
    img = np.clip(img, 0.0, 1.0)
    return np.expand_dims(img, axis=-1)


def is_axial_orientation(image_orientation: List[float]) -> bool:
    """
    Verifica si una serie DICOM tiene orientación axial
    
    Args:
        image_orientation: ImageOrientationPatient de DICOM
    
    Returns:
        True si es orientación axial (corte transversal)
    """
    if len(image_orientation) != 6:
        return False
    
    row = np.array(image_orientation[:3])
    col = np.array(image_orientation[3:])
    
    # Calcular vector normal
    normal = np.cross(row, col)
    
    # Si componente Z es dominante, es axial
    return abs(normal[2]) > 0.9


def select_best_series(study_path: Path) -> Optional[Path]:
    """
    Selecciona la mejor serie para análisis de TEP
    Prioriza series axiales con mayor número de slices
    
    Args:
        study_path: Path al directorio del estudio
    
    Returns:
        Path a la mejor serie o None
    """
    series_dirs = [p for p in study_path.iterdir() if p.is_dir()]
    if not series_dirs:
        return None
    
    best_series = None
    max_axial_slices = 0
    
    for series_dir in series_dirs:
        dicom_files = list(series_dir.glob('*.dcm'))
        if len(dicom_files) < 10:  # Mínimo 10 slices
            continue
        
        try:
            # Leer metadata del primer archivo
            sample_ds = pydicom.dcmread(str(dicom_files[0]), stop_before_pixels=True)
            image_orientation = sample_ds.get('ImageOrientationPatient', None)
            
            # Verificar si es axial
            if image_orientation and is_axial_orientation(image_orientation):
                if len(dicom_files) > max_axial_slices:
                    max_axial_slices = len(dicom_files)
                    best_series = series_dir
        except Exception as e:
            logger.debug(f"Error verificando orientación: {e}")
            continue
    
    # Fallback: serie con más slices sin importar orientación
    if best_series is None:
        best_series = max(series_dirs, key=lambda p: len(list(p.glob('*.dcm'))))
    
    return best_series


def parse_dicom_study_improved(study_path: Union[str, bytes]) -> np.ndarray:
    """
    Parsea un estudio DICOM con selección inteligente de series
    
    Args:
        study_path: Path al directorio del estudio
    
    Returns:
        Volumen 3D normalizado de shape (TARGET_DEPTH, H, W, 1)
    """
    try:
        study_path_str = study_path.decode('utf-8') if isinstance(study_path, bytes) else study_path
        study_path_obj = Path(study_path_str)
        
        if not study_path_obj.exists():
            logger.warning(f"Estudio no existe: {study_path_str}")
            return create_zero_volume()
        
        # Seleccionar mejor serie
        best_series = select_best_series(study_path_obj)
        if best_series is None:
            logger.warning(f"No se encontró serie válida en {study_path_str}")
            return create_zero_volume()
        
        # Cargar archivos DICOM ordenados por SliceLocation
        dicom_files = sorted(
            best_series.glob('*.dcm'),
            key=lambda f: float(pydicom.dcmread(str(f), stop_before_pixels=True).get('SliceLocation', 0))
        )
        
        if len(dicom_files) < 10:
            logger.warning(f"Serie tiene muy pocos slices: {len(dicom_files)}")
            return create_zero_volume()
        
        logger.info(f"Procesando {len(dicom_files)} slices de {best_series.name}")
        
        # Procesar imágenes
        valid_images = []
        for dcm_file in dicom_files:
            try:
                ds = pydicom.dcmread(str(dcm_file), force=True)
                processed_img = process_dicom_image_improved(ds)
                if processed_img is not None:
                    valid_images.append(processed_img)
                del ds, processed_img
                gc.collect()
            except Exception as e:
                logger.debug(f"Error procesando {dcm_file.name}: {e}")
                continue
        
        if len(valid_images) < 10:
            logger.warning(f"Solo {len(valid_images)} imágenes válidas")
            return create_zero_volume()
        
        # Crear volumen
        volume = np.stack(valid_images[:config.TARGET_DEPTH], axis=0)
        resized_volume = resize_volume_depth(volume, config.TARGET_DEPTH)
        
        if resized_volume.shape == (config.TARGET_DEPTH, *config.IMAGE_SIZE, 1):
            logger.info(f"✅ Volumen procesado: {resized_volume.shape}")
            return resized_volume
        else:
            logger.warning(f"Forma inválida: {resized_volume.shape}")
            return create_zero_volume()
            
    except Exception as e:
        logger.error(f"Error crítico procesando {study_path}: {e}")
        return create_zero_volume()


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
    except Exception as e:
        logger.warning(f"Error en zoom: {e}")
        return create_zero_volume()


def create_zero_volume() -> np.ndarray:
    """Crea volumen vacío con forma esperada"""
    return np.zeros((config.TARGET_DEPTH, *config.IMAGE_SIZE, 1), dtype=np.float32)


# =============================================================================
# DATASET
# =============================================================================

class RSNADataset(Dataset):
    """Dataset mejorado con mejor manejo de transforms"""
    
    def __init__(self, data_df, is_train: bool = False):
        self.data_df = data_df
        self.is_train = is_train
        self.transforms = create_strong_train_transforms() if is_train else create_val_transforms()
    
    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, idx):
        npy_path = self.data_df["preprocessed_path"][idx]
        label = self.data_df["label"][idx]
        
        # Cargar volumen
        volume = np.load(npy_path)
        volume_tensor = torch.from_numpy(volume).permute(3, 0, 1, 2).float()
        
        # Aplicar transforms
        if self.transforms:
            volume_tensor = self.transforms({"image": volume_tensor})["image"]
        
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return volume_tensor, label_tensor


# =============================================================================
# ARQUITECTURA MEJORADA - MS-ResNet3D-PE
# =============================================================================

class SpatialAttention3D(nn.Module):
    """
    Mecanismo de atención espacial 3D
    Ayuda al modelo a enfocarse en regiones con posible TEP
    """
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // 8, kernel_size=1, bias=False),
            nn.BatchNorm3d(in_channels // 8),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // 8, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        attention = self.conv(x)
        return x * attention


class ResidualBlock3D(nn.Module):
    """
    Bloque residual 3D optimizado
    Similar a ResNet básico pero adaptado para 3D
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=3, 
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)
        
        return out


class ImprovedResNet3D(nn.Module):
    """
    ResNet3D mejorado inspirado en ResNet34
    
    Arquitectura:
    - Initial conv: 1 -> 32 channels
    - Layer1: 3 bloques, 32 -> 64 channels
    - Layer2: 4 bloques, 64 -> 128 channels
    - Layer3: 6 bloques, 128 -> 256 channels
    - Layer4: 3 bloques, 256 -> 512 channels
    - Attention mechanism
    - FC layers: 512 -> 256 -> 1
    
    Total: 16 bloques residuales (vs 3 en versión anterior)
    """
    
    def __init__(self, input_channels: int = 1, num_classes: int = 1):
        super().__init__()
        
        # Convolución inicial
        self.initial_conv = nn.Sequential(
            nn.Conv3d(input_channels, 32, kernel_size=7, stride=(1, 2, 2), 
                     padding=3, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )
        
        # 4 capas de bloques residuales (arquitectura similar a ResNet34)
        self.layer1 = self._make_layer(32, 64, num_blocks=3, stride=1)
        self.layer2 = self._make_layer(64, 128, num_blocks=4, stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks=6, stride=2)
        self.layer4 = self._make_layer(256, 512, num_blocks=3, stride=2)
        
        # Mecanismo de atención
        self.attention = SpatialAttention3D(512)
        
        # Pooling y clasificación
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.dropout1 = nn.Dropout(0.3)  # Reducido de 0.5
        self.fc1 = nn.Linear(512, 256)
        self.relu = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, num_classes)
        
        # Inicializar pesos
        self._initialize_weights()
    
    def _make_layer(self, in_channels: int, out_channels: int, 
                    num_blocks: int, stride: int) -> nn.Sequential:
        """Crea una capa con múltiples bloques residuales"""
        layers = []
        
        # Primer bloque puede tener stride
        layers.append(ResidualBlock3D(in_channels, out_channels, stride))
        
        # Bloques subsecuentes tienen stride=1
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock3D(out_channels, out_channels, stride=1))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Inicializa pesos con Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.initial_conv(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.attention(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        
        return x


def build_improved_resnet3d(input_channels: int = 1, num_classes: int = 1) -> nn.Module:
    """Construye e inicializa el modelo mejorado"""
    logger.info(f"Creando ImprovedResNet3D: input_channels={input_channels}, num_classes={num_classes}")
    
    model = ImprovedResNet3D(input_channels, num_classes).to(device)
    
    # Contar parámetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parámetros totales: {total_params:,}")
    logger.info(f"Parámetros entrenables: {trainable_params:,}")
    
    # Test forward pass
    test_input = torch.randn(1, input_channels, config.TARGET_DEPTH, 
                            *config.IMAGE_SIZE).to(device)
    model.eval()
    with torch.no_grad():
        test_output = model(test_input)
        logger.info(f"✅ Test exitoso - Output shape: {test_output.shape}")
    
    model.train()
    del test_input, test_output
    torch.cuda.empty_cache()
    
    return model


# =============================================================================
# AUGMENTATIONS MEJORADAS
# =============================================================================

def create_strong_train_transforms() -> Compose:
    """
    Augmentaciones robustas optimizadas para imágenes médicas 3D
    Basadas en mejores prácticas de nnU-Net y competencias Kaggle
    """
    return Compose([
        # Geometric augmentations (alta probabilidad)
        RandFlipD(keys="image", spatial_axis=0, prob=0.5),  # Flip en D
        RandFlipD(keys="image", spatial_axis=1, prob=0.5),  # Flip en H
        RandFlipD(keys="image", spatial_axis=2, prob=0.5),  # Flip en W
        
        # Rotaciones leves (anatomía debe ser reconocible)
        RandRotateD(
            keys="image",
            range_x=np.pi / 12,  # ±15 grados
            range_y=np.pi / 12,
            range_z=np.pi / 12,
            prob=0.5
        ),
        
        # Affine transformations (simula variabilidad anatómica)
        RandAffineD(
            keys="image",
            scale_range=((-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1)),
            rotate_range=(np.pi/12, np.pi/12, np.pi/12),
            translate_range=(10, 10, 10),
            prob=0.5,
            padding_mode="border"
        ),
        
        # Intensity augmentations (CRÍTICO para CT)
        RandAdjustContrastD(keys="image", gamma=(0.7, 1.3), prob=0.5),
        RandGaussianNoiseD(keys="image", std=0.05, prob=0.3),
        RandGaussianSmoothD(
            keys="image",
            sigma_x=(0.5, 1.0),
            sigma_y=(0.5, 1.0),
            sigma_z=(0.5, 1.0),
            prob=0.3
        ),
        
        # Simulación de variabilidad de scanner
        RandScaleIntensityD(keys="image", factors=0.1, prob=0.3),
        RandShiftIntensityD(keys="image", offsets=0.1, prob=0.3),
    ])


def create_val_transforms() -> Compose:
    """Transforms para validación (sin augmentación)"""
    return Compose([])


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss para manejar desbalance de clases
    Enfoca el entrenamiento en ejemplos difíciles
    
    Args:
        alpha: Factor de balance para clases (0-1)
        gamma: Factor de modulación (típicamente 2.0)
        pos_weight: Peso adicional para clase positiva
    """
    
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, 
                 pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, pos_weight=self.pos_weight, reduction='none'
        )
        
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        return focal_loss.mean()


def create_criterion(class_weights: torch.Tensor, use_focal: bool = True) -> nn.Module:
    """
    Crea función de pérdida
    
    Args:
        class_weights: Pesos de clase [peso_neg, peso_pos]
        use_focal: Si usar Focal Loss (recomendado)
    
    Returns:
        Loss function
    """
    if use_focal:
        logger.info("Usando Focal Loss con alpha=0.75, gamma=2.0")
        return FocalLoss(alpha=0.75, gamma=2.0, pos_weight=class_weights[1])
    else:
        logger.info("Usando BCEWithLogitsLoss")
        return nn.BCEWithLogitsLoss(pos_weight=class_weights[1])


# =============================================================================
# OPTIMIZER Y SCHEDULER
# =============================================================================

def create_optimizer(model: nn.Module, learning_rate: float = 1e-4) -> optim.Optimizer:
    """
    Crea optimizador AdamW con weight decay apropiado
    """
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,  # Aumentado de 1e-5
        betas=(0.9, 0.999)
    )
    logger.info(f"Optimizador: AdamW(lr={learning_rate}, weight_decay=1e-4)")
    return optimizer


def create_scheduler(optimizer: optim.Optimizer, 
                    num_epochs: int,
                    scheduler_type: str = 'cosine') -> optim.lr_scheduler._LRScheduler:
    """
    Crea learning rate scheduler
    
    Args:
        optimizer: PyTorch optimizer
        num_epochs: Número total de épocas
        scheduler_type: 'cosine', 'step', o 'plateau'
    
    Returns:
        LR scheduler
    """
    if scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,  # Primer ciclo de 10 épocas
            T_mult=2,  # Cada ciclo subsecuente es 2x más largo
            eta_min=1e-6
        )
        logger.info("Usando CosineAnnealingWarmRestarts scheduler")
    
    elif scheduler_type == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=15,
            gamma=0.5
        )
        logger.info("Usando StepLR scheduler")
    
    elif scheduler_type == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )
        logger.info("Usando ReduceLROnPlateau scheduler")
    
    else:
        raise ValueError(f"Scheduler type {scheduler_type} no reconocido")
    
    return scheduler


# =============================================================================
# MÉTRICAS
# =============================================================================

def create_metrics(device: torch.device) -> Dict:
    """Crea diccionario de métricas"""
    return {
        'accuracy': BinaryAccuracy().to(device),
        'precision': BinaryPrecision().to(device),
        'recall': BinaryRecall().to(device),
        'f1': BinaryF1Score().to(device),
        'auc': BinaryAUROC().to(device),
        'specificity': BinarySpecificity().to(device),
        'mcc': MatthewsCorrCoef(task='binary').to(device)
    }


def calculate_metrics(labels: List, preds: List, metrics: Dict) -> Dict:
    """Calcula todas las métricas"""
    labels_tensor = torch.tensor([int(l) for l in labels], dtype=torch.long).to(device)
    preds_tensor = torch.tensor(preds, dtype=torch.float).to(device)
    
    metric_dict = {}
    for key in ['accuracy', 'precision', 'recall', 'f1', 'auc', 'specificity', 'mcc']:
        metric_dict[key] = metrics[key](preds_tensor, labels_tensor).item()
    
    # PR-AUC
    precision, recall, _ = PrecisionRecallCurve(task='binary')(preds_tensor, labels_tensor)
    pr_auc = auc(recall.cpu().numpy(), precision.cpu().numpy())
    metric_dict['pr_auc'] = pr_auc
    
    return metric_dict


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_epoch(model: nn.Module, 
                data_loader: DataLoader,
                optimizer: optim.Optimizer,
                criterion: nn.Module,
                scheduler: Optional[optim.lr_scheduler._LRScheduler],
                device: torch.device,
                metrics: Dict,
                epoch: int,
                scaler: Optional[GradScaler] = None,
                accumulation_steps: int = 1) -> Dict:
    """
    Entrena una época con gradient accumulation y mixed precision
    
    Args:
        accumulation_steps: Acumular gradientes para simular batch más grande
    """
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    logger.info(f"Iniciando época {epoch}")
    
    optimizer.zero_grad()
    
    for batch_idx, (images, labels) in enumerate(data_loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Mixed precision forward pass
        with autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
            outputs = model(images)
            loss = criterion(outputs, labels.unsqueeze(1).float())
            loss = loss / accumulation_steps  # Normalizar por accumulation
        
        # Backward pass
        if scaler:
            scaler.scale(loss).backward()
            
            # Update weights cada accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            loss.backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
        
        # Acumular métricas
        running_loss += loss.item() * accumulation_steps
        preds = torch.sigmoid(outputs).detach().cpu().numpy().flatten()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy().flatten())
        
        # Liberar memoria
        del images, labels, outputs, loss
        
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    # Step scheduler si es por época
    if scheduler is not None and not isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step()
    
    # Calcular métricas de época
    avg_loss = running_loss / len(data_loader)
    metrics_dict = calculate_metrics(all_labels, all_preds, metrics)
    metrics_dict['loss'] = avg_loss
    
    logger.info(f"Época {epoch} - Loss: {avg_loss:.4f}, AUC: {metrics_dict['auc']:.4f}, "
                f"Acc: {metrics_dict['accuracy']:.4f}, F1: {metrics_dict['f1']:.4f}")
    
    return metrics_dict


def validate_epoch(model: nn.Module,
                   data_loader: DataLoader,
                   criterion: nn.Module,
                   device: torch.device,
                   metrics: Dict) -> Dict:
    """Valida el modelo"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            with autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
                outputs = model(images)
                loss = criterion(outputs, labels.unsqueeze(1).float())
            
            running_loss += loss.item()
            preds = torch.sigmoid(outputs).cpu().numpy().flatten()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy().flatten())
            
            del images, labels, outputs, loss
            
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    
    avg_loss = running_loss / len(data_loader)
    metrics_dict = calculate_metrics(all_labels, all_preds, metrics)
    metrics_dict['loss'] = avg_loss
    
    logger.info(f"Validación - Loss: {avg_loss:.4f}, AUC: {metrics_dict['auc']:.4f}, "
                f"Acc: {metrics_dict['accuracy']:.4f}, F1: {metrics_dict['f1']:.4f}")
    
    return metrics_dict


# =============================================================================
# UTILIDADES
# =============================================================================

def save_checkpoint(model: nn.Module, path: str, val_auc: float, epoch: int):
    """Guarda checkpoint del modelo"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'val_auc': val_auc,
        'epoch': epoch,
    }, path)
    logger.info(f"✅ Checkpoint guardado: {path} (AUC: {val_auc:.4f})")


def clear_gpu_memory():
    """Limpia memoria GPU"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# =============================================================================
# MAIN TRAINING FUNCTION (solo función, sin ejecución)
# =============================================================================

def pretrain_model():
    """
    Función principal de entrenamiento mejorado
    
    IMPORTANTE: Esta función NO se ejecuta automáticamente.
    Debe ser llamada explícitamente desde tu script principal.
    """
    logger.info("="*80)
    logger.info("INICIANDO PREENTRENAMIENTO MEJORADO MS-ResNet3D-PE")
    logger.info("="*80)
    
    clear_gpu_memory()
    
    # 1. Cargar datos
    logger.info("1. Cargando metadatos...")
    metadata_path = config.RSNA_CSV_PREPROCESSED_DATA_TRAIN_DIR
    df = pl.read_csv(metadata_path)
    logger.info(f"✅ Metadatos cargados: {len(df)} estudios")
    
    # 2. Split datos
    logger.info("2. Dividiendo datos...")
    train_df, val_df = train_test_split(
        df.to_pandas(),
        test_size=0.2,
        random_state=SEED,
        stratify=df["label"]
    )
    train_df = pl.from_pandas(train_df)
    val_df = pl.from_pandas(val_df)
    logger.info(f"✅ Train: {len(train_df)}, Val: {len(val_df)}")
    
    # 3. Calcular pesos de clase
    logger.info("3. Calculando pesos de clase...")
    train_pos = len(train_df.filter(pl.col("label") == 1))
    train_neg = len(train_df) - train_pos
    
    n_total = len(train_df)
    class_weights = torch.tensor([
        (1 / train_neg) * (n_total / 2.0) if train_neg > 0 else 1.0,
        (1 / train_pos) * (n_total / 2.0) if train_pos > 0 else 1.0
    ], device=device)
    logger.info(f"✅ Pesos de clase: {class_weights.tolist()}")
    
    # 4. Crear datasets y dataloaders
    logger.info("4. Creando datasets...")
    train_dataset = RSNADataset(train_df, is_train=True)
    val_dataset = RSNADataset(val_df, is_train=False)
    
    # Ajustar batch size según memoria disponible
    batch_size = config.BATCH_SIZE if hasattr(config, 'BATCH_SIZE') else 4
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS if hasattr(config, 'NUM_WORKERS') else 2,
        pin_memory=True if device.type == 'cuda' else False,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS if hasattr(config, 'NUM_WORKERS') else 2,
        pin_memory=True if device.type == 'cuda' else False,
        persistent_workers=True
    )
    
    logger.info(f"✅ DataLoaders creados: {len(train_loader)} train batches, "
                f"{len(val_loader)} val batches")
    
    # 5. Crear modelo
    logger.info("5. Creando modelo mejorado...")
    model = build_improved_resnet3d()
    
    # 6. Configurar entrenamiento
    logger.info("6. Configurando entrenamiento...")
    learning_rate = config.LEARNING_RATE
    optimizer = create_optimizer(model, learning_rate)
    criterion = create_criterion(class_weights, use_focal=True)
    scheduler = create_scheduler(optimizer, config.EPOCHS, scheduler_type='cosine')
    metrics = create_metrics(device)
    
    use_mixed_precision = device.type == 'cuda'
    scaler = GradScaler() if use_mixed_precision else None
    
    logger.info(f"✅ Configuración completa - LR: {learning_rate}, "
                f"Mixed Precision: {use_mixed_precision}")
    
    # 7. Training loop
    logger.info(f"7. Iniciando entrenamiento por {config.EPOCHS} épocas...")
    
    history = {
        'loss': [], 'accuracy': [], 'precision': [], 'recall': [], 
        'auc': [], 'f1': [], 'specificity': [], 'mcc': [], 'pr_auc': [],
        'val_loss': [], 'val_accuracy': [], 'val_precision': [], 'val_recall': [],
        'val_auc': [], 'val_f1': [], 'val_specificity': [], 'val_mcc': [], 'val_pr_auc': []
    }
    
    best_val_auc = 0.0
    patience = config.PATIENCE_EARLY_STOPPING  # Aumentado de 5
    no_improve_count = 0
    
    for epoch in range(1, config.EPOCHS + 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"ÉPOCA {epoch}/{config.EPOCHS}")
        logger.info(f"{'='*80}")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, scheduler,
            device, metrics, epoch, scaler
        )
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, criterion, device, metrics)
        
        # Update scheduler si es ReduceLROnPlateau
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_metrics['auc'])
        
        # Guardar historia
        for key in ['loss', 'accuracy', 'precision', 'recall', 'auc', 'f1', 
                    'specificity', 'mcc', 'pr_auc']:
            history[key].append(train_metrics.get(key, 0))
            history[f'val_{key}'].append(val_metrics.get(key, 0))
        
        # Guardar mejor modelo
        current_val_auc = val_metrics['auc']
        if current_val_auc > best_val_auc:
            best_val_auc = current_val_auc
            save_checkpoint(model, config.RSNA_BEST_MODEL_AUC, best_val_auc, epoch)
            no_improve_count = 0
            logger.info(f"🌟 NUEVO MEJOR MODELO: AUC = {best_val_auc:.4f}")
        else:
            no_improve_count += 1
            logger.info(f"Sin mejora por {no_improve_count} épocas")
        
        # Early stopping
        if no_improve_count >= patience:
            logger.info(f"🛑 Early stopping en época {epoch}")
            break
        
        clear_gpu_memory()
    
    # 8. Guardar modelo final
    logger.info("\n" + "="*80)
    logger.info("ENTRENAMIENTO COMPLETADO")
    logger.info("="*80)
    logger.info(f"Mejor AUC de validación: {best_val_auc:.4f}")
    
    final_model_path = config.RSNA_PRETRAINED_MODEL
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"💾 Modelo final guardado: {final_model_path}")
    
    # 9. Visualizaciones (si tienes las funciones implementadas)
    logger.info("Generando visualizaciones...")
    # plot_training_curves(history, config.RSNA_GRAPHS_METRICS_DIR)
    # calculate_confusion_matrix(val_df, model, val_loader, device)
    
    logger.info("✅ Preentrenamiento completado exitosamente")
    
    return history, best_val_auc


# FIN DEL ARCHIVO - No hay ejecución automática