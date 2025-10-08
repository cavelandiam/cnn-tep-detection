import gc
import os
import random
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, List

import numpy as np
import polars as pl
import pydicom
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from monai.transforms import Compose, RandFlipD, RandRotateD, RandAdjustContrastD, RandGaussianNoiseD
from torchmetrics.classification import BinaryAUROC, BinaryPrecision, BinaryRecall, BinaryAccuracy, BinaryF1Score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import zoom
from skimage.transform import resize
import psutil

from utils import logger, config

# --- CONFIGURACIÓN PARA REPRODUCIBILIDAD ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True  # Optimizado para GPU

# --- CONFIGURACIÓN DE LOGGING Y DISPOSITIVO ---
logger.init_logger("log_process_data_rsna")

# Selección dinámica del dispositivo
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
logger.info(f"Usando dispositivo: {device}")
logger.info(f"¿CUDA disponible?: {torch.cuda.is_available()}")
if device.type == "cuda":
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}, Memoria: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    logger.info("Usando CPU para entrenamiento")
    logger.info(f"RAM disponible: {psutil.virtual_memory().available / 1e9:.2f} GB")

logger.info(torch.__version__)
logger.info(torch.version.cuda)
logger.info(torch.cuda.get_device_capability())

# --- FUNCIONES DE PROCESAMIENTO DE DATOS ---

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

# --- DATASET ---

class RSNADataset(Dataset):
    def __init__(self, data_df):
        self.data_df = data_df
    
    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, idx):
        study_path = self.data_df["study_path"][idx]
        label = self.data_df["label"][idx]
        
        volume = parse_dicom_study(study_path)
        volume_tensor = torch.from_numpy(volume).permute(3, 0, 1, 2).float()
        
        return volume_tensor, torch.tensor(label, dtype=torch.float32)

def create_validation_dataset(df: pl.DataFrame) -> Dataset:
    """Crea dataset de validación"""
    return RSNADataset(df)

# --- MODEL COMPONENTS ---

class ResidualBlock(nn.Module):
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
        logger.debug(f"ResidualBlock entrada: {x.shape}, dispositivo: {x.device}")
        out = self.conv1(x)
        logger.debug(f"Después de conv1: {out.shape}, dispositivo: {out.device}")
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        shortcut_out = self.shortcut(x)
        logger.debug(f"Shortcut salida: {shortcut_out.shape}, dispositivo: {shortcut_out.device}")
        
        if out.shape != shortcut_out.shape:
            logger.error(f"Forma de main path {out.shape} != shortcut {shortcut_out.shape}")
            raise ValueError("Desajuste de formas en ResidualBlock")
        
        out += shortcut_out
        out = self.relu_final(out)
        logger.debug(f"ResidualBlock salida: {out.shape}, dispositivo: {out.device}")
        return out

class ResNet3D(nn.Module):
    def __init__(self, input_channels: int = 1, num_classes: int = 1):
        super().__init__()
        self.initial_block = nn.Sequential(
            nn.Conv3d(input_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=1, padding=1)
        )
        self.layer1 = ResidualBlock(16, 32)
        self.layer2 = ResidualBlock(32, 64, stride=2)
        self.layer3 = ResidualBlock(64, 128, stride=2)
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128, 256)
        self.relu = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, num_classes)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        logger.debug(f"Entrada al modelo: {x.shape}, dispositivo: {x.device}")
        try:
            x = self.initial_block(x)
            logger.debug(f"Después de initial_block: {x.shape}, dispositivo: {x.device}")
            x = self.layer1(x)
            logger.debug(f"Después de layer1: {x.shape}, dispositivo: {x.device}")
            x = self.layer2(x)
            logger.debug(f"Después de layer2: {x.shape}, dispositivo: {x.device}")
            x = self.layer3(x)
            logger.debug(f"Después de layer3: {x.shape}, dispositivo: {x.device}")
            x = self.global_pool(x)
            logger.debug(f"Después de global_pool: {x.shape}, dispositivo: {x.device}")
            x = x.view(x.size(0), -1)
            logger.debug(f"Después de flatten: {x.shape}, dispositivo: {x.device}")
            x = self.dropout1(x)
            x = self.fc1(x)
            logger.debug(f"Después de fc1: {x.shape}, dispositivo: {x.device}")
            x = self.relu(x)
            x = self.dropout2(x)
            x = self.fc2(x)
            logger.debug(f"Después de fc2: {x.shape}, dispositivo: {x.device}")
            x = self.sigmoid(x)
            logger.debug(f"Salida final: {x.shape}, dispositivo: {x.device}")
            return x
        except Exception as e:
            logger.error(f"Error en ResNet3D.forward: {str(e)}")
            raise

def initialize_model_weights(model: nn.Module):
    """Inicializa pesos del modelo en el dispositivo correcto"""
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            m.weight.data = m.weight.data.to(device)  # Asegurar que los pesos estén en device
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
    logger.info(f"Creando ResNet3D: input_channels={input_channels}, num_classes={num_classes}")
    
    # Crear modelo y moverlo a device inmediatamente
    model = ResNet3D(input_channels, num_classes).to(device)
    logger.info(f"Modelo creado en dispositivo: {next(model.parameters()).device}")
    
    # Inicializar pesos en el dispositivo correcto
    initialize_model_weights(model)
    
    # Probar modelo con entrada de prueba
    test_input = torch.randn(1, input_channels, 128, 224, 224).to(device)
    logger.debug(f"Probando modelo con entrada: {test_input.shape}, dispositivo: {test_input.device}")
    
    model.eval()
    with torch.no_grad():
        try:
            test_output = model(test_input)
            logger.debug(f"Salida del modelo: {test_output.shape}, dispositivo: {test_output.device}")
            if test_output.shape != torch.Size([1, num_classes]):
                logger.error(f"Forma de salida inesperada: {test_output.shape}, esperado: [1, {num_classes}]")
                raise ValueError("Forma de salida del modelo incorrecta")
        except Exception as e:
            logger.error(f"Error al probar modelo: {str(e)}")
            raise
    
    model.train()
    return model

# --- TRANSFORMACIONES ---

def create_train_transforms() -> Compose:
    return Compose([
        RandFlipD(keys="image", spatial_axis=2, prob=0.5),
        RandRotateD(keys="image", range_x=20, prob=0.5),
        RandAdjustContrastD(keys="image", gamma=(0.8, 1.2), prob=0.5),
        RandGaussianNoiseD(keys="image", std=0.02, prob=0.5)
    ])

def create_val_transforms() -> Compose:
    return Compose([])

# --- MÉTRICAS Y VISUALIZACIÓN ---

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
    plt.savefig("logs/training_curves.png", dpi=300, bbox_inches='tight')
    plt.show()

def calculate_confusion_matrix(val_df: pl.DataFrame, model: nn.Module, val_loader: DataLoader, device: torch.device):
    model.eval()
    predictions, labels = [], []
    with torch.no_grad():
        for images, batch_labels in val_loader:
            images = images.to(device)
            outputs = model(images).squeeze(-1)
            predictions.extend(outputs.cpu().numpy())
            labels.extend(batch_labels.numpy())
    
    pred_binary = (np.array(predictions) > 0.5).astype(int)
    true_binary = np.array(labels).astype(int)
    cm = confusion_matrix(true_binary, pred_binary)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusión')
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.savefig("logs/confusion_matrix.png", dpi=300)
    plt.show()
    f1_val = f1_score(true_binary, pred_binary)
    logger.info(f"F1-Score validación: {f1_val:.4f}")
    return f1_val

# --- ENTRENAMIENTO ---

def create_optimizer(model: nn.Module, learning_rate: float) -> optim.Optimizer:
    return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

def create_criterion(class_weights: torch.Tensor) -> nn.Module:
    return nn.BCELoss(reduction='none')

def create_metrics(device: torch.device):
    return {
        'auc': BinaryAUROC().to(device),
        'precision': BinaryPrecision().to(device),
        'recall': BinaryRecall().to(device),
        'accuracy': BinaryAccuracy().to(device),
        'f1': BinaryF1Score().to(device)
    }

def train_epoch(model: nn.Module, train_loader: DataLoader, optimizer: optim.Optimizer,
                criterion: nn.Module, class_weights: torch.Tensor, device: torch.device,
                metrics: Dict[str, nn.Module], epoch: int, scaler: Optional[GradScaler] = None) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    logger.info(f"Iniciando época {epoch}, total de lotes esperados: {len(train_loader)}")
    logger.info(f"Modelo en dispositivo: {next(model.parameters()).device}")
    
    for metric in metrics.values():
        metric.reset()
    
    try:
        for batch_idx, (images, labels) in enumerate(train_loader):
            logger.info(f"Procesando lote {batch_idx}: imágenes {images.shape}, etiquetas {labels.tolist()}, "
                       f"dispositivo imágenes: {images.device}, dispositivo etiquetas: {labels.device}")
            
            images, labels = images.to(device), labels.to(device)
            #logger.info(f"Memoria antes de forward: {'GPU: ' + str(torch.cuda.memory_allocated()/1e9) + ' GB' if device.type == 'cuda' else 'RAM: ' + str(psutil.virtual_memory().used / 1e9) + ' GB'}")
            logger.info(f"Memoria antes de forward: RAM: {psutil.virtual_memory().used / 1e9:.2f} GB")

            optimizer.zero_grad(set_to_none=True)
            
            try:
                logger.debug(f"Ejecutando model.forward...")
                outputs = model(images)
                logger.debug(f"Salida del modelo: {outputs.shape}")
                outputs = outputs.squeeze(-1)
                logger.debug(f"Salida después de squeeze: {outputs.shape}")
                loss = criterion(outputs, labels)
                loss = (loss * class_weights[labels.long()]).mean()
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                for name, metric in metrics.items():
                    metric.update(outputs, labels.int())
                
                logger.info(f"Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
                logger.info(f"Memoria después de forward: RAM: {psutil.virtual_memory().used / 1e9:.2f} GB")
                
                del images, labels, outputs, loss
                gc.collect()
            
            except Exception as e:
                logger.error(f"Error en lote {batch_idx} durante forward/backward: {str(e)}")
                raise
        
    except Exception as e:
        logger.error(f"Error en train_epoch, lote {batch_idx}: {str(e)}")
        raise
    
    epoch_metrics = {'loss': total_loss / num_batches if num_batches > 0 else 0.0}
    for name, metric in metrics.items():
        epoch_metrics[name] = metric.compute().item()
    
    logger.info(f"Época {epoch} completada: {num_batches} lotes procesados, pérdida promedio: {epoch_metrics['loss']:.4f}")
    return epoch_metrics

def validate_epoch(model: nn.Module, val_loader: DataLoader, criterion: nn.Module,
                  device: torch.device, metrics: Dict[str, nn.Module]) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    for metric in metrics.values():
        metric.reset()
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).squeeze(-1)
            loss = criterion(outputs, labels).mean()
            
            total_loss += loss.item()
            num_batches += 1
            for name, metric in metrics.items():
                metric.update(outputs, labels.int())
    
    epoch_metrics = {'loss': total_loss / num_batches}
    for name, metric in metrics.items():
        epoch_metrics[name] = metric.compute().item()
    
    return epoch_metrics

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
    logger.info("Probando modelo con lote de ejemplo...")
    test_images = torch.randn(1, 1, 128, 224, 224).to(device)
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
    del model, test_images, test_labels
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

def pretrain_model():
    logger.info("=== INICIANDO PREENTRENAMIENTO 3D-CNN (PyTorch) ===")

    # Verificar dispositivo del modelo
    logger.info("5. Creando modelo ResNet3D...")
    model = build_resnet3d_model(input_channels=1, num_classes=1)
    logger.info(f"Modelo en dispositivo: {next(model.parameters()).device}")

    test_model(device)    
    
    logger.info("1. Cargando metadatos...")
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
    logger.info(f"Estudios en train_df: {train_df['StudyInstanceUID'].to_list()}")
    
    train_pos = len(train_df.filter(pl.col("label") == 1))
    train_neg = len(train_df) - train_pos
    logger.info(f"   Clases train: {train_pos} positivos ({train_pos/len(train_df)*100:.1f}%), "
               f"{train_neg} negativos ({train_neg/len(train_df)*100:.1f}%)")
    
    logger.info("3. Calculando pesos de clase...")
    n_total, n_neg, n_pos = len(train_df), train_neg, train_pos
    class_weights = torch.tensor([
        (1 / n_neg) * (n_total / 2.0) if n_neg > 0 else 1.0,
        (1 / n_pos) * (n_total / 2.0) if n_pos > 0 else 1.0
    ], device=device)
    logger.info(f"✅ Pesos: clase 0={class_weights[0]:.2f}, clase 1={class_weights[1]:.2f}")

    logger.info("4. Creando datasets...")
    train_dataset = RSNADataset(train_df)
    val_dataset = create_validation_dataset(val_df)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=False)
    
    logger.info(f"Tamaño de train_dataset: {len(train_dataset)}")
    logger.info(f"Batch size: {config.BATCH_SIZE}")
    logger.info(f"Esperado número de lotes: {len(train_loader)}")
    
    logger.info("5. Creando modelo ResNet3D...")
    model = build_resnet3d_model(input_channels=1, num_classes=1).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"✅ Modelo creado: {total_params/1e6:.1f}M params total, {trainable_params/1e6:.1f}M entrenables")
    
    optimizer = create_optimizer(model, config.LEARNING_RATE)
    criterion = create_criterion(class_weights)
    metrics = create_metrics(device)
    
    use_mixed_precision = False
    scaler = GradScaler() if use_mixed_precision else None
    logger.info(f"✅ Optimizador: Adam(lr={config.LEARNING_RATE}), Mixed precision: {use_mixed_precision}")
    
    history = {
        'loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'auc': [], 'f1': [],
        'val_loss': [], 'val_accuracy': [], 'val_precision': [], 'val_recall': [], 'val_auc': [], 'val_f1': []
    }
    
    best_val_auc = 0.0
    patience = 5
    no_improve_count = 0
    
    logger.info(f"🎯 Iniciando entrenamiento por {config.EPOCHS} épocas...")
    for epoch in range(config.EPOCHS):
        logger.info(f"\n--- ÉPOCA {epoch+1}/{config.EPOCHS} ---")
        
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, 
            class_weights, device, metrics, epoch+1, scaler
        )
        
        val_metrics = validate_epoch(model, val_loader, criterion, device, metrics)
        
        for key in ['loss', 'accuracy', 'precision', 'recall', 'auc', 'f1']:
            history[key].append(train_metrics[key])
            history[f'val_{key}'].append(val_metrics[key])
        
        logger.info(f"RESULTADOS ÉPOCA {epoch+1}:")
        logger.info(f"  Train - Loss: {train_metrics['loss']:.4f}, AUC: {train_metrics['auc']:.4f}, "
                   f"Acc: {train_metrics['accuracy']:.3f}, F1: {train_metrics['f1']:.3f}")
        logger.info(f"  Valid - Loss: {val_metrics['loss']:.4f}, AUC: {val_metrics['auc']:.4f}, "
                   f"Acc: {val_metrics['accuracy']:.3f}, F1: {val_metrics['f1']:.3f}")
        
        current_val_auc = val_metrics['auc']
        if current_val_auc > best_val_auc:
            best_val_auc = current_val_auc
            save_model_checkpoint(model, 'models/best_model_auc.pth', best_val_auc, epoch+1)
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
    
    logger.info(f"\n=== ENTRENAMIENTO COMPLETADO ===")
    logger.info(f"Mejor AUC validación: {best_val_auc:.4f}")
    
    final_model_path = 'models/pretrained_rsna_final.pth'
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