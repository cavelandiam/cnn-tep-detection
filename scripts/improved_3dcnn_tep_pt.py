import gc
import os
import random
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import polars as pl
import pydicom
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from monai.transforms import Compose, LoadImageD, ScaleIntensityRangeD, ResizeD, RandFlipD, RandRotateD, RandAdjustContrastD, RandGaussianNoiseD
from torchmetrics.classification import BinaryAUROC, BinaryPrecision, BinaryRecall, BinaryAccuracy, BinaryF1Score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import zoom

from utils import logger, config

# --- CONFIGURACIÓN PARA REPRODUCIBILIDAD ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False  # Para reproducibilidad, pero puede ser True para mejor rendimiento

# --- CONFIGURACIÓN DE LOGGING Y GPU ---
logger.init_logger("log_process_data_rsna")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Usando dispositivo: {device}")
if torch.cuda.is_available():
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}, Memoria: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# --- FUNCIONES DE PROCESAMIENTO DE DATOS ---
def _process_dicom_image(ds: pydicom.FileDataset) -> Optional[np.ndarray]:
    if ds.pixel_array.ndim != 2:
        return None
    
    img = ds.pixel_array.astype(np.float32)
    if 'RescaleSlope' in ds and 'RescaleIntercept' in ds:
        img = img * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
    
    window_center, window_width = -600, 1600
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img = np.clip(img, img_min, img_max)
    
    img = (img - img_min) / (img_max - img_min)
    return np.expand_dims(img, axis=-1)

def _resize_depth(volume: np.ndarray, target_depth: int) -> np.ndarray:
    current_depth = volume.shape[0]
    if current_depth < 1:
        logger.warning(f"Invalid volume depth {current_depth}, returning zeros")
        return np.zeros((target_depth, *volume.shape[1:]), dtype=np.float32)
    
    if current_depth == target_depth:
        return volume
    
    scale = target_depth / current_depth
    resized = zoom(volume, (scale, 1, 1, 1), order=1)
    if resized.shape != (target_depth, *volume.shape[1:]):
        logger.warning(f"Invalid resized shape {resized.shape}, returning zeros")
        return np.zeros((target_depth, *volume.shape[1:]), dtype=np.float32)
    
    return resized.astype(np.float32)

def _parse_dicom_study(study_path: Union[str, bytes]) -> np.ndarray:
    try:
        study_path_str = study_path.decode('utf-8') if isinstance(study_path, bytes) else study_path
        study_path_obj = Path(study_path_str)
        
        if not study_path_obj.exists() or not study_path_obj.is_dir():
            logger.warning(f"Study folder {study_path_str} does not exist or is not a directory")
            return np.zeros((config.TARGET_DEPTH, *config.IMAGE_SIZE, 1), dtype=np.float32)
        
        series_dirs = [p for p in study_path_obj.iterdir() if p.is_dir()]
        if not series_dirs:
            logger.warning(f"No series found for study {study_path_str}")
            return np.zeros((config.TARGET_DEPTH, *config.IMAGE_SIZE, 1), dtype=np.float32)
        
        series_dir = max(series_dirs, key=lambda p: len(list(p.glob('*.dcm'))))
        dicom_files = list(series_dir.glob('*.dcm'))
        if not dicom_files:
            logger.warning(f"No DICOM files found in {series_dir}")
            return np.zeros((config.TARGET_DEPTH, *config.IMAGE_SIZE, 1), dtype=np.float32)
        
        dicom_data = []
        for f in dicom_files:
            try:
                ds = pydicom.dcmread(str(f), force=True)
                if hasattr(ds, 'pixel_array') and ds.pixel_array.ndim == 2:
                    dicom_data.append(ds)
                else:
                    logger.warning(f"DICOM file {f} has no valid pixel data or incorrect dimensions")
            except Exception as e:
                logger.warning(f"Failed to read DICOM file {f}: {e}")
        
        if not dicom_data:
            logger.warning(f"No valid DICOM data in {study_path_str}")
            return np.zeros((config.TARGET_DEPTH, *config.IMAGE_SIZE, 1), dtype=np.float32)
        
        dicom_data.sort(key=lambda x: float(x.get('SliceLocation', 0.0)))
        volume = [_process_dicom_image(ds) for ds in dicom_data]
        volume = [img for img in volume if img is not None]
        
        if not volume:
            logger.warning(f"No valid images in {study_path_str}")
            return np.zeros((config.TARGET_DEPTH, *config.IMAGE_SIZE, 1), dtype=np.float32)
        
        volume_np = np.stack(volume, axis=0)
        logger.info(f"Original volume shape for {study_path_str}: {volume_np.shape}")
        
        if volume_np.shape[0] < 1:
            logger.warning(f"Volume has no valid slices in {study_path_str}")
            return np.zeros((config.TARGET_DEPTH, *config.IMAGE_SIZE, 1), dtype=np.float32)
        
        resized_volume = _resize_depth(volume_np, config.TARGET_DEPTH)
        logger.info(f"Resized volume shape for {study_path_str}: {resized_volume.shape}")
        
        if resized_volume.shape != (config.TARGET_DEPTH, *config.IMAGE_SIZE, 1):
            logger.warning(f"Invalid resized volume shape for {study_path_str}: {resized_volume.shape}")
            return np.zeros((config.TARGET_DEPTH, *config.IMAGE_SIZE, 1), dtype=np.float32)
        
        return resized_volume.astype(np.float32)
    
    except Exception as e:
        logger.error(f"Error processing study {study_path_str}: {e}")
        return np.zeros((config.TARGET_DEPTH, *config.IMAGE_SIZE, 1), dtype=np.float32)

class RSNADataset(Dataset):
    def __init__(self, df: pl.DataFrame, transform: Optional[Compose] = None):
        self.df = df
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        study_path = self.df["study_path"][idx]
        label = self.df["label"][idx]
        
        volume = _parse_dicom_study(study_path)
        volume = torch.from_numpy(volume).permute(3, 0, 1, 2)  # (C, D, H, W)
        
        if self.transform:
            volume = self.transform({"image": volume})["image"]
        
        return volume, torch.tensor(label, dtype=torch.float32)

# --- MODELO RESNET3D ---
class ResNet3D(nn.Module):
    def __init__(self, input_shape: Tuple[int, ...], dropout_rate1: float = 0.5, dropout_rate2: float = 0.3):
        super().__init__()
        
        def res_block(in_channels: int, out_channels: int, kernel_size: Tuple[int, ...] = (3, 3, 3), stride: int = 1):
            layers = []
            shortcut = nn.Sequential()
            if stride > 1 or in_channels != out_channels:
                shortcut = nn.Sequential(
                    nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                    nn.BatchNorm3d(out_channels)
                )
            
            layers.extend([
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=1, bias=False),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_channels, out_channels, kernel_size, stride=1, padding=1, bias=False),
                nn.BatchNorm3d(out_channels)
            ])
            
            return nn.Sequential(*layers), shortcut
        
        self.initial_conv = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=1, padding=1)
        )
        
        self.block1, self.shortcut1 = res_block(16, 32)
        self.block2, self.shortcut2 = res_block(32, 64, stride=2)
        self.block3, self.shortcut3 = res_block(64, 128, stride=2)
        
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.dropout1 = nn.Dropout(dropout_rate1)
        self.fc1 = nn.Linear(128, 256)
        self.relu = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(dropout_rate2)
        self.fc2 = nn.Linear(256, 1)
        
        # Inicialización de pesos
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial_conv(x)
        
        residual = x
        x = self.block1(x)
        x = x + self.shortcut1(residual)
        x = self.relu(x)
        
        residual = x
        x = self.block2(x)
        x = x + self.shortcut2(residual)
        x = self.relu(x)
        
        residual = x
        x = self.block3(x)
        x = x + self.shortcut3(residual)
        x = self.relu(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return torch.sigmoid(x)

# --- MÉTRICAS Y VISUALIZACIÓN ---
def plot_training_curves_improved(history: Dict[str, list]):
    metrics = ['loss', 'accuracy', 'precision', 'recall', 'auc', 'f1']
    plt.figure(figsize=(18, 12))
    for i, metric in enumerate(metrics):
        plt.subplot(2, 3, i+1)
        plt.plot(history[metric], label=f'Train {metric}', color='royalblue')
        plt.plot(history[f'val_{metric}'], label=f'Val {metric}', color='orangered', linestyle='--')
        plt.title(f'Evolución de {metric.capitalize()}')
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig("logs/training_curves_improved.png", dpi=300)
    plt.show()

def plot_confusion_matrix(val_df: pl.DataFrame, model: nn.Module, val_loader: DataLoader, device: torch.device):
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predictions.extend(outputs.cpu().numpy().flatten())
            true_labels.extend(labels.cpu().numpy())
    
    pred_labels = (np.array(predictions) > 0.5).astype(int)
    cm = confusion_matrix(true_labels, pred_labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusión')
    plt.xlabel('Predicción')
    plt.ylabel('Verdadero')
    plt.savefig("logs/confusion_matrix.png")
    plt.show()
    
    f1 = BinaryF1Score().compute(torch.tensor(predictions), torch.tensor(true_labels)).item()
    logger.info(f"F1-Score en validación: {f1:.4f}")

# --- ENTRENAMIENTO ---
def pretrain_model():
    logger.info("Iniciando preentrenamiento optimizado del modelo 3D-CNN con PyTorch.")
    
    # Cargar datos
    df = pl.read_csv(config.RSNA_CSV_TRAIN_DIR)
    studies_df = df.group_by("StudyInstanceUID").agg(
        pl.col("negative_exam_for_pe").first()
    ).with_columns(
        pl.col("StudyInstanceUID").map_elements(
            lambda uid: str(Path(config.RSNA_DATASET_TRAIN_DIR) / uid), 
            return_dtype=pl.String
        ).alias("study_path"),
        (1 - pl.col("negative_exam_for_pe")).alias("label")
    )
    
    studies_df = studies_df.filter(
        pl.col("study_path").map_elements(
            lambda p: Path(p).exists() and Path(p).is_dir(), 
            return_dtype=pl.Boolean
        )
    )
    logger.info(f"Metadatos cargados para {len(studies_df)} estudios con carpetas existentes.")
    if len(studies_df) == 0:
        logger.error("No se encontraron carpetas de estudios válidas. Verifica RSNA_DATASET_TRAIN_DIR.")
        raise ValueError("No hay estudios válidos para procesar.")
    
    missing_studies = df.filter(
        ~df["StudyInstanceUID"].is_in(studies_df["StudyInstanceUID"])
    )["StudyInstanceUID"].to_list()
    if missing_studies:
        logger.warning(f"Se omitieron {len(missing_studies)} estudios porque sus carpetas no existen: {missing_studies[:10]}...")
    
    train_df, val_df = train_test_split(
        studies_df.to_pandas(),  # Convert to pandas for sklearn compatibility
        test_size=0.2,
        random_state=SEED,
        stratify=studies_df["label"]
    )
    train_df = pl.from_pandas(train_df)
    val_df = pl.from_pandas(val_df)
    logger.info(f"Datos divididos: {len(train_df)} para entrenamiento, {len(val_df)} para validación.")
    logger.info(f"Estudios de entrenamiento: {len(train_df)}, Estudios de validación: {len(val_df)}")
    
    # Calcular pesos de clase
    n_total = len(train_df)
    n_neg = len(train_df.filter(pl.col("label") == 0))
    n_pos = n_total - n_neg
    class_weight = torch.tensor([
        (1 / n_neg) * (n_total / 2.0) if n_neg > 0 else 1.0,
        (1 / n_pos) * (n_total / 2.0) if n_pos > 0 else 1.0
    ]).to(device)
    logger.info(f"Pesos de clase calculados -> 0: {class_weight[0]:.2f}, 1: {class_weight[1]:.2f}")
    
    # Transformaciones
    train_transforms = Compose([
        RandFlipD(keys="image", spatial_axis=2, prob=0.5),
        RandRotateD(keys="image", range_x=20.0, prob=0.5),
        RandAdjustContrastD(keys="image", gamma=(0.8, 1.2), prob=0.5),
        RandGaussianNoiseD(keys="image", prob=0.5, std=0.02)
    ])
    val_transforms = Compose([])
    
    # Datasets y DataLoaders
    train_dataset = RSNADataset(train_df, transform=train_transforms)
    val_dataset = RSNADataset(val_df, transform=val_transforms)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    # Modelo
    model = ResNet3D(input_shape=(1, config.TARGET_DEPTH, *config.IMAGE_SIZE)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.BCELoss(reduction='none')
    scaler = GradScaler()
    
    # Métricas
    auc_metric = BinaryAUROC().to(device)
    precision_metric = BinaryPrecision().to(device)
    recall_metric = BinaryRecall().to(device)
    accuracy_metric = BinaryAccuracy().to(device)
    f1_metric = BinaryF1Score().to(device)
    
    # Callbacks
    best_val_auc = 0.0
    patience = 5
    reduce_lr_counter = 0
    history = {'loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'auc': [], 'f1': [],
               'val_loss': [], 'val_accuracy': [], 'val_precision': [], 'val_recall': [], 'val_auc': [], 'val_f1': []}
    
    logger.info("Iniciando el entrenamiento del modelo...")
    for epoch in range(config.EPOCHS):
        model.train()
        train_loss, train_auc, train_precision, train_recall, train_accuracy, train_f1 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            with autocast():
                outputs = model(images).squeeze()
                loss = criterion(outputs, labels)
                loss = (loss * class_weight[labels.long()]).mean()
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            auc_metric.update(outputs, labels)
            precision_metric.update(outputs, labels)
            recall_metric.update(outputs, labels)
            accuracy_metric.update(outputs, labels)
            f1_metric.update(outputs, labels)
            
            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{config.EPOCHS}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        train_loss /= len(train_loader)
        train_auc = auc_metric.compute().item()
        train_precision = precision_metric.compute().item()
        train_recall = recall_metric.compute().item()
        train_accuracy = accuracy_metric.compute().item()
        train_f1 = f1_metric.compute().item()
        
        # Validación
        model.eval()
        val_loss, val_auc, val_precision, val_recall, val_accuracy, val_f1 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                with autocast():
                    outputs = model(images).squeeze()
                    loss = criterion(outputs, labels).mean()
                
                val_loss += loss.item()
                auc_metric.update(outputs, labels)
                precision_metric.update(outputs, labels)
                recall_metric.update(outputs, labels)
                accuracy_metric.update(outputs, labels)
                f1_metric.update(outputs, labels)
        
        val_loss /= len(val_loader)
        val_auc = auc_metric.compute().item()
        val_precision = precision_metric.compute().item()
        val_recall = recall_metric.compute().item()
        val_accuracy = accuracy_metric.compute().item()
        val_f1 = f1_metric.compute().item()
        
        # Guardar métricas
        history['loss'].append(train_loss)
        history['accuracy'].append(train_accuracy)
        history['precision'].append(train_precision)
        history['recall'].append(train_recall)
        history['auc'].append(train_auc)
        history['f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_auc'].append(val_auc)
        history['val_f1'].append(val_f1)
        
        logger.info(f"Epoch {epoch+1}/{config.EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                    f"Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}")
        
        # Checkpoint
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), 'models/best_model_auc.pth')
            logger.info(f"Guardado mejor modelo con val_auc: {val_auc:.4f}")
            reduce_lr_counter = 0
        else:
            reduce_lr_counter += 1
        
        # ReduceLROnPlateau
        if reduce_lr_counter >= patience:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.2
                if param_group['lr'] < 1e-6:
                    param_group['lr'] = 1e-6
                logger.info(f"Reduciendo LR a {param_group['lr']:.6f}")
            reduce_lr_counter = 0
        
        # Reset métricas
        auc_metric.reset()
        precision_metric.reset()
        recall_metric.reset()
        accuracy_metric.reset()
        f1_metric.reset()
    
    # Guardar modelo final
    torch.save(model.state_dict(), 'models/pretrained_rsna_final.pth')
    plot_training_curves_improved(history)
    plot_confusion_matrix(val_df, model, val_loader, device)
    
    logger.info("Entrenamiento completado exitosamente.")
    
    del model, train_loader, val_loader
    gc.collect()
    torch.cuda.empty_cache()