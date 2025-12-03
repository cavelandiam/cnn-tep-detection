import gc
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler
from pathlib import Path
import polars as pl
import numpy as np
from sklearn.metrics import auc
from sklearn.model_selection import StratifiedGroupKFold
import matplotlib.pyplot as plt
from monai.transforms import Compose, RandFlipD, RandRotateD, RandAdjustContrastD, RandGaussianNoiseD
from torchmetrics.classification import BinaryAUROC, BinaryPrecision, BinaryRecall, BinaryAccuracy, BinaryF1Score, BinarySpecificity, MatthewsCorrCoef, PrecisionRecallCurve

from utils import logger, config, visualization

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def clear_gpu_memory():
    """Limpia memoria GPU antes de empezar entrenamiento"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    logger.info("🧹 GPU Memory Cleared")

def log_gpu_memory():
    """Monitorea y registra uso de memoria GPU"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        max_allocated = torch.cuda.max_memory_allocated() / 1e9
        logger.info(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Max: {max_allocated:.2f}GB")

# =============================================================================
# MODELO SIN WARNINGS + DROPOUT FUERTE
# =============================================================================
def build_model():
    logger.info("CARGANDO r3d_18 PREENTRENADO EN KINETICS-400")
    from torchvision.models.video import R3D_18_Weights
    model = torch.hub.load('pytorch/vision', 'r3d_18', weights=R3D_18_Weights.KINETICS400_V1)
    
    old = model.stem[0]
    new = nn.Conv3d(1, 64, kernel_size=old.kernel_size, stride=old.stride,
                    padding=old.padding, bias=False)
    with torch.no_grad():
        new.weight = nn.Parameter(old.weight.mean(dim=1, keepdim=True))
    model.stem[0] = new
    
    model.fc = nn.Sequential(
        nn.Dropout(0.6),
        nn.Linear(model.fc.in_features, 1)
    )
    return model.to(device)

# =============================================================================
# GRÁFICAS COMPLETAS
# =============================================================================
def plot_all_metrics(history, fold, save_dir):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    metrics = ['loss','accuracy','precision','recall','f1','auc','specificity','mcc','pr_auc']
    
    plt.figure(figsize=(15, 10))
    for i, metric in enumerate(metrics, 1):
        plt.subplot(3, 3, i)
        if metric in history and f'val_{metric}' in history:
            epochs = range(1, len(history[metric]) + 1)
            plt.plot(epochs, history[metric], 'b-', label='Train', linewidth=2)
            plt.plot(epochs, history[f'val_{metric}'], 'r--', label='Valid', linewidth=2)
            plt.title(f'{metric.upper()} - Fold {fold}')
            plt.xlabel('Época')
            plt.ylabel(metric.capitalize())
            plt.legend()
            plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/fold_{fold}_all_metrics.png", dpi=300, bbox_inches='tight')
    plt.close()


# =============================================================================
# DATASET
# =============================================================================

class HucsrDataset(Dataset):
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





def train_epoch(model, data_loader, optimizer, criterion, class_weights, device, metrics, epoch, scaler=None):
    """Entrenamiento optimizado para memoria GPU"""
    model.train()
    running_loss = 0.0
    total_batches = len(data_loader)
    all_preds = []
    all_labels = []
    
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

def save_model_checkpoint(model: nn.Module, path: str, val_auc: float, epoch: int):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'val_auc': val_auc,
        'epoch': epoch,
    }, path)
    logger.info(f"Checkpoint guardado: {path} (AUC: {val_auc:.4f}, Epoch: {epoch})")


# =============================================================================
# ENTRENAMIENTO FINAL CON 5-FOLD Y GRÁFICAS
# =============================================================================
def train_hucsr():
    logger.init_logger("log_hucsr_create_model", metrics_file="metrics_hucsr_create_model.json")
    logger.info("=== MODELO CLÍNICO FINAL HUCSR ===")
    
    clear_gpu_memory()

    df = pl.read_csv(config.HUCSR_CSV_PREPROCESSED_DATA_DIR)
    logger.info(f"Cargados {df['patient_name'].n_unique()} pacientes")

    patients = df.group_by("patient_name").agg(pl.col("label").first())
    X = patients["patient_name"].to_list()
    y = patients["label"].to_list()
    groups = X

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    fold_aucs = []

    visualization.plot_model_architecture(
        model= build_model(),
        save_dir=config.RSNA_GRAPHS_DIR,
        model_name= config.RSNA_GRAPHS_MODEL_NAME,
        input_size=(1, config.TARGET_DEPTH, *config.IMAGE_SIZE)
    )

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        logger.info(f"\n{'='*20} FOLD {fold+1}/5 {'='*20}")
        
        train_pats = [X[i] for i in train_idx]
        val_pats = [X[i] for i in val_idx]
        
        train_df = df.filter(pl.col("patient_name").is_in(train_pats))
        val_df = df.filter(pl.col("patient_name").is_in(val_pats))
        
        train_dataset = HucsrDataset(train_df, is_train=True)
        val_dataset = HucsrDataset(val_df, is_train=False)
        
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
                                  num_workers=config.NUM_WORKERS, pin_memory=True, persistent_workers=True)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                                num_workers=config.NUM_WORKERS, pin_memory=True)

        model = build_model()
        optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.0, device=device))
        scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
        metrics = create_metrics(device)

        history = {k: [] for k in [
            'loss','accuracy','precision','recall','auc','f1','specificity','mcc','pr_auc',
            'val_loss','val_accuracy','val_precision','val_recall','val_auc','val_f1',
            'val_specificity','val_mcc','val_pr_auc'
        ]}
        best_auc = 0.0
        patience = 0

        for epoch in range(1, config.EPOCHS):
            train_metrics = train_epoch(model, train_loader, optimizer, criterion, None, device, metrics, epoch, scaler)
            val_metrics = validate_epoch(model, val_loader, criterion, device, metrics)

            # Guardar métricas
            for k in ['loss','accuracy','precision','recall','auc','f1','specificity','mcc','pr_auc']:
                history[k].append(train_metrics.get(k, 0.0))
                history[f'val_{k}'].append(val_metrics.get(k, 0.0))

            update_history_and_log(epoch, train_metrics, val_metrics, history)

            if val_metrics['auc'] > best_auc:
                best_auc = val_metrics['auc']
                save_model_checkpoint(model, f"{config.HUCSR_FINETUNED_MODEL}_fold{fold+1}.pth", best_auc, epoch)
                patience = 0
                logger.info(f"MEJOR FOLD {fold+1} → AUC: {best_auc:.4f}")
            else:
                patience += 1
                if patience >= config.PATIENCE_EARLY_STOPPING:
                    logger.info("Early stopping")
                    break

            # GRÁFICAS POR FOLD
            plot_all_metrics(history, fold+1, f"{config.HUCSR_GRAPHS_METRICS_DIR}/fold_{fold+1}")

        fold_aucs.append(best_auc)

    mean_auc = np.mean(fold_aucs)
    std_auc = np.std(fold_aucs)
    logger.info(f"\n5-FOLD CV COMPLETO → Media AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    logger.info(f"GRÁFICAS GUARDADAS EN: {config.HUCSR_GRAPHS_METRICS_DIR}")
    logger.info("MODELO CLÍNICO FINAL LISTO")

    return mean_auc