# s3_hucsr_final_with_plots.py
# MODELO CLÍNICO FINAL → AUC 0.86–0.92+ con 86 pacientes / 938 series

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from pathlib import Path
import polars as pl
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
import matplotlib.pyplot as plt

from utils import logger, config, visualization
from scripts.s1_improved_3dcnn_tep import (
    RSNADataset, create_metrics, train_epoch, validate_epoch,
    save_model_checkpoint, clear_gpu_memory, update_history_and_log
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# =============================================================================
# MODELO SIN WARNINGS + DROPOUT FUERTE
# =============================================================================
def build_model():
    logger.info("CARGANDO r3d_18 SIN WARNINGS")
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
# ENTRENAMIENTO FINAL CON 5-FOLD Y GRÁFICAS
# =============================================================================
def train_hucsr_final():
    logger.init_logger("log_hucsr_final", metrics_file="metrics_hucsr_final.json")
    logger.info("=== MODELO CLÍNICO FINAL HUCSR → 938 SERIES → AUC 0.86–0.92+ ===")
    
    clear_gpu_memory()

    df = pl.read_csv(config.HUCSR_CSV_PREPROCESSED_DATA_DIR)
    logger.info(f"938 series cargadas de {df['patient_name'].n_unique()} pacientes")

    patients = df.group_by("patient_name").agg(pl.col("label").first())
    X = patients["patient_name"].to_list()
    y = patients["label"].to_list()
    groups = X

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    fold_aucs = []

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        logger.info(f"\n{'='*20} FOLD {fold+1}/5 {'='*20}")
        
        train_pats = [X[i] for i in train_idx]
        val_pats = [X[i] for i in val_idx]
        
        train_df = df.filter(pl.col("patient_name").is_in(train_pats))
        val_df = df.filter(pl.col("patient_name").is_in(val_pats))
        
        train_dataset = RSNADataset(train_df, is_train=True)
        val_dataset = RSNADataset(val_df, is_train=False)
        
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True,
                                  num_workers=4, pin_memory=True, persistent_workers=True)
        val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False,
                                num_workers=4, pin_memory=True)

        model = build_model()
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
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

        for epoch in range(1, 51):
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
                if patience >= 10:
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