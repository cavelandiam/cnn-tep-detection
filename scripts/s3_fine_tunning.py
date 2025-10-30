# scripts/finetune_3dcnn_hucsr.py
import gc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score
import polars as pl
from sklearn.model_selection import train_test_split

from utils import logger, config, visualization
from scripts.s1_improved_3dcnn_tep import (
    RSNADataset, ResNet3D, build_resnet3d_model, initialize_model_weights,
    create_optimizer, create_criterion, create_metrics,
    train_epoch, validate_epoch, save_model_checkpoint,
    plot_training_curves, calculate_confusion_matrix, clear_gpu_memory
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def finetune_model():
    logger.init_logger("log_finetune_hucsr", metrics_file="metrics_finetune.json")
    logger.info("=== INICIANDO FINE-TUNING CON HUCSR ===")

    clear_gpu_memory()

    # Cargar metadata HUCSR
    metadata_path = config.HUCSR_CSV_PREPROCESSED_DATA_DIR
    if not Path(metadata_path).exists():
        raise FileNotFoundError(f"No existe {metadata_path}. Ejecuta process_hucsr_data.py primero.")
    
    df = pl.read_csv(metadata_path)
    logger.info(f"Metadata HUCSR cargado: {len(df)} series")

    # Split estratificado por paciente
    unique_patients = df.group_by("patient_name").agg(pl.col("label").first())
    train_patients, val_patients = train_test_split(
        unique_patients["patient_name"].to_list(),
        test_size=0.2,
        random_state=42,
        stratify=unique_patients["label"]
    )
    train_df = df.filter(pl.col("patient_name").is_in(train_patients))
    val_df = df.filter(pl.col("patient_name").is_in(val_patients))

    logger.info(f"Train: {len(train_df)} series ({train_df['label'].sum()} pos), "
                f"Val: {len(val_df)} series ({val_df['label'].sum()} pos)")

    # Datasets
    train_dataset = RSNADataset(train_df, is_train=True)
    val_dataset = RSNADataset(val_df, is_train=False)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE//2, shuffle=True,
                              num_workers=config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE//2, shuffle=False,
                            num_workers=config.NUM_WORKERS, pin_memory=True)

    # Modelo preentrenado
    model = build_resnet3d_model().to(device)
    pretrained_path = config.RSNA_PRETRAINED_MODEL
    model.load_state_dict(torch.load(pretrained_path, map_location=device))
    logger.info(f"Modelo RSNA cargado desde {pretrained_path}")

    # Congelar capas iniciales
    for name, param in model.named_parameters():
        if 'fc' not in name and 'layer3' not in name:
            param.requires_grad = False
    logger.info("Capas iniciales congeladas. Entrenando layer3 + FC")

    # Optimizador solo en parámetros entrenables
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    class_weights = torch.tensor([1.0, 3.0], device=device)  # Ajustar si imbalance
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[1])
    scaler = GradScaler() if device.type == 'cuda' else None
    metrics = create_metrics(device)

    # Entrenamiento

    history = {k: [] for k in ['loss','accuracy','precision','recall','auc','f1','specificity',
                               'mcc','pr_auc','val_loss','val_accuracy','val_precision','val_recall',
                               'val_auc','val_f1','val_specificity','val_mcc','val_pr_auc']}
    best_val_auc = 0.0
    patience, no_improve = 5, 0

    for epoch in range(20):
        logger.info(f"\n--- ÉPOCA {epoch+1}/20 ---")
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, None, device, metrics, epoch+1, scaler)
        val_metrics = validate_epoch(model, val_loader, criterion, device, metrics)

        # Log
        for k in history.keys():
            if k.startswith('val_'):
                history[k].append(val_metrics.get(k[4:], 0))
            else:
                history[k].append(train_metrics.get(k, 0))

        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            save_model_checkpoint(model, config.HUCSR_FINETUNED_MODEL, best_val_auc, epoch+1)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info("Early stopping")
                break

    # Guardar final    
    torch.save(model.state_dict(), config.HUCSR_FINETUNED_MODEL)
    logger.info(f"Modelo final guardado: {config.HUCSR_FINETUNED_MODEL}")

    plot_training_curves(history, config.HUCSR_GRAPHS_METRICS_DIR)
    logger.info(f"Gráficas de métricas guardadas: {config.HUCSR_GRAPHS_METRICS_DIR}")

    f1 = calculate_confusion_matrix(val_df, model, val_loader, device, config.HUCSR_GRAPHS_CONFUSION_MATRIX_DIR)
    logger.info(f"F1 Final: {f1:.4f}, AUC: {best_val_auc:.4f}")

    visualization.plot_model_architecture(
        model=model,
        save_dir=config.HUCSR_GRAPHS_METRICS_DIR,
        model_name= config.HUCSR_GRAPHS_MODEL_NAME,
        input_size=(1, config.TARGET_DEPTH, *config.IMAGE_SIZE, 1)
    )
    logger.info(f"Arquitectura del modelo guardada en {config.HUCSR_GRAPHS_METRICS_DIR}")

    return history, best_val_auc