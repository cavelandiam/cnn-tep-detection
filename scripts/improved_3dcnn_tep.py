# Imports optimizados y organizados
import os
import gc
import logging
import warnings
from pathlib import Path

import numpy as np
import polars as pl
import pydicom
import tensorflow as tf
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from scipy.ndimage import zoom
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv3D, BatchNormalization, Activation, 
                                    MaxPooling3D, Add, GlobalAveragePooling3D, 
                                    Dropout, Dense)

# Configuración desde utils.config
from utils.config import IMAGE_SIZE, TARGET_DEPTH, RSNA_CSV_TRAIN_DIR, RSNA_DATASET_TRAIN_DIR, BATCH_SIZE, EPOCHS, LEARNING_RATE

# --- CONFIGURACIÓN DE TENSORFLOW Y LOGGING ---

print("Versión de TensorFlow:", tf.__version__)
print("Dispositivos físicos:", tf.config.list_physical_devices('GPU'))
print("Dispositivos lógicos:", tf.config.list_logical_devices('GPU'))

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logging.info(f"Crecimiento de memoria habilitado para {len(gpus)} GPU(s).")
    except RuntimeError as e:
        logging.error(f"Error al configurar la memoria de la GPU: {e}")

tf.keras.mixed_precision.set_global_policy('mixed_float16')
tf.debugging.set_log_device_placement(True)
logging.info("Política de precisión mixta de Keras establecida en 'mixed_float16'.")

class IgnoreInvalidVRUIFilter(logging.Filter):
    def filter(self, record):
        return "Invalid value for VR UI" not in record.getMessage()

logging.getLogger("pydicom").addFilter(IgnoreInvalidVRUIFilter())
warnings.filterwarnings('ignore', category=UserWarning, message='Invalid value for VR UI')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/train_rsna.log'),
        logging.StreamHandler()
    ]
)

# --- FUNCIONES DE PROCESAMIENTO DE DATOS ---

def _process_dicom_image(ds: pydicom.FileDataset):
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
    img_resized = resize(img, IMAGE_SIZE, anti_aliasing=True)
    return np.expand_dims(img_resized, axis=-1)

def _resize_depth(volume, target_depth):
    current_depth = volume.shape[0]
    if current_depth < 1:
        logging.warning(f"Invalid volume depth {current_depth}, returning zeros")
        return np.zeros((target_depth, *volume.shape[1:]), dtype=np.float32)
    
    if current_depth == target_depth:
        return volume
    
    scale = target_depth / current_depth
    resized = zoom(volume, (scale, 1, 1, 1), order=1)
    if resized.shape != (target_depth, *volume.shape[1:]):
        logging.warning(f"Invalid resized shape {resized.shape}, returning zeros")
        return np.zeros((target_depth, *volume.shape[1:]), dtype=np.float32)
    
    return resized.astype(np.float32)

def _parse_dicom_study(study_path: bytes):
    try:
        study_path_str = study_path.decode('utf-8') if isinstance(study_path, bytes) else study_path.numpy().decode('utf-8')
        study_path_obj = Path(study_path_str)
        
        if not study_path_obj.exists() or not study_path_obj.is_dir():
            logging.warning(f"Study folder {study_path_str} does not exist or is not a directory")
            return np.zeros((TARGET_DEPTH, *IMAGE_SIZE, 1), dtype=np.float32)
        
        series_dirs = [p for p in study_path_obj.iterdir() if p.is_dir()]
        if not series_dirs:
            logging.warning(f"No series found for study {study_path_str}")
            return np.zeros((TARGET_DEPTH, *IMAGE_SIZE, 1), dtype=np.float32)
        
        series_dir = max(series_dirs, key=lambda p: len(list(p.glob('*.dcm'))))
        dicom_files = list(series_dir.glob('*.dcm'))
        if not dicom_files:
            logging.warning(f"No DICOM files found in {series_dir}")
            return np.zeros((TARGET_DEPTH, *IMAGE_SIZE, 1), dtype=np.float32)
        
        dicom_data = []
        for f in dicom_files:
            try:
                ds = pydicom.dcmread(str(f), force=True)
                if hasattr(ds, 'pixel_array') and ds.pixel_array.ndim == 2:
                    dicom_data.append(ds)
                else:
                    logging.warning(f"DICOM file {f} has no valid pixel data or incorrect dimensions")
            except Exception as e:
                logging.warning(f"Failed to read DICOM file {f}: {e}")
                continue
        
        if not dicom_data:
            logging.warning(f"No valid DICOM data in {study_path_str}")
            return np.zeros((TARGET_DEPTH, *IMAGE_SIZE, 1), dtype=np.float32)
        
        dicom_data.sort(key=lambda x: float(x.get('SliceLocation', 0.0)))
        volume = [_process_dicom_image(ds) for ds in dicom_data]
        volume = [img for img in volume if img is not None]
        
        if not volume:
            logging.warning(f"No valid images in {study_path_str}")
            return np.zeros((TARGET_DEPTH, *IMAGE_SIZE, 1), dtype=np.float32)
        
        volume_np = np.stack(volume, axis=0)
        logging.info(f"Original volume shape for {study_path_str}: {volume_np.shape}")
        
        if volume_np.shape[0] < 1:
            logging.warning(f"Volume has no valid slices in {study_path_str}")
            return np.zeros((TARGET_DEPTH, *IMAGE_SIZE, 1), dtype=np.float32)
        
        resized_volume = _resize_depth(volume_np, TARGET_DEPTH)
        logging.info(f"Resized volume shape for {study_path_str}: {resized_volume.shape}")
        
        if resized_volume.shape != (TARGET_DEPTH, *IMAGE_SIZE, 1):
            logging.warning(f"Invalid resized volume shape for {study_path_str}: {resized_volume.shape}")
            return np.zeros((TARGET_DEPTH, *IMAGE_SIZE, 1), dtype=np.float32)
        
        return resized_volume.astype(np.float32)
    
    except Exception as e:
        logging.error(f"Error processing study {study_path_str}: {e}")
        return np.zeros((TARGET_DEPTH, *IMAGE_SIZE, 1), dtype=np.float32)

@tf.function
def _tf_load_and_process_study(path: tf.Tensor, label: tf.Tensor):
    volume_shape = (TARGET_DEPTH, *IMAGE_SIZE, 1)
    volume = tf.py_function(_parse_dicom_study, [path], tf.float32)
    volume.set_shape(volume_shape)
    
    is_valid = tf.reduce_any(tf.not_equal(volume, 0)) & \
               tf.reduce_all(tf.equal(tf.shape(volume), volume_shape))
    return volume, label, is_valid

@tf.function
def _augment_volume(volume, label):
    expected_shape = (TARGET_DEPTH, *IMAGE_SIZE, 1)
    volume_shape = tf.shape(volume)
    shape_matches = tf.reduce_all(tf.equal(volume_shape, expected_shape))
    
    def true_fn():
        tf.print("Invalid volume shape in augmentation, returning unchanged")
        return volume, label
    
    def false_fn():
        augmented_volume = volume
        if tf.random.uniform([]) > 0.5:
            augmented_volume = tf.reverse(augmented_volume, axis=[2])
        if tf.random.uniform([]) > 0.5:
            noise = tf.random.normal(shape=tf.shape(augmented_volume), mean=0.0, stddev=0.02, dtype=tf.float32)
            augmented_volume = tf.clip_by_value(augmented_volume + noise, 0.0, 1.0)
        return augmented_volume, label
    
    return tf.cond(shape_matches, false_fn, true_fn)

def create_optimized_tf_dataset(df: pl.DataFrame, batch_size: int, is_training: bool, class_weights: dict = None):
    dataset = tf.data.Dataset.from_tensor_slices((df["study_path"].to_list(), df["label"].to_list()))
    
    if is_training:
        dataset = dataset.shuffle(buffer_size=len(df), reshuffle_each_iteration=True)
    
    dataset = dataset.map(_tf_load_and_process_study, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.filter(lambda volume, label, is_valid: is_valid)
    dataset = dataset.map(lambda volume, label, is_valid: (volume, label), num_parallel_calls=tf.data.AUTOTUNE)
    
    if is_training:
        dataset = dataset.map(_augment_volume, num_parallel_calls=tf.data.AUTOTUNE)
    
    cache_path = Path(f'./cache/{"train" if is_training else "val"}_tfdata_cache')
    cache_path.parent.mkdir(exist_ok=True)
    dataset = dataset.cache(str(cache_path))
    
    if is_training:
        dataset = dataset.repeat()  # Repetir el dataset para el entrenamiento
    
    dataset = dataset.batch(batch_size)
    
    if is_training and class_weights:
        def apply_sample_weights(volume, label):
            sample_weight = tf.gather(tf.constant([class_weights[0], class_weights[1]], dtype=tf.float32), 
                                      tf.cast(label, tf.int32))
            return volume, label, sample_weight
        dataset = dataset.map(apply_sample_weights, num_parallel_calls=tf.data.AUTOTUNE)
    
    return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

# --- FUNCIONES DE MODELO Y ENTRENAMIENTO ---

def _build_resnet3d_model(input_shape, learning_rate=1e-4):
    def res_block(x, filters, kernel_size=(3, 3, 3), strides=1):
        shortcut = x
        if strides > 1 or x.shape[-1] != filters:
            shortcut = Conv3D(filters, (1, 1, 1), strides=strides, padding='same', kernel_initializer='he_normal')(shortcut)
            shortcut = BatchNormalization()(shortcut)

        y = Conv3D(filters, kernel_size, strides=strides, padding='same', kernel_initializer='he_normal')(x)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        
        y = Conv3D(filters, kernel_size, padding='same', kernel_initializer='he_normal')(y)
        y = BatchNormalization()(y)

        y = Add()([shortcut, y])
        y = Activation('relu')(y)
        return y

    inputs = Input(shape=input_shape, dtype=tf.float32)
    x = Conv3D(16, (3, 3, 3), padding='same', kernel_initializer='he_normal')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling3D(pool_size=(3, 3, 3))(x)

    x = res_block(x, 32)
    x = res_block(x, 64, strides=2)
    x = res_block(x, 128, strides=2)
    
    x = GlobalAveragePooling3D()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu', kernel_initializer='he_normal')(x)
    x = Dropout(0.3)(x)
    
    outputs = Dense(1, activation='sigmoid', dtype='float32')(x)
    
    model = Model(inputs, outputs)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            'accuracy'
        ]
    )
    return model

def plot_training_curves_improved(history):
    metrics = ['loss', 'accuracy', 'precision', 'recall', 'auc']
    plt.figure(figsize=(18, 10))
    for i, metric in enumerate(metrics):
        plt.subplot(2, 3, i+1)
        if metric in history.history:
            plt.plot(history.history[metric], label=f'Train {metric}', color='royalblue')
        val_metric = f'val_{metric}'
        if val_metric in history.history:
            plt.plot(history.history[val_metric], label=f'Val {metric}', color='orangered', linestyle='--')
        plt.title(f'Evolución de {metric.capitalize()}')
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig("logs/training_curves_improved.png", dpi=300)
    plt.show()

def pretrain_model():
    logging.info("Iniciando preentrenamiento optimizado del modelo 3D-CNN.")
    
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    
    tf.keras.backend.clear_session()
    gc.collect()
    
    df = pl.read_csv(RSNA_CSV_TRAIN_DIR)
    studies_df = df.group_by("StudyInstanceUID").agg(
        pl.col("negative_exam_for_pe").first()
    ).with_columns(
        pl.col("StudyInstanceUID").map_elements(
            lambda uid: str(Path(RSNA_DATASET_TRAIN_DIR) / uid), 
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
    logging.info(f"Metadatos cargados para {len(studies_df)} estudios con carpetas existentes.")
    if len(studies_df) == 0:
        logging.error("No se encontraron carpetas de estudios válidas. Verifica RSNA_DATASET_TRAIN_DIR.")
        raise ValueError("No hay estudios válidos para procesar.")
    
    missing_studies = df.filter(
        ~df["StudyInstanceUID"].is_in(studies_df["StudyInstanceUID"])
    )["StudyInstanceUID"].to_list()
    if missing_studies:
        logging.warning(f"Se omitieron {len(missing_studies)} estudios porque sus carpetas no existen: {missing_studies[:10]}...")
    
    train_df, val_df = train_test_split(
        studies_df,
        test_size=0.2,
        random_state=42,
        stratify=studies_df["label"]
    )
    logging.info(f"Datos divididos: {len(train_df)} para entrenamiento, {len(val_df)} para validación.")
    print(f"Estudios de entrenamiento: {len(train_df)}, Estudios de validación: {len(val_df)}")
    
    n_total = len(train_df)
    n_neg = len(train_df.filter(pl.col("label") == 0))
    n_pos = n_total - n_neg
    class_weight = {
        0: (1 / n_neg) * (n_total / 2.0) if n_neg > 0 else 1.0,
        1: (1 / n_pos) * (n_total / 2.0) if n_pos > 0 else 1.0,
    }
    logging.info(f"Pesos de clase calculados -> 0: {class_weight[0]:.2f}, 1: {class_weight[1]:.2f}")
        
    step_per_epoch = max(1, len(train_df) // BATCH_SIZE)
    validation_steps = max(1, len(val_df) // BATCH_SIZE)
    train_dataset = create_optimized_tf_dataset(train_df, BATCH_SIZE, is_training=True, class_weights=class_weight)
    val_dataset = create_optimized_tf_dataset(val_df, BATCH_SIZE, is_training=False)
    
    model = _build_resnet3d_model(input_shape=(TARGET_DEPTH, *IMAGE_SIZE, 1))
    model.summary()
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            'models/best_model_auc.keras', monitor='val_auc', save_best_only=True, mode='max', verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.2, patience=5, min_lr=LEARNING_RATE, verbose=1
        ),
        
        #tf.keras.callbacks.EarlyStopping(
        #    monitor='val_auc', patience=15, mode='max', restore_best_weights=True, verbose=1
        #),
        tf.keras.callbacks.CSVLogger('logs/training_log.csv'),
        tf.keras.callbacks.TerminateOnNaN()
    ]
    
    logging.info("Iniciando el entrenamiento del modelo...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        steps_per_epoch=step_per_epoch,
        validation_steps=validation_steps,
        batch_size=1,
        callbacks=callbacks,
        verbose=1
    )
    
    model.save("models/pretrained_rsna_final.keras")
    plot_training_curves_improved(history)
    
    logging.info("Entrenamiento completado exitosamente.")
    
    del model, history, train_dataset, val_dataset
    gc.collect()
    tf.keras.backend.clear_session()