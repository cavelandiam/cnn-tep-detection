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
import matplotlib.pyplot as plt


# Se mantienen tus imports de configuración
from utils.config import IMAGE_SIZE, TARGET_DEPTH, RSNA_CSV_TRAIN_DIR, RSNA_DATASET_TRAIN_DIR

from tensorflow.keras.models import Model
from scipy.ndimage import zoom
from tensorflow.keras.layers import (Input, Conv3D, BatchNormalization, Activation, 
                                     MaxPooling3D, Add, GlobalAveragePooling3D, 
                                     Dropout, Dense)


# --- CONFIGURACIÓN DE TENSORFLOW Y LOGGING (MEJORADA) ---

# Habilita el crecimiento de memoria para evitar que TF reserve toda la VRAM
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logging.info(f"Crecimiento de memoria habilitado para {len(gpus)} GPU(s).")
    except RuntimeError as e:
        logging.error(f"Error al configurar la memoria de la GPU: {e}")

# Habilita la precisión mixta para acelerar el entrenamiento y reducir el uso de memoria
# ¡CRÍTICO! Esto permite al modelo usar float16 para cálculos y float32 para variables,
# lo que resulta en un rendimiento mucho mayor en GPUs compatibles (Tensor Cores).
#tf.keras.mixed_precision.set_global_policy('mixed_float16')
logging.info("Política de precisión mixta de Keras establecida en 'mixed_float16'.")

# Se mantiene tu filtro de logging
class IgnoreInvalidVRUIFilter(logging.Filter):
    def filter(self, record):
        return "Invalid value for VR UI" not in record.getMessage()

logging.getLogger("pydicom").addFilter(IgnoreInvalidVRUIFilter())
warnings.filterwarnings('ignore', category=UserWarning, message='Invalid value for VR UI')

# Se mantiene tu configuración básica de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/train_rsna.log'),
        logging.StreamHandler()
    ]
)

# --- FUNCIONES DE PROCESAMIENTO DE DATOS (REFACTORIZADAS PARA TF.DATA) ---

def _parse_dicom_study(study_path: bytes):
    """
    Función de bajo nivel para procesar un estudio DICOM completo.
    Diseñada para ser envuelta en tf.py_function. Recibe bytes.
    """
    try:
        # Convertir tensor a string si es necesario
        if isinstance(study_path, tf.Tensor):
            study_path_str = study_path.numpy().decode('utf-8')
        else:
            study_path_str = study_path.decode('utf-8')

        series_dirs = [p for p in Path(study_path_str).iterdir() if p.is_dir()]
        if not series_dirs:
            return np.array([], dtype=np.float32)

        # Heurística: usa la carpeta de serie con más archivos DICOM
        series_dir = max(series_dirs, key=lambda p: len(list(p.glob('*.dcm'))))
        
        dicom_files = list(series_dir.glob('*.dcm'))
        if not dicom_files:
            return np.array([], dtype=np.float32)

        dicom_data = []
        for f in dicom_files:
            ds = pydicom.dcmread(str(f), force=True)
            if hasattr(ds, 'pixel_array'):
                dicom_data.append(ds)
        
        # Ordenar por SliceLocation para garantizar el orden anatómico correcto
        dicom_data.sort(key=lambda x: float(x.get('SliceLocation', 0.0)))
        
        volume = [_process_dicom_image(ds) for ds in dicom_data]
        volume = [img for img in volume if img is not None]
        
        if not volume:
            return np.array([], dtype=np.float32)
        
        volume_np = np.stack(volume, axis=0)
        resized_volume = _resize_depth(volume_np, TARGET_DEPTH)
        
        return resized_volume.astype(np.float32)

    except Exception as e:
        logging.debug(f"Fallo al procesar el estudio {study_path.numpy().decode('utf-8', errors='ignore')}: {e}")
        return np.array([], dtype=np.float32)

def _process_dicom_image(ds: pydicom.FileDataset):
    """Procesa una única imagen DICOM a un array normalizado."""
    if ds.pixel_array.ndim > 2: return None  # Ignorar imágenes de resumen/color

    img = ds.pixel_array.astype(np.float32)
    if 'RescaleSlope' in ds and 'RescaleIntercept' in ds:
        img = img * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
    
    # Ventana pulmonar estándar (HU)
    window_center, window_width = -600, 1600
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img = np.clip(img, img_min, img_max)
    
    # Normalizar a [0, 1]
    img = (img - img_min) / (img_max - img_min)
    
    img_resized = resize(img, IMAGE_SIZE, anti_aliasing=True)
    return np.expand_dims(img_resized, axis=-1)

def _resize_depth(volume, target_depth):
    """Ajusta la profundidad del volumen a target_depth usando interpolación trilineal."""
    current_depth = volume.shape[0]
    if current_depth == target_depth:
        return volume

    # Calcular factor de escala en el eje profundidad (D)
    scale = target_depth / current_depth

    # Interpolación trilineal usando zoom en el eje 0 (profundidad)
    resized = zoom(volume, (scale, 1, 1, 1), order=1)  # order=1 → interpolación lineal
    return resized.astype(np.float32)

@tf.function
def _tf_load_and_process_study(path: tf.Tensor, label: tf.Tensor):
    """
    Wrapper de TensorFlow que usa tf.py_function para ejecutar el procesamiento.
    Esta función se mapea sobre el dataset de rutas.
    """
    volume_shape = (TARGET_DEPTH, *IMAGE_SIZE, 1)
    
    # Ejecuta la función de numpy/pydicom dentro del grafo de TensorFlow
    volume = tf.py_function(_parse_dicom_study, [path], tf.float32)
    volume.set_shape(volume_shape) # Es crucial redefinir la forma aquí
    
    # Filtra los volúmenes que fallaron en la carga (devuelven un array vacío)
    is_valid = tf.shape(volume)[0] > 0
    return volume, label, is_valid

@tf.function
def _augment_volume(volume, label):
    """Aplica aumentos 3D eficientes usando operaciones de TF."""
    # El volumen tiene forma (D, H, W, C)
    # Flip horizontal aleatorio (a lo largo del eje de anchura)
    if tf.random.uniform([]) > 0.5:
        volume = tf.reverse(volume, axis=[2])
    
    # Añadir ruido gaussiano
    if tf.random.uniform([]) > 0.5:
        noise = tf.random.normal(shape=tf.shape(volume), mean=0.0, stddev=0.02, dtype=tf.float32)
        volume = tf.clip_by_value(volume + noise, 0.0, 1.0)
        
    return volume, label

def create_optimized_tf_dataset(df: pl.DataFrame, batch_size: int, is_training: bool, class_weights: dict = None):
    """
    Crea un pipeline de tf.data altamente optimizado.
    Esta es la mejora de rendimiento MÁS IMPORTANTE.
    """
    dataset = tf.data.Dataset.from_tensor_slices((df["study_path"].to_list(), df["label"].to_list()))

    if is_training:
        dataset = dataset.shuffle(buffer_size=len(df), reshuffle_each_iteration=True)

    # Mapear la función de carga y procesamiento en paralelo
    dataset = dataset.map(_tf_load_and_process_study, num_parallel_calls=tf.data.AUTOTUNE)

    # Filtrar elementos que fallaron en la carga
    dataset = dataset.filter(lambda volume, label, is_valid: is_valid)
    # Descartar el indicador de validez
    dataset = dataset.map(lambda volume, label, is_valid: (volume, label), num_parallel_calls=tf.data.AUTOTUNE)

    if is_training:
        dataset = dataset.map(_augment_volume, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size)

    # **MEJORA CLAVE:** Usar cache en disco. La primera época será lenta,
    # pero las siguientes leerán desde este archivo ultra-rápido.
    cache_path = Path(f'./cache/{"train2" if is_training else "val"}_tfdata_cache')
    cache_path.parent.mkdir(exist_ok=True)
    dataset = dataset.cache(str(cache_path))

    # Aplicar pesos de muestra si se proporcionan (para manejar desbalance de clases)
    if is_training and class_weights:
        def apply_sample_weights(volume, label):
            sample_weight = tf.gather(tf.constant([class_weights[0], class_weights[1]], dtype=tf.float32), 
                                      tf.cast(label, tf.int32))
            return volume, label, sample_weight
        dataset = dataset.map(apply_sample_weights, num_parallel_calls=tf.data.AUTOTUNE)

    # Pre-cargar lotes en la GPU para evitar cuellos de botella
    return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)


# --- FUNCIONES DE MODELO Y ENTRENAMIENTO (REFACTORIZADAS) ---

def _build_resnet3d_model(input_shape, learning_rate=1e-4):
    """Crea un modelo 3D-CNN con bloques residuales, siguiendo las mejores prácticas."""

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

        # Conexión residual: la activación viene DESPUÉS de la suma
        y = Add()([shortcut, y])
        y = Activation('relu')(y)
        return y

    inputs = Input(shape=input_shape, dtype=tf.float32)

    # Bloque inicial
    x = Conv3D(32, (3, 3, 3), padding='same', kernel_initializer='he_normal')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)

    # Bloques residuales
    x = res_block(x, 64)
    x = res_block(x, 128, strides=2)
    x = res_block(x, 256, strides=2)
    
    # Cabeza de clasificación
    x = GlobalAveragePooling3D()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu', kernel_initializer='he_normal')(x)
    x = Dropout(0.3)(x)
    
    # Capa de salida con activación sigmoid en float32 por estabilidad numérica
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


# --- FUNCIÓN PRINCIPAL DE ORQUESTACIÓN ---

def pretrain_model():
    """
    Función principal de entrenamiento, ahora más limpia y orquestando funciones modulares.
    """
    logging.info("Iniciando preentrenamiento optimizado del modelo 3D-CNN.")
    
    # 1. Cargar metadatos y crear DataFrame con rutas y etiquetas
    df = pl.read_csv(RSNA_CSV_TRAIN_DIR)
    studies_df = df.group_by("StudyInstanceUID").agg(
        pl.col("negative_exam_for_pe").first()
    ).with_columns(
        # --- ESTA ES LA LÍNEA CORREGIDA ---
        pl.col("StudyInstanceUID").map_elements(
            lambda uid: str(Path(RSNA_DATASET_TRAIN_DIR) / uid), 
            return_dtype=pl.String
        ).alias("study_path"),
        # ----------------------------------
        (1 - pl.col("negative_exam_for_pe")).alias("label")
    )
    studies_df = studies_df.filter(pl.col("study_path").map_elements(lambda p: Path(p).exists(), return_dtype=pl.Boolean))
    logging.info(f"Metadatos cargados para {len(studies_df)} estudios existentes.")

    # 2. División de datos estratificada
    train_df, val_df = train_test_split(
        studies_df,
        test_size=0.2,
        random_state=42,
        stratify=studies_df["label"] # ¡Clave! Mantiene la proporción de clases
    )
    logging.info(f"Datos divididos: {len(train_df)} para entrenamiento, {len(val_df)} para validación.")

    # 3. Calcular pesos de clase para el desbalance
    n_total = len(train_df)
    n_neg = len(train_df.filter(pl.col("label") == 0))
    n_pos = n_total - n_neg
    class_weight = {
        0: (1 / n_neg) * (n_total / 2.0),
        1: (1 / n_pos) * (n_total / 2.0),
    }
    logging.info(f"Pesos de clase calculados -> 0: {class_weight[0]:.2f}, 1: {class_weight[1]:.2f}")

    # 4. Crear los pipelines de datos optimizados
    BATCH_SIZE = 4 # Se puede aumentar gracias a la optimización de memoria
    step_per_epoch = n_total // BATCH_SIZE
    validation_steps = len(val_df) // BATCH_SIZE
    train_dataset = create_optimized_tf_dataset(train_df, BATCH_SIZE, is_training=True, class_weights=class_weight)
    val_dataset = create_optimized_tf_dataset(val_df, BATCH_SIZE, is_training=False)
    
    # 5. Construir el modelo
    model = _build_resnet3d_model(input_shape=(TARGET_DEPTH, *IMAGE_SIZE, 1))
    model.summary()

    # 6. Definir Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            'models/best_model_auc.keras', monitor='val_auc', save_best_only=True, mode='max', verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_auc', patience=15, mode='max', restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.CSVLogger('logs/training_log.csv'),
        tf.keras.callbacks.TerminateOnNaN()
    ]

    # 7. Entrenar el modelo
    logging.info("Iniciando el entrenamiento del modelo...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=3,
        steps_per_epoch = step_per_epoch,
        validation_steps = validation_steps,
        callbacks=callbacks,
        verbose=1
    )

    # 8. Guardar y visualizar resultados
    model.save("models/pretrained_rsna_final.keras")
    plot_training_curves_improved(history) # Se puede reusar tu función de ploteo
    
    logging.info("Entrenamiento completado exitosamente.")
    
    # 9. Limpieza de memoria
    del model, history, train_dataset, val_dataset
    gc.collect()
    tf.keras.backend.clear_session()
    
    # La función debe retornar algo si main.py espera una salida
    # Basado en tu código original, retornaba model y history.
    # return model, history

# Puedes mantener tus funciones de ploteo y validación tal como están,
# ya que eran correctas. Aquí una copia para que el script sea autocontenido.
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