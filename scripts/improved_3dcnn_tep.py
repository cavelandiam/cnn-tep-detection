import os
import numpy as np
import pydicom
import polars as pl
from utils.config import IMAGE_SIZE, TARGET_DEPTH, RSNA_CSV_TRAIN_DIR, RSNA_DATASET_TRAIN_DIR
from pathlib import Path
import logging
from skimage.transform import resize
from concurrent.futures import ThreadPoolExecutor, as_completed
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import Input
from tensorflow.keras.layers import (Conv3D, MaxPooling3D, GlobalAveragePooling3D, 
                                   Dense, Dropout, BatchNormalization, Add, 
                                   Activation)
import warnings
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import gc

# Configuración optimizada de TensorFlow
tf.config.experimental.enable_memory_growth = True
#tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Filtro para ocultar solo "Invalid value for VR UI"
class IgnoreInvalidVRUIFilter(logging.Filter):
    def filter(self, record):
        return "Invalid value for VR UI" not in record.getMessage()

logging.getLogger("pydicom").addFilter(IgnoreInvalidVRUIFilter())
warnings.filterwarnings('ignore', category=UserWarning, message='Invalid value for VR UI')

# Configuración de logging mejorada
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/train_rsna.log'),
        logging.StreamHandler()
    ]
)

class DataQualityValidator:
    """Clase para validar la calidad de los datos procesados"""
    
    @staticmethod
    def validate_volume(volume, study_id):
        """Valida la calidad de un volumen 3D"""
        if volume is None:
            return False, f"Volumen nulo para {study_id}"
        
        if volume.shape[0] != TARGET_DEPTH:
            return False, f"Profundidad incorrecta {volume.shape[0]} != {TARGET_DEPTH} para {study_id}"
        
        if np.any(np.isnan(volume)) or np.any(np.isinf(volume)):
            return False, f"Valores NaN o Inf encontrados en {study_id}"
        
        if volume.min() < 0 or volume.max() > 1:
            return False, f"Valores fuera del rango [0,1]: [{volume.min():.3f}, {volume.max():.3f}] en {study_id}"
        
        # Validar que no sea un volumen completamente vacío
        if volume.std() < 1e-6:
            return False, f"Volumen sin variación (posiblemente vacío) en {study_id}"
        
        return True, "OK"

def create_improved_model():
    """Arquitectura mejorada con residual connections y batch normalization"""
    inputs = Input(shape=(TARGET_DEPTH, *IMAGE_SIZE, 1))
    
    # Bloque inicial
    x = Conv3D(32, (3, 3, 3), padding='same', dtype=tf.float32)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling3D((2, 2, 2), dtype=tf.float32)(x)  # (128, 112, 112, 32)
    
    # Bloque residual 1
    shortcut = Conv3D(64, (1, 1, 1), padding='same', dtype=tf.float32)(x)
    shortcut = BatchNormalization()(shortcut)
    
    x = Conv3D(64, (3, 3, 3), padding='same', dtype=tf.float32)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv3D(64, (3, 3, 3), padding='same', dtype=tf.float32)(x)
    x = BatchNormalization()(x)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    x = MaxPooling3D((2, 2, 2), dtype=tf.float32)(x)  # (64, 56, 56, 64)
    
    # Bloque residual 2
    shortcut = Conv3D(128, (1, 1, 1), padding='same', dtype=tf.float32)(x)
    shortcut = BatchNormalization()(shortcut)
    
    x = Conv3D(128, (3, 3, 3), padding='same', dtype=tf.float32)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv3D(128, (3, 3, 3), padding='same', dtype=tf.float32)(x)
    x = BatchNormalization()(x)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    x = MaxPooling3D((2, 2, 2), dtype=tf.float32)(x)  # (32, 28, 28, 128)
    
    # Capas adicionales
    x = Conv3D(256, (3, 3, 3), activation='relu', padding='same', dtype=tf.float32)(x)
    x = BatchNormalization()(x)
    x = MaxPooling3D((2, 2, 2), dtype=tf.float32)(x)  # (16, 14, 14, 256)
    x = Dropout(0.3)(x)
    
    x = Conv3D(512, (3, 3, 3), activation='relu', padding='same', dtype=tf.float32)(x)
    x = BatchNormalization()(x)
    x = MaxPooling3D((2, 2, 2), dtype=tf.float32)(x)  # (8, 7, 7, 512)
    x = Dropout(0.4)(x)
    
    # Cabeza de clasificación
    x = GlobalAveragePooling3D()(x)
    x = Dense(512, activation='relu', dtype=tf.float32)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu', dtype=tf.float32)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation='sigmoid', dtype=tf.float32)(x)  # Mixed precision fix
    
    model = Model(inputs, outputs)
    
    # Optimizador con schedule de learning rate
    initial_learning_rate = 1e-3
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=1000,
        decay_rate=0.9,
        staircase=True
    )
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')           
        ]
    )
    
    return model

def grouped_patients_by_study():
    """Función mejorada para agrupar pacientes con validación"""
    df = pl.read_csv(RSNA_CSV_TRAIN_DIR)
    needed_cols = [
        'StudyInstanceUID',
        'SeriesInstanceUID', 
        'SOPInstanceUID',
        'pe_present_on_image',
        'negative_exam_for_pe'
    ]
    
    df = df.select(needed_cols)
    
    # Validar consistencia de etiquetas por estudio
    study_labels = df.group_by("StudyInstanceUID").agg([
        pl.col("negative_exam_for_pe").first().alias("negative_exam"),
        pl.col("negative_exam_for_pe").n_unique().alias("label_consistency")
    ])
    
    # Filtrar estudios con etiquetas inconsistentes
    inconsistent_studies = study_labels.filter(pl.col("label_consistency") > 1)
    if len(inconsistent_studies) > 0:
        logging.warning(f"Se encontraron {len(inconsistent_studies)} estudios con etiquetas inconsistentes")
        study_labels = study_labels.filter(pl.col("label_consistency") == 1)
    
    # Crear etiqueta binaria (invertir negative_exam_for_pe para que 1 = TEP)
    study_labels = study_labels.with_columns([
        (1 - pl.col("negative_exam")).alias("diagnosis")
    ]).select(["StudyInstanceUID", "diagnosis"])
    
    logging.info(f"Estudios válidos después de filtrado: {len(study_labels)}")
    
    return study_labels.to_pandas(), df.to_pandas()

def load_dicom_image_robust(dicom_path, target_size):
    """Versión robusta de carga de imágenes DICOM con mejor manejo de errores"""
    if not os.path.exists(dicom_path) or os.path.getsize(dicom_path) < 10 * 1024:
        return None
    
    try:
        dicom_data = pydicom.dcmread(dicom_path, force=True)
        
        if not hasattr(dicom_data, 'pixel_array'):
            return None
            
        # Descompresión si es necesario
        if hasattr(dicom_data.file_meta, 'TransferSyntaxUID') and dicom_data.file_meta.TransferSyntaxUID.is_compressed:
            dicom_data.decompress()
        
        img = dicom_data.pixel_array.astype(np.float32)
        
        # Filtrar imágenes de resumen diagnóstico (RGB)
        if img.ndim == 3 and img.shape[-1] == 3:
            return None
        
        # Aplicar ventana pulmonar mejorada
        # Ventana pulmonar: nivel=-600 HU, ancho=1600 HU
        window_center = -600
        window_width = 1600
        img_min = window_center - window_width // 2
        img_max = window_center + window_width // 2
        
        # Aplicar rescale slope e intercept si están disponibles
        if hasattr(dicom_data, 'RescaleSlope') and hasattr(dicom_data, 'RescaleIntercept'):
            img = img * float(dicom_data.RescaleSlope) + float(dicom_data.RescaleIntercept)
        
        # Aplicar ventana pulmonar
        img = np.clip(img, img_min, img_max)
        img = (img - img_min) / (img_max - img_min)
        
        # Redimensionar con preservación de rango
        img = resize(img, target_size, anti_aliasing=True, preserve_range=True)
        
        # Validar el resultado
        if np.any(np.isnan(img)) or np.any(np.isinf(img)):
            return None
            
        return np.expand_dims(img, axis=-1)
        
    except Exception as e:
        logging.debug(f"Error al procesar {dicom_path}: {e}")
        return None

def process_study_robust(study_path, train_csv, validator):
    """Versión mejorada de procesamiento de estudios con validación"""
    study_id = study_path.name
    
    # Obtener etiqueta del CSV
    study_rows = train_csv[train_csv['StudyInstanceUID'] == study_id]
    if study_rows.empty:
        return None, None, f"No se encontró etiqueta para {study_id}"
    
    # Usar la primera etiqueta encontrada (ya validada en grouped_patients_by_study)
    label = 1 - int(study_rows['negative_exam_for_pe'].iloc[0])
    
    # Validar estructura de directorios
    series_dirs = [p for p in study_path.iterdir() if p.is_dir()]
    if len(series_dirs) != 1:
        return None, None, f"Estructura de directorios inválida para {study_id}: {len(series_dirs)} series"
    
    series_dir = series_dirs[0]
    files = list(series_dir.glob('*.dcm'))
    
    if len(files) == 0:
        return None, None, f"No se encontraron archivos DICOM en {study_id}"
    
    # Cargar y ordenar archivos DICOM
    dicom_data = []
    for f in files:
        try:
            ds = pydicom.dcmread(f, force=True, stop_before_pixels=True)
            instance_num = getattr(ds, 'InstanceNumber', float('inf'))
            slice_location = getattr(ds, 'SliceLocation', 0)
            dicom_data.append((instance_num, slice_location, f))
        except Exception:
            continue
    
    if not dicom_data:
        return None, None, f"No se pudieron leer metadatos DICOM para {study_id}"
    
    # Ordenar por InstanceNumber primero, luego por SliceLocation
    dicom_data.sort(key=lambda x: (x[0], x[1]))
    sorted_files = [x[2] for x in dicom_data]
    
    # Cargar imágenes con ThreadPoolExecutor
    patient_volume = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_to_file = {
            executor.submit(load_dicom_image_robust, f, IMAGE_SIZE): f 
            for f in sorted_files
        }
        
        for future in as_completed(future_to_file):
            img = future.result()
            if img is not None:
                patient_volume.append(img)
    
    if len(patient_volume) == 0:
        return None, None, f"No se generaron imágenes válidas para {study_id}"
    
    # Convertir a volumen 3D
    volume = np.array(patient_volume)
    
    # Ajustar profundidad
    if volume.shape[0] != TARGET_DEPTH:
        volume = pad_or_trim_volume_robust(volume, TARGET_DEPTH)
    
    # Validar calidad del volumen
    is_valid, message = validator.validate_volume(volume, study_id)
    if not is_valid:
        return None, None, message
    
    return volume, label, "OK"

def pad_or_trim_volume_robust(volume, target_depth):
    """Versión mejorada de ajuste de profundidad con interpolación"""
    D, H, W, C = volume.shape
    
    if D == target_depth:
        return volume
    
    if D < target_depth:
        # Padding con interpolación para suavizar transiciones
        pad_size = target_depth - D
        pad_before = pad_size // 2
        pad_after = pad_size - pad_before
        
        # Crear padding usando los valores de los extremos
        pad_value_start = volume[0:1]
        pad_value_end = volume[-1:]
        
        padding_before = np.tile(pad_value_start, (pad_before, 1, 1, 1))
        padding_after = np.tile(pad_value_end, (pad_after, 1, 1, 1))
        
        return np.concatenate([padding_before, volume, padding_after], axis=0)
    else:
        # Recorte inteligente preservando el centro
        trim_size = D - target_depth
        trim_before = trim_size // 2
        trim_after = trim_size - trim_before
        return volume[trim_before:D - trim_after, :, :, :]

class ImprovedDataGenerator(tf.keras.utils.Sequence):
    """Generador de datos mejorado con tf.keras.utils.Sequence"""
    
    def __init__(self, studies, train_csv, batch_size=8, shuffle=True, augment=False, **kwargs):
        super().__init__(**kwargs)  # Llamar al constructor padre
        self.studies = studies
        self.train_csv = train_csv
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.validator = DataQualityValidator()
        self.indices = np.arange(len(self.studies))
        self.failed_studies = set()
        
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        return len(self.studies) // self.batch_size
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_studies = [self.studies[i] for i in batch_indices]
        
        X_batch = []
        y_batch = []
        
        for study in batch_studies:
            if study.name in self.failed_studies:
                continue
                
            volume, label, message = process_study_robust(study, self.train_csv, self.validator)
            
            if volume is not None:
                if self.augment:
                    volume = self._augment_volume(volume)
                X_batch.append(volume)
                y_batch.append(label)
            else:
                self.failed_studies.add(study.name)
                logging.debug(f"Falló procesamiento: {message}")
        
        # Si el batch queda vacío, rellenar con el siguiente
        while len(X_batch) == 0 and idx < len(self) - 1:
            return self.__getitem__(idx + 1)
        
        if len(X_batch) == 0:
            # Fallback: crear batch dummy
            X_batch = [np.zeros((TARGET_DEPTH, *IMAGE_SIZE, 1))]
            y_batch = [0]
        
        return np.array(X_batch), np.array(y_batch)
    
    def _augment_volume(self, volume):
        """Aplicar augmentación de datos simple en 3D"""
        if np.random.random() > 0.5:
            # Flip horizontal
            volume = np.flip(volume, axis=2)
        
        if np.random.random() > 0.7:
            # Añadir ruido gaussiano suave
            noise = np.random.normal(0, 0.01, volume.shape)
            volume = np.clip(volume + noise, 0, 1)
        
        return volume
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

def create_balanced_generators(studies, train_csv, batch_size=4, validation_split=0.2):
    """Crear generadores balanceados para entrenamiento y validación"""
    
    # Obtener etiquetas para todos los estudios
    study_labels = {}
    validator = DataQualityValidator()
    
    valid_studies = []
    for study in studies:
        study_id = study.name
        study_rows = train_csv[train_csv['StudyInstanceUID'] == study_id]
        if not study_rows.empty:
            label = 1 - int(study_rows['negative_exam_for_pe'].iloc[0])
            study_labels[study_id] = label
            valid_studies.append(study)
    
    # Separar por clases
    positive_studies = [s for s in valid_studies if study_labels[s.name] == 0]
    negative_studies = [s for s in valid_studies if study_labels[s.name] == 1]
    
    logging.info(f"Estudios TEP positivos: {len(positive_studies)}")
    logging.info(f"Estudios TEP negativos: {len(negative_studies)}")
    
    # Dividir cada clase en train/val
    pos_train, pos_val = train_test_split(positive_studies, test_size=validation_split, random_state=42)
    neg_train, neg_val = train_test_split(negative_studies, test_size=validation_split, random_state=42)
    
    # Combinar y mezclar
    train_studies = pos_train + neg_train
    val_studies = pos_val + neg_val
    
    np.random.shuffle(train_studies)
    np.random.shuffle(val_studies)
    
    train_gen = ImprovedDataGenerator(train_studies, train_csv, batch_size, shuffle=True, augment=True)
    val_gen = ImprovedDataGenerator(val_studies, train_csv, batch_size, shuffle=False, augment=False)
    
    return train_gen, val_gen

def pretrain_model():
    """Función principal de entrenamiento mejorada"""
    logging.info("Iniciando preentrenamiento mejorado del modelo 3D-CNN para TEP")
    
    # Cargar y validar datos
    patients_by_study, train_csv = grouped_patients_by_study()
    
    # Obtener lista de estudios válidos
    studies = [p for p in Path(RSNA_DATASET_TRAIN_DIR).iterdir() if p.is_dir()]
    valid_studies = [s for s in studies if s.name in patients_by_study['StudyInstanceUID'].values]
    
    logging.info(f"Estudios válidos encontrados: {len(valid_studies)}")
    
    # Crear modelo mejorado
    model = create_improved_model()
    model.summary()
    
    # Configurar generadores de datos
    batch_size = 8  # Reducido para ajustarse a la RAM
    train_gen, val_gen = create_balanced_generators(valid_studies, train_csv, batch_size)
    
    logging.info(f"Batches de entrenamiento: {len(train_gen)}")
    logging.info(f"Batches de validación: {len(val_gen)}")
    
    # Calcular class weights
    n_positive = len([s for s in valid_studies if (1 - int(train_csv[train_csv['StudyInstanceUID'] == s.name]['negative_exam_for_pe'].iloc[0])) == 1])
    n_negative = len(valid_studies) - n_positive
    
    class_weight = {
        0: len(valid_studies) / (2 * n_negative),
        1: len(valid_studies) / (2 * n_positive)
    }
    
    logging.info(f"Pesos de clase - No-TEP: {class_weight[0]:.3f}, TEP: {class_weight[1]:.3f}")
    
    # Callbacks mejorados
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_auc',
            patience=15,
            restore_best_weights=True,
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'models/best_model_checkpoint.keras',
            monitor='val_auc',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger('logs/training_log.csv'),
        tf.keras.callbacks.TerminateOnNaN()
    ]
    
    # Entrenar modelo
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=150,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1
    )
    
    # Guardar modelo final y graficar curvas
    model.save("models/pretrained_rsna_improved.keras")
    plot_training_curves_improved(history)
    
    # Limpieza de memoria
    del train_gen, val_gen
    gc.collect()
    
    logging.info("Entrenamiento completado exitosamente")
    
    return model, history

def plot_training_curves_improved(history):
    """Función mejorada para graficar curvas de entrenamiento"""
    metrics = ['loss', 'accuracy', 'precision', 'recall', 'auc']
    
    plt.figure(figsize=(15, 10))
    
    for i, metric in enumerate(metrics):
        plt.subplot(2, 3, i+1)
        
        if metric in history.history:
            plt.plot(history.history[metric], label=f'Train {metric}', color='blue')
        
        val_metric = f'val_{metric}'
        if val_metric in history.history:
            plt.plot(history.history[val_metric], label=f'Val {metric}', color='red')
        
        plt.title(f'{metric.upper()} Evolution')
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("logs/training_curves_improved.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Imprimir métricas finales
    if 'val_auc' in history.history:
        max_auc = max(history.history['val_auc'])
        min_loss = min(history.history['val_loss'])
        max_recall = max(history.history['val_recall']) if 'val_recall' in history.history else 0
        max_accuracy = max(history.history['val_accuracy']) if 'val_accuracy' in history.history else 0
        
        logging.info("=== MÉTRICAS FINALES ===")
        logging.info(f"Mejor AUC de validación: {max_auc:.4f}")
        logging.info(f"Menor pérdida de validación: {min_loss:.4f}")
        logging.info(f"Mejor recall de validación: {max_recall:.4f}")
        logging.info(f"Mejor accuracy de validación: {max_accuracy:.4f}")

if __name__ == "__main__":
    try:
        model, history = pretrain_model()
    except Exception as e:
        logging.error(f"Error durante el entrenamiento: {e}")
        raise