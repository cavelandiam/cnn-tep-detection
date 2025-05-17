import numpy as np
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from scripts import load_images_hucsr, validate_load_images_hucsr, calculate_target_depth, pretrain_rsna
import matplotlib.pyplot as plt
import pydicom

def test_pipeline():
    """Ejecuta todo el flujo de trabajo: carga de datos, entrenamiento, evaluación y predicción."""
    
    print("INICIANDO PRUEBA DEL PIPELINE COMPLETO...\n")

    print("Calcular target depth HUCSR...")
    #calculate_target_depth.calculate()
    print("Cargando las imágenes del HUCSR ...")
    #load_images_hucsr.load_all_datasets()
    print("Validando las imágenes del HUCSR ...")
    validate_load_images_hucsr.validate()

    #print("Calcular target depth RSNA...")
    #calculate_target_depth.calculate() NO SE USA
    #print("Cargando las imágenes del RSNA ...")
    #load_images_rsna.load_all_datasets() NO SE USA
    #print("Validando las imágenes del RSNA ...")
    #validate_load_images_hucsr.validate() NO SE USA

    print("Preentrenando el modelo RSNA ...")
    #pretrain_rsna.pretrain_model()
    #train_file = dataset_loader.load_all_datasets()

    # 1️⃣ Cargar el dataset (Imágenes RNSA)
    #rsna_train_file = process_rsna.load_data_rsna()
    print("EJECIÓN DEL PIPELINE COMPLETA.\n")
    #print(f"✅ Datos cargados: {X_train.shape[0]} imágenes de entrenamiento, {X_val.shape[0]} imágenes de validación.\n")

    # 2️⃣ Construir el modelo
    #print("🔧 Construyendo el modelo EfficientNet + 3D-CNN...")
    #model = build_3d_cnn()
    #print("✅ Modelo construido correctamente.\n")

    # 3️⃣ Entrenar el modelo
    #print("🚀 Iniciando entrenamiento...")
    #model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=3, batch_size=4)
    #print("✅ Entrenamiento completado.\n")

    # 4️⃣ Evaluar el modelo
    #print("📊 Evaluando el modelo en el conjunto de validación...")
    #val_loss, val_acc = model.evaluate(X_val, y_val)
    #print(f"📈 Resultados: Pérdida: {val_loss:.4f}, Precisión: {val_acc:.4f}\n")

    # 5️⃣ Guardar el modelo entrenado
    #print("💾 Guardando el modelo entrenado...")
    #model.save(TRAINED_MODEL_PATH)
    #print(MESSAGES["model_saved"].format(TRAINED_MODEL_PATH))
    
    #print("✅ 🚀 PRUEBA DEL PIPELINE COMPLETA.")

def visualizar_imagen_dicom(ruta_directorio):
    """
    Visualiza una imagen DICOM en escala de grises.

    Args:
        ruta_directorio (str): Ruta de la carpeta que contiene los archivos DICOM.
        nombre_archivo (str): Nombre del archivo DICOM a visualizar.
    """
    dicom_path = os.path.join(ruta_directorio)

    # Cargar imagen DICOM
    dicom_data = pydicom.dcmread(dicom_path)

    # Obtener el array de píxeles
    dicom_image = dicom_data.pixel_array

    # Mostrar imagen
    plt.figure(figsize=(6, 6))
    plt.imshow(dicom_image, cmap="gray")
    plt.colorbar()
    plt.title(f"Imagen DICOM: DCM3249")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    test_pipeline()
