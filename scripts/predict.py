import os
import numpy as np
import pydicom
from tensorflow.keras.models import load_model
from scripts.preprocess import load_dicom_image
from utils.config import TRAINED_MODEL_PATH, MESSAGES

def predict_image(dicom_path):
    """
    Carga una imagen DICOM y realiza una predicción con el modelo entrenado.
    Args:
        dicom_path (str): Ruta del archivo DICOM a analizar.
    Returns:
        dict: Resultado de la predicción.
    """
    if not os.path.exists(dicom_path):
        return {"error": f"Archivo no encontrado: {dicom_path}"}

    try:
        # Cargar modelo entrenado en formato `.keras`
        model = load_model(TRAINED_MODEL_PATH)
        print(MESSAGES["model_loaded"].format(TRAINED_MODEL_PATH))

        # Preprocesar imagen DICOM
        img = load_dicom_image(dicom_path)
        img = np.expand_dims(img, axis=0)  # Agregar dimensión batch

        # Realizar predicción
        prediction = model.predict(img)[0][0]
        result = "TEP Detectado" if prediction > 0.5 else "No TEP"

        print(MESSAGES["prediction_success"])
        return {"resultado": result, "confianza": float(prediction)}

    except Exception as e:
        print(MESSAGES["error_loading_dicom"].format(dicom_path), str(e))
        return {"error": "No se pudo procesar la imagen"}

if __name__ == "__main__":
    # Ruta de prueba
    test_dicom_path = "data/patients_tep_true/sample.dcm"
    prediction = predict_image(test_dicom_path)
    print(prediction)
