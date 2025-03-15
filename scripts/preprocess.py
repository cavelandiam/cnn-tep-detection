import pydicom
import cv2
import numpy as np
from utils.config import IMAGE_SIZE

def load_dicom_image(dicom_path):
    """Carga y preprocesa una imagen DICOM."""
    dicom = pydicom.dcmread(dicom_path)
    img = dicom.pixel_array
    img = img.astype(np.float32)
    img = (img - np.min(img)) / (np.max(img) - np.min(img))  # Normalización
    img = cv2.resize(img, IMAGE_SIZE)
    img = np.stack((img,) * 3, axis=-1)  # Convertir a RGB
    return img
