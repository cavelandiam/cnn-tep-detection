import matplotlib.pyplot as plt
import numpy as np
import os
import pydicom
import torch
import multiprocessing as mp

from scripts import s1_improved_3dcnn_tep, s2_load_images_hucsr, s3_fine_tunning, s4_inference_tep
from utils import logger, config

def test_pipeline():
    
    logger.info("INICIANDO PRUEBA DEL PIPELINE: CNN-TEP DETECTION")
    logger.info("PROCESANDO DATOS DE ENTRENAMIENTO RSNA")

    history, best_auc = s1_improved_3dcnn_tep.pretrain_model()

    # logger.info(f"🎉 PREENTRENAMIENTO RSNA COMPLETADO! MEJOR AUC: {best_auc:.4f}")

    logger.info("PROCESANDO DATOS DE ENTRENAMIENTO HUCSR")

    #s2_load_images_hucsr.load_images_hucsr()

    logger.info("PROCESANDO FINE-TUNNING DE HUCSR HACIA RSNA MODEL")

    #s3_fine_tunning.finetune_model()

    logger.info("GENERANDO INFERENCIA DE TEP EN PACIENTES HUCSR")

    #patient_dir = os.path.join(config.INFERENCE_NEW_PATIENTS_DIR, "999999999")    
    #s4_inference_tep.run_inference_tep(patient_dir=patient_dir)

    logger.info("FINALIZA EJECUIÓN DEL PIPELINE: CNN-TEP DETECTION")

    logger.info("✅ 🚀 PRUEBA DEL PIPELINE COMPLETA.")


if __name__ == "__main__":
    test_pipeline()    
