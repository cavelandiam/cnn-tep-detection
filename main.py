import matplotlib.pyplot as plt
import numpy as np
import os
import pydicom
import torch
import multiprocessing as mp
from pathlib import Path
import sys

from scripts.steps import s1_preprocess_data_hucsr as step1
from scripts.steps import s2_create_model as step2
from scripts.steps import s3_inference as step3
from utils import logger, config
from scripts.s4_inference_tep_v2 import run_inference

# --- CONFIGURACIÓN DE LOGGING Y DISPOSITIVO ---
if mp.current_process().name == 'MainProcess':
    logger.init_logger("log_main")

def test_pipeline_v2():

    lineSeparator = "=" * 80

    logger.info(f"START PIPELINE: CNN-TEP DETECTION\n{lineSeparator}")

    logger.info(f"STEP 1: DATA HUCSR - PREPROCESSING\n{lineSeparator}")
    
    #step1.load_images_hucsr()

    logger.info(f"STEP 2: CREATE MODEL WITH BEST FOLD\n{lineSeparator}")

    #step2.train_hucsr()

    logger.info(f"STEP 3: INFERENCE\n{lineSeparator}")
    patient_dir = os.path.join(config.INFERENCE_NEW_PATIENTS_DIR, "TTEP86872473")   

    step3.run_inference("TTEP86872473")

    logger.info(f"END PIPELINE: CNN-TEP DETECTION\n{lineSeparator}")

    logger.info("✅ 🚀 PIPELINE TEST COMPLETE.")


def test_pipeline_v1():
    
    logger.info("INICIANDO PRUEBA DEL PIPELINE: CNN-TEP DETECTION")
    logger.info("PROCESANDO DATOS DE ENTRENAMIENTO RSNA")

    #history, best_auc = s1_improved_3dcnn_tep_copy1.pretrain_model()

    # logger.info(f"🎉 PREENTRENAMIENTO RSNA COMPLETADO! MEJOR AUC: {best_auc:.4f}")

    logger.info("PROCESANDO DATOS DE ENTRENAMIENTO HUCSR")

    #s2_load_images_hucsr.load_images_hucsr()

    logger.info("PROCESANDO FINE-TUNNING DE HUCSR HACIA RSNA MODEL")

    #s3_fine_tunning.train_hucsr_final()

    logger.info("GENERANDO INFERENCIA DE TEP EN PACIENTES HUCSR")

    patient_dir = os.path.join(config.INFERENCE_NEW_PATIENTS_DIR, "TTEP86872473")    
    #s4_inference_tep.final_evaluation(patient_dir=patient_dir)
    
    # if not os.path.exists(patient_dir):
    #     print(f"❌ ERROR: No existe {patient_dir}")
    #     sys.exit(1)
    
    # result = run_inference("TTEP86872473")
    
    # if result is None:
    #     print("\n❌ Inferencia fallida")
    #     sys.exit(1)
    
    print("\n✅ Inferencia exitosa")

    logger.info("FINALIZA EJECUIÓN DEL PIPELINE: CNN-TEP DETECTION")

    logger.info("✅ 🚀 PRUEBA DEL PIPELINE COMPLETA.")


if __name__ == "__main__":
    test_pipeline_v2()    
