Se ha instalado Python 3.9.13, que viene con la versión pip 22.0.4

Se crea el ambiente virtual con el comando: python -m venv venv/env-cnn-tep-detection

Se activa el ambiente virtual: .\venv\env-cnn-tep-detection\Scripts\activate
 si sale un error, ejecutar: Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force

Esto te permitirá ejecutar scripts .ps1 que hayan sido creados localmente, como los de los entornos virtuales.

🔒 Es seguro si solo ejecutas scripts en los que confías.
✅ Solo afecta a tu usuario, no al sistema completo.

```bash
CNN-TEP-DETECTION/
│── data/                               # 📂 Carpeta de datos (imágenes DICOM)
│ ├── patients_tep_false/                   # 📂 Pacientes con TEP HUCSR (imágenes DICOM)
│ ├── patients_tep_true/                    # 📂 Pacientes sin TEP HUCSR (imágenes DICOM)
│ ├── rsna_dataset/                         # 📂 Dataset RSNA Pulmonary Embolism
│── logs/                               # 📂 Guardar los logs de ejecución
│ ├── training.log                          # 📄 Logs de entrenamiento
│── models/                             # 📂 Guardar modelos entrenados
│ ├── efficientnet_radimagenet.keras        # 📂 Pesos de EfficientNet en RadImageNet
│ ├── rsna_pretrained_model.keras           # 📂 Modelo preentrenado en RSNA
│ ├── trained_model.keras                   # 📂 Modelo final entrenado
│── notebooks/                          # 📂 Jupyter Notebooks para pruebas
│ ├── exploratory_analysis.ipynb            # 📓 Análisis exploratorio inicial
│ ├── training_experiment.ipynb             # 📓 Experimentos de entrenamiento
│── scripts/                            # 📂 Código modular del proyecto
│ ├── dataset_loader.py                     # 📝 Carga y etiquetado del dataset del HUCSR
│ ├── evaluate.py                           # 📝 Evaluación del modelo
│ ├── model.py                              # 📝 Definición de EfficientNet + 3D-CNN
│ ├── predict.py                            # 📝 Predicción sobre nuevas imágenes
│ ├── preprocess_rsna.py                    # 📝 Preprocesamiento de imágenes DICOM RSNA
│ ├── train.py                              # 📝 Entrenamiento del modelo
│── utils/                              # 📂 Funciones auxiliares
│ ├── config.py                             # 📝 Parámetros globales del proyecto
│ ├── visualization.py                      # 📝 Funciones para visualizar imágenes y Grad-CAM
│── venv/                               # 📂 Ambiente virtual
│ ├── env-cnn-tep-detection/                # 📂 Ambiente virtual para CNN
│── .gitignore                          # 📄 Ignora las carpetas marcadas
│── main.py                             # 📄 Archivo temporal para probar cada uno de los archivos de la carpeta scripts
│── Makefile                            # 📄 Comandos automatizados
│── README.md                           # 📄 Descripción del proyecto
│── requirements.txt                    # 📄 Dependencias del proyecto
```
