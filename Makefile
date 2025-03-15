# Variables globales con detección de OS
OS := $(shell uname -s)
ifeq ($(OS), Windows_NT)
    VENV=venv/env-cnn-tep-detection/Scripts/activate
    PYTHON=venv/env-cnn-tep-detection/Scripts/python
    PIP=venv/env-cnn-tep-detection/Scripts/pip
else
    VENV=venv/env-cnn-tep-detection/bin/activate
    PYTHON=venv/env-cnn-tep-detection/bin/python
    PIP=venv/env-cnn-tep-detection/bin/pip
endif

# 📌 1️⃣ Configuración e Instalación
install:
	python -m venv venv/env-cnn-tep-detection && source $(VENV) && $(PIP) install -r requirements.txt

# 📌 2️⃣ Entrenamiento del modelo
train:
	source $(VENV) && $(PYTHON) scripts/train.py

# 📌 3️⃣ Evaluación del modelo en datos de validación
evaluate:
	source $(VENV) && $(PYTHON) scripts/evaluate.py

# 📌 4️⃣ Predicción sobre nuevas imágenes DICOM
predict:
	source $(VENV) && $(PYTHON) scripts/predict.py data/patients_tep_true/sample.dcm

# 📌 5️⃣ Limpiar archivos temporales y logs
clean:
	rm -rf __pycache__/ logs/*.log models/*.keras venv/

# 📌 6️⃣ Probar todo el pipeline (carga → entrenamiento → evaluación → guardado) #NIVEL DE DESARROLLO
test:
	source $(VENV) && $(PYTHON) main.py