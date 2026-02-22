
# Introducción a WSL2 para Ubuntu 22.04 LTS

Para garantizar compatibilidad plena con las bibliotecas de aprendizaje profundo y aprovechar el rendimiento de las GPUs NVIDIA, este proyecto se desarrolla en un entorno basado en Windows Subsystem for Linux 2 (WSL2) ejecutando Ubuntu 22.04 LTS.

A partir de TensorFlow 2.11, Google dejó de ofrecer soporte oficial para CUDA en Windows nativo, lo que impide el uso directo de la GPU en sistemas operativos Windows convencionales. Para superar esta limitación, se optó por WSL2, una capa de virtualización integrada en Windows 11 que permite ejecutar una distribución Linux completa con acceso directo a los drivers NVIDIA instalados en el host.

Ubuntu 22.04 fue seleccionado por su estabilidad, amplia compatibilidad con herramientas de ciencia de datos y soporte oficial para las versiones de CUDA requeridas por TensorFlow. Esta configuración no solo habilita el entrenamiento eficiente de redes neuronales convolucionales en GPU, sino que también asegura coherencia con los entornos de producción y investigación en entornos Linux, facilitando la reproducibilidad y el despliegue futuro.

Así, aunque el desarrollo se realiza en un equipo con Windows 11, el código se ejecuta bajo un entorno Linux auténtico — combinando la comodidad del sistema anfitrión con el poder y la confiabilidad del ecosistema de deep learning en Linux.

Fuente: https://www.tensorflow.org/install/pip?hl=es-419#windows-wsl2

## Instalación de WSL y Ubuntu 22.04 LTS

En Windows 11, desde una consola de PowerShell abierta como administrador, se ejecutan los siguientes comandos para la instalación:

```bash
 wsl --install
 wsl --update
 wsl --list --online
 wsl --install -d Ubuntu-22.04
```
## Configuración de Ubuntu 22.04 LTS

Luego de la instalación, se busca la aplicación instalada con el nombre de "Ubuntu 22.04 LTS" para iniciar el sistema operativo y se ejecutan los siguientes comandos que permitirán: actualizar el sistema operativo, instalar el administrador de paquetes "pip", e instalar el paquete para crear ambientes virtuales:

```bash
 sudo apt update && sudo apt upgrade -y
 sudo apt update && sudo apt upgrade -y python3-pip python3-venv
 sudo apt-get install graphviz
```

Después de realizar la configuración necesaria, se descarga el repositorio y se accede a la carpeta "venv", si no existe, entonces crearla.

## Preparación del proyecto

Se crea la carpeta raiz del proyecto y se ingresa:

```bash
 mkdir cnn-tep-detection
 cd cnn-tep-detection
```

Se crea el ambiente virtual con el siguiente comando: 
```bash
 python3 -m venv venv/env-cnn-tep-detection
```

### Opción 1: Instalar Tensorflow

Se activa el ambiente virtual, se actualiza el paquete "pip" y se instala tensorflow con los siguientes comandos: 
```bash
 source ./venv/env-cnn-tep-detection/bin/activate
 pip install --upgrade pip 
 pip install tensorflow[and-cuda]
```

Para verificar que se reconoce la GPU de windows se ejecuta el siguiente comando:

```bash
 python -c "
import tensorflow as tf
print('TensorFlow:', tf.__version__)
print('GPU disponible:', tf.config.list_physical_devices('GPU'))
print('Num GPUs Available: ', len(tf.config.list_physical_devices('GPU')))
"
```

### Opción 2: Instalar Pythorch

Se activa el ambiente virtual, se actualiza el paquete "pip" y se instala tensorflow con los siguientes comandos: 
```bash
 source ./venv/env-cnn-tep-detection/bin/activate
 pip install --upgrade pip 
 pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Para verificar que se reconoce la GPU de windows se ejecuta el siguiente comando:

```bash
 python -c "
import torch
print('Pythorch:', torch.__version__)
print('GPU disponible:', [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
"
```

Para instalar todos los paquetes se debe estar en el ambiente virtual y se ejecuta el siguiente comando: 

```bash
 pip install -r ./requirements.txt
```

## Estructura del Proyecto

```
├── 📁 graphs
│   ├── 📁 hucsr
│   │   └── 📁 metrics
│   │       ├── 📁 fold_1
│   │       │   └── 🖼️ fold_1_all_metrics.png
│   │       ├── 📁 fold_2
│   │       │   └── 🖼️ fold_2_all_metrics.png
│   │       ├── 📁 fold_3
│   │       │   └── 🖼️ fold_3_all_metrics.png
│   │       ├── 📁 fold_4
│   │       │   └── 🖼️ fold_4_all_metrics.png
│   │       └── 📁 fold_5
│   │           └── 🖼️ fold_5_all_metrics.png
│   └── 📁 rsna
│       └── 📁 resnet3d_pretrained
│           ├── 🖼️ graph.png
│           ├── 🖼️ graph.svg
│           ├── 🌐 report.html
│           └── 📄 summary.txt
├── 📁 inferences
│   ├── 📁 FTEP86974623
│   │   ├── ⚙️ resultado.json
│   │   └── 🖼️ resultado.png
│   ├── 📁 TTEP86872473
│   │   ├── ⚙️ resultado.json
│   │   └── 🖼️ resultado_from_npy.png
│   └── 📁 TTEP90262556
│       ├── ⚙️ resultado.json
│       └── 🖼️ resultado_from_npy.png
├── 📁 scripts
│   ├── 📁 steps
│   │   ├── 🐍 s1_preprocess_data_hucsr.py
│   │   ├── 🐍 s2_create_model.py
│   │   └── 🐍 s3_inference.py
│   ├── 🐍 preprocess_rsna.py
│   ├── 🐍 s1_improved_3dcnn_tep.py
│   ├── 🐍 s1_improved_3dcnn_tep_copy1.py
│   ├── 🐍 s2_load_images_hucsr.py
│   ├── 🐍 s3_fine_tunning.py
│   ├── 🐍 s4_inference_tep.py
│   └── 🐍 s4_inference_tep_v2.py
├── 📁 utils
│   ├── 🐍 __init__.py
│   ├── 🐍 config.py
│   ├── 🐍 logger.py
│   └── 🐍 visualization.py
├── ⚙️ .gitignore
├── 📄 Makefile
├── 📝 README.MD
├── 🐍 main.py
├── 📄 requirements-cpu.txt
├── 📄 requirements.txt
└── 🐍 test.py
```
