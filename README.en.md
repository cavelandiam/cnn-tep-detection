# Introduction to WSL2 for Ubuntu 22.04 LTS

To ensure full compatibility with deep learning libraries and leverage the performance of NVIDIA GPUs, this project is developed in an environment based on Windows Subsystem for Linux 2 (WSL2) running Ubuntu 22.04 LTS.

Starting with TensorFlow 2.11, Google discontinued official support for CUDA on native Windows, which prevents direct GPU usage on conventional Windows operating systems. To overcome this limitation, WSL2 was chosen, a virtualization layer integrated into Windows 11 that allows a full Linux distribution to run with direct access to the NVIDIA drivers installed on the host.

Ubuntu 22.04 was selected for its stability, broad compatibility with data science tools, and official support for the CUDA versions required by TensorFlow. This configuration not only enables efficient training of convolutional neural networks on GPUs, but also ensures consistency with production and research environments in Linux environments, facilitating reproducibility and future deployment.

Thus, even though development is done on a Windows 11 machine, the code runs under an authentic Linux environment—combining the convenience of the host system with the power and reliability of the deep learning ecosystem on Linux.

Source: https://www.tensorflow.org/install/pip?hl=es-419#windows-wsl2

## Installing WSL and Ubuntu 22.04 LTS

In Windows 11, from a PowerShell console opened as an administrator, run the following commands for installation:

```bash
 wsl --install
 wsl --update
 wsl --list --online
 wsl --install -d Ubuntu-22.04
```
## Configuring Ubuntu 22.04 LTS

After installation, search for the installed application named “Ubuntu 22.04 LTS” to start the operating system and run the following commands to: update the operating system, install the “pip” package manager, and install the package to create virtual environments:

```bash
 sudo apt update && sudo apt upgrade -y
 sudo apt update && sudo apt upgrade -y python3-pip python3-venv
 sudo apt-get install graphviz
```

After performing the necessary configuration, download the repository and access the “venv” folder. If it does not exist, create it.

## Project preparation

Create the project root folder and enter:

```bash
 mkdir cnn-tep-detection
 cd cnn-tep-detection
```

Create the virtual environment with the following command:
```bash
 python3 -m venv venv/env-cnn-tep-detection
```

### Option 1: Install Tensorflow

Activate the virtual environment, update the “pip” package, and install tensorflow with the following commands:
```bash
 source ./venv/env-cnn-tep-detection/bin/activate
 pip install --upgrade pip 
 pip install tensorflow[and-cuda]
```

To verify that the Windows GPU is recognized, run the following command:

```bash
 python -c "
import tensorflow as tf
print(‘TensorFlow:’, tf.__version__)
print(‘GPU available:’, tf.config.list_physical_devices(‘GPU’))
print(‘Num GPUs Available: ’, len(tf.config.list_physical_devices(‘GPU’)))
“
```

### Option 2: Install Pythorch

Activate the virtual environment, update the ”pip" package, and install TensorFlow with the following commands:
 
```bash
 source ./venv/env-cnn-tep-detection/bin/activate
 pip install --upgrade pip 
 pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

To verify that the Windows GPU is recognized, run the following command:

```bash
 python -c “
import torch
print(‘Pythorch:’, torch.__version__)
print(‘GPU available:’, [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
”
```

To install all packages, you must be in the virtual environment and run the following command: 

```bash
 pip install -r ./requirements.txt
```

## Project Structure

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
│   │   ├── ⚙️ result.json
│   │   └── 🖼️ result.png
│   ├── 📁 TTEP86872473
│   │   ├── ⚙️ result.json
│   │   └── 🖼️ result_from_npy.png
│   └── 📁 TTEP90262556
│       ├── ⚙️ result.json
│       └── 🖼️ result_from_npy.png
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
