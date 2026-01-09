<div align="center">

# ✍️ Handwritten Character Recognition (OCR) System

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)]()

*A real-time OCR application powered by Deep Learning for recognizing handwritten digits and letters*

[Features](#-key-features) • [Installation](#-installation--setup) • [Usage](#-usage-guide) • [Troubleshooting](#-troubleshooting)

</div>

---

## 📖 Project Overview

This project is a real-time **Optical Character Recognition (OCR)** application powered by Deep Learning. It allows users to draw handwritten characters (digits 0–9 and letters A–Z) on a digital canvas and instantly receive a prediction from a trained AI model.

Unlike standard MNIST implementations that only recognize digits, this engine utilizes the **EMNIST (Balanced)** dataset, enabling it to classify **47 distinct classes** of alphanumeric characters with high accuracy.

---

## 🏗️ Architecture

> **The "Factory & Product" Model**

The system is divided into two distinct components to separate training logic from inference:

| Component | File | Description |
|-----------|------|-------------|
| **🏭 The Factory** | `main.py` | A robust TensorFlow pipeline that downloads the EMNIST dataset, pre-processes the data (normalization, rotation, reshaping), constructs a Deep Neural Network, trains it over multiple epochs, and saves the model (`.h5`). |
| **📦 The Product** | `gui_app.py` | A user-friendly desktop interface built with Tkinter. Loads the pre-trained model and provides a 500×500 drawing canvas with real-time image preprocessing. |

---

## 🚀 Key Features

| Feature | Description |
|---------|-------------|
| 🔤 **Extended Recognition** | Supports 0–9, A–Z, and select lowercase letters (47 classes total) |
| 🧠 **Deep Learning Backend** | Built on TensorFlow/Keras with a custom Sequential CNN architecture |
| 🖥️ **Interactive UI** | Large drawing canvas with "Clear" and "Predict" functionality |
| 📊 **Confidence Scoring** | Displays predicted character with model confidence percentage |
| ⚙️ **Robust Preprocessing** | Handles image transposition and scaling for EMNIST orientation quirks |

---

## 🛠 Installation & Setup

### Prerequisites

- Python 3.8+
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/WSalim2024/Mcert-TensorFlow_Model.git
cd Mcert-TensorFlow_Model
```

### Step 2: Create Virtual Environment *(Recommended)*

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

<details>
<summary>📦 Key Libraries</summary>

- `tensorflow`
- `numpy`
- `emnist`
- `pillow` (PIL)
- `matplotlib`

</details>

---

## 💻 Usage Guide

### 1. Build the Brain (Training)

> ⚠️ **Note:** Only necessary if `my_emnist_model.h5` is missing or you wish to retrain.

```bash
python main.py
```

| Parameter | Value |
|-----------|-------|
| ⏱️ Duration | 2–5 minutes (CPU/GPU dependent) |
| 📁 Output | `my_emnist_model.h5` |

### 2. Launch the Application (Inference)

```bash
python gui_app.py
```

**How to use:**

1. ✏️ Draw a character in the black box
2. 🔮 Click **PREDICT** to see the AI's result
3. 🧹 Click **CLEAR** to reset the canvas

---

## ⚠️ Troubleshooting

<details>
<summary><strong>🔴 "BadZipFile" Error during training</strong></summary>

If the EMNIST download fails or gets corrupted due to network firewalls:

1. Download `gzip.zip` manually from the [NIST website](https://www.nist.gov/itl/products-and-services/emnist-dataset)
2. Rename it to `emnist.zip`
3. Place it in your user cache folder: `~/.cache/emnist/`
4. Rerun `main.py`

</details>

---

## ⚖️ Disclaimer

> This software is provided **"as is"**, without warranty of any kind, express or implied. It is intended for **educational and research purposes** to demonstrate machine learning capabilities using TensorFlow.

While the model achieves high accuracy on the test dataset, real-world performance depends heavily on the user's drawing style (mouse vs. stylus) and input consistency. The author accepts no liability for any errors or issues arising from the use of this code.

---

<div align="center">

## 👤 Author

**Waqar Salim**

[![GitHub](https://img.shields.io/badge/GitHub-WSalim2024-181717?style=flat-square&logo=github)](https://github.com/WSalim2024)

---

⭐ *If you found this project useful, consider giving it a star!*

</div>
