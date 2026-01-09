<div align="center">

# 🤖 Hybrid AI Recognition System

### TensorFlow + PyTorch

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)]()

*A Dual-Engine AI System integrating TensorFlow and PyTorch into a single unified dashboard*

[Features](#-key-features) • [Architecture](#-architecture) • [Installation](#-installation--setup) • [Usage](#-usage-guide)

**Version 1.2**

</div>

---

## 📖 Project Overview

This project is a **Dual-Engine AI System** that integrates two powerful Deep Learning frameworks into a single dashboard. It allows users to switch between:

| Engine | Framework | Dataset | Purpose |
|--------|-----------|---------|---------|
| 🔤 **Character Recognition (OCR)** | TensorFlow | EMNIST | Recognize handwritten digits & letters |
| 🎯 **Object Detection** | PyTorch | CIFAR-10 | Identify objects from sketches |

The goal is to demonstrate how different neural network architectures (Custom CNNs) and frameworks can coexist in a modular Python application.

---

## 🏗️ Architecture

> **The "Factory & Product" Model with Unified Frontend**

```
┌─────────────────────────────────────────────────────────────┐
│                    🖥️ unified_app.py                        │
│                   (Tkinter Dashboard)                       │
├─────────────────────────┬───────────────────────────────────┤
│    📝 Character Mode    │         🎨 Object Mode            │
├─────────────────────────┼───────────────────────────────────┤
│   my_emnist_model.h5    │        cifar_net.pth              │
│      (TensorFlow)       │          (PyTorch)                │
└─────────────────────────┴───────────────────────────────────┘
```

### Component Breakdown

| Type | Component | File | Description |
|------|-----------|------|-------------|
| 🏭 | **Factory** | `main.py` | Trains TensorFlow model for character recognition |
| 🏭 | **Factory** | `pytorch_train.py` | Trains PyTorch model for object detection |
| 📦 | **Product** | `unified_app.py` | Tkinter dashboard that dynamically loads the appropriate AI backend |
| 🧠 | **Brain** | `my_emnist_model.h5` | TensorFlow character weights |
| 🧠 | **Brain** | `cifar_net.pth` | PyTorch object weights |

---

## 🚀 Key Features

| Feature | Description |
|---------|-------------|
| 🔀 **Hybrid Backend** | Seamlessly switches between TensorFlow (Grayscale/28×28) and PyTorch (RGB/32×32) pipelines |
| 🔤 **47-Class OCR** | Recognizes Digits (0–9) and Letters (A–Z) using EMNIST Balanced |
| 🎯 **10-Class Object Detection** | Identifies sketches of Planes, Cars, Birds, Cats, and more using CIFAR-10 |
| ⚡ **Real-Time Inference** | Instant prediction on drawn canvas inputs |
| 🔧 **Smart Preprocessing** | Auto-handles resizing, normalization, and transposition for both frameworks |

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
- `torch`
- `torchvision`
- `emnist`
- `pillow` (PIL)
- `numpy`

</details>

---

## 💻 Usage Guide

### 1. Build the Brains (Training)

> ⚠️ **Important:** You must generate the model files before running the app.

#### Train Character Recognizer (TensorFlow)

```bash
python main.py
```

| Parameter | Value |
|-----------|-------|
| 📁 Output | `my_emnist_model.h5` |
| 🔤 Classes | 47 (digits + letters) |

#### Train Object Detector (PyTorch)

```bash
python pytorch_train.py
```

| Parameter | Value |
|-----------|-------|
| 📁 Output | `cifar_net.pth` |
| 🎯 Classes | 10 (plane, car, bird, cat, etc.) |

### 2. Launch the Unified Dashboard

```bash
python unified_app.py
```

**How to use:**

| Step | Action |
|------|--------|
| 1️⃣ | Select **"Read Characters"** to draw numbers/letters |
| 2️⃣ | Select **"See Objects"** to draw shapes (cars, birds, etc.) |
| 3️⃣ | Draw on the canvas |
| 4️⃣ | Click **ACTIVATE AI** to predict |

---

## ⚖️ Disclaimer

> This software is provided **"as is"** for **educational purposes**.

⚠️ **Accuracy Note:** The Object Detector (PyTorch) was trained on photographs (CIFAR-10), so asking it to recognize hand-drawn sketches is an **experimental challenge**. Accuracy on sketches will be lower than on photos.

---

<div align="center">

## 👤 Author

**Waqar Salim**

[![GitHub](https://img.shields.io/badge/GitHub-WSalim2024-181717?style=flat-square&logo=github)](https://github.com/WSalim2024)

---

⭐ *If you found this project useful, consider giving it a star!*

</div>
