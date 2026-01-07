# 🩸 HemaliteV10: Lightweight Medical Object Detection

**HemaliteV10** is a specialized, lightweight deep learning model engineered for the accurate detection of blood cells in medical imagery. Built on a modified YOLOv10 architecture, it is designed to balance high-depth feature extraction with extreme parameter efficiency (~1.2M parameters), making it suitable for deployment on standard laptops and edge environments.

---

## 🔬 Project Overview

Automated blood cell detection is crucial for diagnosing hematological conditions like Leukemia (B-ALL) and analyzing peripheral blood smears (PBC). Standard object detection models are often too heavy for rapid deployment or too shallow to capture the subtle textures required for medical analysis.

**HemaliteV10** addresses this by introducing a novel **"Strong Backbone, Skinny Head"** architecture. It maintains a deep, high-channel backbone to extract rich medical features but drastically prunes the detection head to minimize computational cost.

### Key Innovations
* **Custom "Skinny Head" Design:** Reduces the parameter count to **~1.2M** by constraining feature fusion layers to 64 channels, significantly lowering memory usage and FLOPs.
* **CBAM Injection:** Integrates **Convolutional Block Attention Modules** (Channel & Spatial Attention) directly into the backbone. This forces the network to focus on relevant cell features (nuclei texture, cytoplasm shape) while ignoring background noise.

---

## 🧠 Architecture

The architecture is built on a custom YOLOv10 base, optimized for the trade-off between feature depth and efficiency.

* **Backbone:** Maintains **128 channels** in deep layers (P4, P5) to ensure sufficient capacity for learning complex cell morphologies.
* **Neck & Head:** All feature pyramid network (PANet) layers are restricted to **64 channels**. This creates a "skinny" information highway that retains semantic strength from the backbone while eliminating redundant computations.

---

## 📊 Comparative Accuracy Analysis

We evaluated **HemaliteV10** against a standard **Baseline (YOLOv10n)** across four diverse medical datasets. The results highlight the model's ability to match or exceed the baseline in complex scenarios despite its reduced parameter count.

| Dataset | Model | mAP@50 | mAP@50-95 |
| :--- | :--- | :--- | :--- |
| **BCCD** (Blood Cell Count) | Baseline | 0.93 | 0.69 |
| | **HemaliteV10** | 0.89 | 0.63 |
| | | | |
| **PBC** (Peripheral Blood) | Baseline | 0.99 | 0.99 |
| | **HemaliteV10** | **0.99** | **0.99** |
| | | | |
| **B-ALL** (Leukemia) | Baseline | 0.98 | 0.76 |
| | **HemaliteV10** | 0.97 | 0.72 |
| | | | |
| **yolo_new** (General Medical) | Baseline | 0.97 | 0.87 |
| | **HemaliteV10** | **0.98** | **0.89** |




# 🧟 Frankenstein_MGDA: Multi-Granularity Domain Adaptation

**Frankenstein_MGDA** is a custom, high-performance object detection architecture designed for robust deployment on edge devices like the **NVIDIA Jetson Orin**.

Built by surgically modifying **YOLOv11n**, this project "stitches" together state-of-the-art efficiency modules (SqueezeNet) with advanced Multi-Task Learning (MGDA). The result is a model that achieves extreme compression (**~3.2 MB**) and real-time speed (**~63 FPS**) while remaining robust to domain shifts (lighting, sensor noise, and blur) through Multi-Granularity Domain Adaptation.

---

## 🔬 Project Overview

Standard lightweight models often fail when moving from training data (Source Domain) to real-world deployment (Target Domain) due to subtle "style" shifts like microscope lighting or camera noise.

**Frankenstein_MGDA** solves this by introducing a **Multi-Granularity** approach. It forces the network to align features across domains at three distinct levels:
1.  **Spectral (Input):** Removing style bias via Fourier Transforms.
2.  **Fine (Pixel):** Preserving small object details via Super-Resolution.
3.  **Coarse (Global):** Enforcing semantic consistency via Adversarial Learning.

Simultaneously, it replaces heavy convolutional blocks with **Fire Modules** to drastically reduce parameters for edge storage.

---

## 🧠 Architecture: The "Frankenstein" Surgery

The architecture is a hybrid system, replacing standard YOLO components with specialized modules to balance speed, size, and robustness.

### 1. Input Stage: Spectral Granularity (FFT) 🌊

* **Module:** **Fourier Mixing Block**
* **Function:** Before the network sees an image, we apply a **Fast Fourier Transform (FFT)** to decompose it into **Amplitude** (Style) and **Phase** (Structure).
* **Mechanism:** The model dynamically swaps the Amplitude spectrum of the training batch with others. This mathematically removes "style" bias (lighting/color), forcing the model to learn pure structure.

### 2. Backbone: The "Fire" Compression 🔥

* **Module:** **SqueezeNet Fire Modules** (Replacing `C2f`/`C3k2`)
* **Function:** Extreme Parameter Reduction.
* **Mechanism:** Uses a "Squeeze" layer (1x1 Conv) to compress channel depth before feeding into parallel "Expand" layers (1x1 & 3x3).
* **Impact:** Reduces model size by **~38%** (from 5.2 MB to 3.2 MB) compared to the baseline.

### 3. Neck: Fine & Medium Granularity (MGDA Core) 🔗
Instead of simple feature fusion, the Neck solves two auxiliary tasks via **Forward Hooks**:

* **Fine Granularity (Layer 8): Super-Resolution Decoder**
    * **Goal:** Small Object Recovery.
    * **Mechanism:** Forces the backbone to reconstruct the original high-res image from compressed features. This prevents the network from "pooling away" small targets like blood cells.
* **Medium Granularity (Layer 12): Entropy Minimization**
    * **Goal:** Attention Consistency.
    * **Mechanism:** Penalizes "fuzzy" or uncertain feature maps. Forces the model to have sharp, confident focus on object instances.

### 4. Head: Coarse Granularity & Speed ⚡

* **Coarse Granularity (Layer 27): Adversarial GRL**
    * **Module:** **Gradient Reversal Layer (GRL)**.
    * **Mechanism:** An adversarial classifier tries to guess the domain (Source vs. Target). The GRL flips the gradients to fool it, forcing the head to learn **Domain-Invariant Features**.
* **Optimization:** **Structured Pruning**
    * **Mechanism:** Physically removed redundant filters from the detection head to reduce GFLOPs.

---

## 📊 Performance Benchmark (Jetson Orin)

We evaluated **Frankenstein_MGDA** across multiple datasets (including PBC). The model was converted to **TensorRT (INT8)** for final deployment.

| Model Mode | Size (MB) | FPS (Jetson Orin) | Inference Time | Power |
| :--- | :--- | :--- | :--- | :--- |
| **PyTorch (Baseline)** | 5.24 MB | 28.22 FPS | 35.43 ms | 15W |
| **FP32 (TensorRT)** | 7.23 MB | 50.30 FPS | 19.88 ms | 15W |
| **FP16 (TensorRT)** | 5.42 MB | 61.57 FPS | 16.24 ms | 15W |
| **INT8 (Frankenstein)** | **3.26 MB** | **63.41 FPS** | **15.77 ms** | **15W** |

> **Result:** We achieved a **2.25x Speedup** and **38% Size Reduction** compared to the PyTorch baseline.

---

## 🛠️ Installation & Usage

### 1. Requirements

pip install ultralytics torch torchvision opencv-python psutil






---
---
---


## 👥 Contributors

| Name      | GitHub Profile |
|-----------|----------------|
| Harish K  | [rkhm2004](https://github.com/rkhm2004) |
| Suman     | [Suman-Maitreya](https://github.com/Suman-Maitreya) |
| Rajkumar  | [RajkumarR2006](https://github.com/RajkumarR2006) |

