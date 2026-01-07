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



---


# 🧟 Frankenstein_MGDA: Multi-Granularity Domain Adaptation

**Frankenstein_MGDA** is a specialized, high-performance object detection architecture designed for robust deployment on edge devices like the **NVIDIA Jetson Orin**.

Built by surgically modifying **YOLOv11n**, this project "stitches" together state-of-the-art efficiency modules (SqueezeNet Fire Modules) with advanced **Multi-Granularity Domain Adaptation (MGDA)**. The result is a model that achieves extreme compression (**~46.9% Parameter Reduction**) and real-time speed (**~63 FPS**) while maintaining high accuracy across diverse medical datasets.

---

## 🔬 Project Overview

Automated cell detection often fails when moving between different microscopes or lighting conditions (Domain Shift). Standard lightweight models are either too heavy for edge deployment or too shallow to be robust.

**Frankenstein_MGDA** solves this by introducing a **"Strong Backbone, Multi-Task Neck"** architecture. It forces the network to align features at three distinct levels of granularity:
1.  **Spectral (Input):** Removing style bias via Fourier Transforms.
2.  **Fine (Pixel):** Preserving small object details via Super-Resolution.
3.  **Coarse (Global):** Enforcing semantic consistency via Adversarial Learning.

---

## 🧠 Architecture: The "Frankenstein" Surgery

The architecture is a hybrid system, replacing standard YOLO components with specialized modules to balance speed, size, and robustness.

### 1. Input Stage: Spectral Granularity (FFT) 🌊
* **Module:** **Fourier Mixing Block**
* **Function:** Before the network sees an image, we apply a **Fast Fourier Transform (FFT)** to decompose it into **Amplitude** (Style) and **Phase** (Structure).
* **Mechanism:** The model dynamically swaps the Amplitude spectrum of the training batch with others. This mathematically removes "style" bias (lighting/color), forcing the model to learn pure structure.

### 2. Backbone: The "Fire" Compression 🔥
* **Module:** **SqueezeNet Fire Modules** (Replacing `C2f`)
* **Function:** Extreme Parameter Reduction.
* **Mechanism:** Uses a "Squeeze" layer (1x1 Conv) to compress channel depth before feeding into parallel "Expand" layers (1x1 & 3x3).
* **Impact:** Reduces parameters from **2.58M** to **1.37M** (~47% reduction).

### 3. Neck: Fine & Medium Granularity (MGDA Core) 🔗
Instead of simple feature fusion, the Neck solves two auxiliary tasks via **Forward Hooks**:
* **Layer 8 (Super-Resolution):** Forces the backbone to reconstruct the original high-res image, preventing the loss of small cells.
* **Layer 12 (Entropy Minimization):** Penalizes "fuzzy" feature maps, forcing sharp attention on object instances.

### 4. Head: Coarse Granularity & Speed ⚡
* **Adversarial GRL (Layer 27):** A Gradient Reversal Layer flips gradients during training to fool a domain classifier, creating **Domain-Invariant Features**.
* **Pruning:** Structured pruning removes redundant filters from the detection head to maximize **FPS**.

---

## 📊 Comparative Accuracy Analysis

We evaluated **Frankenstein_MGDA** against a standard **Baseline (YOLOv11n)** across three diverse medical datasets. The results highlight the model's ability to maintain high accuracy despite a massive reduction in model size.

| Dataset / Task | Model Architecture | mAP 0.5 (%) | Difference | Parameters (M) | % Reduction |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Stage 1: BCCD**<br>*(Object Detection)* | Baseline (YOLO11n)<br>**Hybridized (Fire + MGDA)** | 92.9%<br>**89.6%** | -<br>-3.3% | 2.58 M<br>**1.37 M** | -<br>**46.9%** |
| **Stage 2: PBC**<br>*(Subtype Classification)* | Baseline (YOLO11n)<br>**Hybridized (Fire + MGDA)** | 99.5%<br>**99.4%** | -<br>-0.1% | 2.58 M<br>**1.37 M** | -<br>**46.9%** |
| **Stage 3: B-ALL**<br>*(Cancer Detection)* | Baseline (YOLO11n)<br>**Hybridized (Fire + MGDA)** | 98.4%<br>**98.2%** | -<br>-0.2% | 2.58 M<br>**1.37 M** | -<br>**46.9%** |

> **Key Insight:** On the PBC and B-ALL datasets, the **Frankenstein** model achieved nearly identical accuracy (within 0.2%) to the baseline while being **half the size**.

---

## 🚀 Speed & Efficiency (Jetson Orin)

The final model was exported to **TensorRT (INT8)** for deployment.

| Model Mode | Size (MB) | FPS (Jetson Orin) | Inference Time | Power |
| :--- | :--- | :--- | :--- | :--- |
| **PyTorch (Baseline)** | 5.24 MB | 28.22 FPS | 35.43 ms | 15W |
| **FP32 (TensorRT)** | 7.23 MB | 50.30 FPS | 19.88 ms | 15W |
| **FP16 (TensorRT)** | 5.42 MB | 61.57 FPS | 16.24 ms | 15W |
| **INT8 (Frankenstein)** | **3.26 MB** | **63.41 FPS** | **15.77 ms** | **15W** |

---





## 👥 Contributors

| Name      | GitHub Profile |
|-----------|----------------|
| Harish K  | [rkhm2004](https://github.com/rkhm2004) |
| Suman     | [Suman-Maitreya](https://github.com/Suman-Maitreya) |
| Rajkumar  | [RajkumarR2006](https://github.com/RajkumarR2006) |

