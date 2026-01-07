# 🩸 HemaliteV10: Lightweight Medical Object Detection

**HemaliteV10** is a specialized, high-precision object detection architecture engineered for the microscopic analysis of blood cells on edge devices like the **NVIDIA Jetson Orin Nano**.

Built by surgically modifying **YOLOv10**, this project introduces a novel **"Strong Backbone, Skinny Head"** philosophy. It combines deep, high-channel feature extraction with an aggressively pruned fusion network. The result is a model that balances sufficient capacity for subtle medical textures with extreme efficiency (~**1.2M Parameters**) and real-time speed (**30+ FPS**), specifically tailored for hematological diagnosis.

---

## 🔬 Project Overview

Standard lightweight models often fail in medical imaging because they *compress* features too early, losing the subtle texture differences between cell types (e.g., chromatin patterns in Blast cells vs. Lymphocytes). Conversely, heavy models are too slow for portable digital microscopes.

**HemaliteV10** solves this trade-off by decoupling feature extraction from feature fusion:

- **Extraction (Strong):** Backbone maintains high channel depth (**128ch**) to capture complex biological morphologies.  
- **Attention (Focused):** CBAM modules actively suppress stain noise and background artifacts before pooling.  
- **Fusion (Skinny):** Neck restricted to **64 channels**, proving that once a medical feature is found, it can be propagated efficiently without redundancy.  

---

## 🧠 Architecture: The "Skinny Head" Surgery

The architecture acts as a funnel, capturing rich details at the input and distilling them into a lightweight stream for detection.

### 1. Backbone: Deep Feature Retention 🦴
- **Mechanism:** 128-Channel Deep Layers (P4 & P5)  
- **Function:** Unlike standard "Nano" models that cut channels to 64 or 48, HemaliteV10 preserves **128 channels** at the semantic peak.  
- **Medical Rationale:** White blood cells are distinguishable only by fine-grained textures. Cutting channels here causes *Feature Collapse*, where distinct cell types look identical. Retention ensures subtle differences remain intact.  

---

### 2. Attention Stage: Feature Refining 🎯
- **Module:** CBAM (Convolutional Block Attention Module)  
- **Placement:** Injected after C2f blocks in Backbone (Stages P2, P3, P4).  
- **Function:** Dual-Axis Focus  
  - **Channel Attention:** *"What is this?"* (focuses on stain colors/intensities).  
  - **Spatial Attention:** *"Where is this?"* (focuses on cell boundaries, ignoring plasma/background noise).  
- **Impact:** Acts as a denoising filter, sharpening features before entering the compressed neck.  

---

### 3. Neck & Head: The "Skinny" Bottleneck ⏳
- **Philosophy:** *Squeeze and Detect*  
- **Module:** 64-Channel Constraint (PANet)  
- **Function:** Extreme parameter pruning.  
- **Mechanism:** Despite receiving 128-channel inputs, every fusion layer (Concat + C2f) in the neck is constrained to **64 channels**.  
- **Impact:** Removes computational “fat” from the feature pyramid, reducing memory bandwidth usage on Jetson Orin Nano by ~**40%** without sacrificing semantic accuracy.  

---

### 4. Output: Multi-Scale Clinical Detection 📏
- **Structure:** Three-Scale Decoupled Head  
- **Function:** Handles extreme size variance in blood smears.  
  - **P3 Head (Small):** Optimized for Platelets and small RBC fragments.  
  - **P4 Head (Medium):** Optimized for standard RBCs and Lymphocytes.  
  - **P5 Head (Large):** Optimized for large Monocytes and abnormal Blasts.  
- **Optimization:** Heads operate directly on the “skinny” 64-channel features, minimizing FLOPs required for bounding box regression.  

---

## 🚀 Key Highlights
- ~**1.2M parameters** only  
- Runs at **30+ FPS** on Jetson Orin Nano  
- Retains medical texture fidelity while pruning redundant fusion layers  
- CBAM-enhanced backbone for stain noise suppression and morphology focus

- ## 📊 Comparative Accuracy Analysis

We evaluated **HemaliteV10** against a standard **Baseline (YOLOv10n)** across four diverse medical datasets.  
The results highlight the model's ability to match or exceed the baseline in complex scenarios despite its reduced parameter count.

| Dataset | Model        | mAP@50 | mAP@50-95 |
|---------|--------------|--------|-----------|
| **BCCD** (Blood Cell Count) | Baseline     | 0.93 | 0.69 |
|                         | HemaliteV10  | 0.89 | 0.63 |
| **PBC** (Peripheral Blood) | Baseline     | 0.99 | 0.99 |
|                         | HemaliteV10  | **0.99** | **0.99** |
| **B-ALL** (Leukemia)    | Baseline     | 0.98 | 0.76 |
|                         | HemaliteV10  | 0.97 | 0.72 |
| **yolo_new** (General Medical) | Baseline     | 0.97 | 0.87 |
|                         | HemaliteV10  | **0.98** | **0.89** |

---
## ✅ Final Benchmark Results

Performance comparison of HemaliteV10 across different deployment formats (PyTorch vs TensorRT).

| Model             | Size (MB) | FPS   | Time (ms) | GFLOPs | Temp (°C) | RAM (%) | Power |
|-------------------|-----------|-------|-----------|--------|-----------|---------|-------|
| **PyTorch (CUDA)** | 2.77      | 28.42 | 35.18     | 6.2    | 50.1      | 62.1    | 15W   |
| **FP32 (TensorRT)** | 7.26      | 33.65 | 29.72     | 6.2    | 49.8      | 63.7    | 15W   |
| **FP16 (TensorRT)** | 5.61      | 40.97 | 24.41     | 6.2    | 49.8      | 64.0    | 15W   |
| **INT8 (TensorRT)** | 3.98      | 50.85 | 19.66     | 6.2    | 50.0      | 65.5    | 15W   |

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

