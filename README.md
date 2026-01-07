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
---
---
## 👥 Contributors

| Name      | GitHub Profile |
|-----------|----------------|
| Harish K  | [rkhm2004](https://github.com/rkhm2004) |
| Suman     | [Suman-Maitreya](https://github.com/Suman-Maitreya) |
| Rajkumar  | [RajkumarR2006](https://github.com/RajkumarR2006) |

