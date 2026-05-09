# 🧠 Multi-Modal Deep Learning Framework For Brain Tumor Detection And 3D Volumetric Segmentation Using Optimized UNet Architectures 

<div align="center">

![Brain Tumor Segmentation](https://img.shields.io/badge/Task-Medical%20Image%20Segmentation-blue?style=for-the-badge)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch)
![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python)
![BraTS2020](https://img.shields.io/badge/Dataset-BraTS2020-green?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**Published in:** *Journal of Electronics and Information Technology (JEIT), Volume 26, Issue 4, 2026 — Scopus/Elsevier*

**Author:** Shaik Mohammed Feroz · M.Tech in Artificial Intelligence · JAIN (Deemed-to-be University)

**Guide:** Dr. Guruvammal S · Assistant Professor, Dept. of CSE · JAIN (Deemed-to-be University)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Results](#-key-results)
- [Pipeline Architecture](#-pipeline-architecture)
- [UNet Architecture](#-unet-architecture)
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Usage](#-usage)
- [Training](#-training)
- [Evaluation Metrics](#-evaluation-metrics)
- [Visualizations](#-visualizations)
- [Comparison with Literature](#-comparison-with-literature)
- [Project Structure](#-project-structure)
- [Future Scope](#-future-scope)
- [Citation](#-citation)

---

## 🔭 Overview

Brain tumors such as glioblastoma multiforme (GBM) carry a median survival of just 14–16 months even with aggressive treatment. Radiologists currently spend **3–5 hours per patient** manually tracing tumor boundaries across multi-modal MRI scans — a tedious and variable process.

This project presents a **complete, automated end-to-end deep learning pipeline** that:

- Accepts all **4 MRI modalities** (T1, T1ce, T2, FLAIR) simultaneously as a 4-channel input
- Produces **pixel-level binary tumor masks** with a best validation Dice of **0.9451**
- Segments a full patient volume (155 slices) in **under 30 seconds** vs. 3–5 hours manually
- Provides **Grad-CAM explainability**, **3D volumetric visualization**, and **automated severity triage**

> 📄 The paper is published in **JEIT Vol 26, Issue 4, 2026** (Scopus/Elsevier, ISSN: 1009-5896, Impact Factor: 6.1)

---

## 🏆 Key Results

| Metric | Value |
|---|---|
| **Best Validation Dice Score** | **0.9451** |
| **Mean Validation Dice** | 0.8314 |
| **Validation Accuracy** | 99.79% |
| **Precision** | 89.81% |
| **Recall (Sensitivity)** | 82.67% |
| **IoU (Jaccard)** | 76.99% |
| **AUC-ROC** | **0.9988** |
| **Post-Processing Dice** | **0.9537** |
| **Inference Time per Slice** | ~0.19 seconds |
| **Total Parameters** | 7,703,553 |
| **Training Hardware** | NVIDIA GTX 1650 (4.3 GB VRAM) |

---

## 🔄 Pipeline Architecture

The system is organized into **6 stages** and **7 phases**:

```
Stage 1: Data Acquisition & Preprocessing
    BraTS2020 (57,195 slices, 369 patients)
    └── HDF5 loading → Tumor-slice filtering → Min-Max normalization

Stage 2: Augmentation & Splitting
    85% Train (1,275 slices) / 15% Validation (225 slices)
    └── Random H-flip (50%) + V-flip (50%)

Stage 3: U-Net Model (7,703,553 params)
    Encoder: 4 → 64 → 128 → 256 ch (DoubleConv + MaxPool)
    Bottleneck: 512 ch + Dropout2d (p=0.3)
    Decoder: ConvTranspose2d + Skip Connections → Sigmoid output

Stage 4: Training Strategy
    Loss: 0.5 × Dice + 0.5 × BCE
    Optimizer: Adam (lr=1e-4, weight_decay=1e-5)
    Scheduler: ReduceLROnPlateau (factor=0.5, patience=3)
    Early Stopping: patience=5

Stage 5: Evaluation & Post-Processing
    Metrics: Dice, IoU, Precision, Recall, AUC, Confusion Matrix
    Morphological: Opening (remove noise) + Closing (fill holes), 3×3 kernel

Stage 6: Interpretability & Visualization
    Grad-CAM heatmaps → 3D marching cubes (Plotly) → Severity reports
```

> 📷 **Figure:** Six-stage pipeline diagram (see `images/pipeline.png`)

![Pipeline Diagram](images/pipeline.png)

---

## 🏗️ UNet Architecture

A **3-level UNet** encoder-decoder with skip connections:

```
Input: (B, 4, 240, 240)  ← 4 MRI modalities
│
├── Encoder
│   ├── Down1: DoubleConv(4→64)   → (240×240)   [39,424 params]
│   ├── Pool1: MaxPool2d(2)        → (120×120)
│   ├── Down2: DoubleConv(64→128)  → (120×120)   [221,696 params]
│   ├── Pool2: MaxPool2d(2)        → (60×60)
│   ├── Down3: DoubleConv(128→256) → (60×60)     [886,272 params]
│   └── Pool3: MaxPool2d(2)        → (30×30)
│
├── Bottleneck: DoubleConv+Dropout(256→512) → (30×30)  [3,542,016 params]
│
└── Decoder
    ├── Up3: ConvTranspose2d(512→256) + skip → (60×60)
    ├── Conv3: DoubleConv(512→256)             [1,770,496 params]
    ├── Up2: ConvTranspose2d(256→128) + skip → (120×120)
    ├── Conv2: DoubleConv(256→128)             [443,136 params]
    ├── Up1: ConvTranspose2d(128→64)  + skip → (240×240)
    ├── Conv1: DoubleConv(128→64)              [110,848 params]
    └── Final: Conv2d(64→1, 1×1) + Sigmoid  → (B, 1, 240, 240)

Total Trainable Parameters: 7,703,553
```

Each **DoubleConv block**: `Conv3×3 → BatchNorm2d → ReLU → Conv3×3 → BatchNorm2d → ReLU`

> 📷 **Figure:** UNet encoder-decoder diagram (see `images/unet_architecture.png`)

![UNet Architecture](images/unet_architecture.png)

---

## 📁 Dataset

**BraTS2020** — Brain Tumor Segmentation Challenge 2020

| Property | Details |
|---|---|
| Patients | 369 (293 HGG + 76 LGG) |
| Total Slices | 57,195 axial slices |
| Slice Resolution | 240 × 240 pixels |
| MRI Modalities | T1, T1ce, T2, FLAIR |
| Tumor Sub-regions | Edema (28.3%), Necrotic Core (44.9%), Enhancing Tumor (26.8%) |
| Format | HDF5 (.h5), one file per 2D slice |
| Training Subset | 1,500 tumor-containing slices |

### MRI Modalities

| Modality | Full Name | Clinical Role |
|---|---|---|
| T1 | T1-weighted | Anatomical structure; grey/white matter |
| T1ce | T1 + Contrast | Highlights active enhancing tumor core |
| T2 | T2-weighted | Sensitive to water; shows edema |
| FLAIR | Fluid-Attenuated Inversion Recovery | Suppresses CSF; reveals infiltration |

> 📷 **Figure:** Sample BraTS2020 HDF5 slice showing all 4 modalities + ground truth (see `images/sample_mri_slice.png`)

![Sample MRI Slice](images/sample_mri_slice.png)

Download the BraTS2020 dataset from the [official challenge page](https://www.med.upenn.edu/cbica/brats2020/data.html).

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/brain-tumor-detection-unet.git
cd brain-tumor-detection-unet

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision
pip install numpy pandas matplotlib seaborn
pip install scikit-learn scikit-image
pip install h5py opencv-python plotly
pip install tqdm psutil torchsummary
```

### Requirements

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.23.0
h5py>=3.7.0
opencv-python>=4.7.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.14.0
scikit-learn>=1.2.0
scikit-image>=0.20.0
pandas>=2.0.0
tqdm>=4.65.0
psutil>=5.9.0
```

---

## 🚀 Usage

### 1. Prepare Dataset

```python
# Set your dataset path in the notebook/script
DATA_DIR = "./dataset"   # folder containing .h5 files
SUBSET_SIZE = 1500       # number of tumor-containing slices to use
```

### 2. Run the Full Pipeline

Open and run the Jupyter notebook:

```bash
jupyter notebook Brain_Tumor_Detection_and_3D_Segmentation_using_Deep_Learning.ipynb
```

Or run as a Python script:

```bash
python train.py
```

### 3. Inference on a Single Slice

```python
import torch
from model import UNet

model = UNet()
model.load_state_dict(torch.load("best_unet_model.pth"))
model.eval()

# image: torch.Tensor of shape (1, 4, 240, 240)
with torch.no_grad():
    prediction = model(image)
    binary_mask = (prediction > 0.5).float()
```

---

## 🏋️ Training

### Hyperparameters

| Parameter | Value |
|---|---|
| Epochs | 20 |
| Batch Size | 8 |
| Learning Rate | 1e-4 |
| Weight Decay | 1e-5 |
| LR Scheduler | ReduceLROnPlateau (factor=0.5, patience=3) |
| Early Stopping Patience | 5 |
| Dropout (Bottleneck) | 0.3 |
| Loss Function | 0.5 × Dice + 0.5 × BCE |
| Optimizer | Adam |
| Train/Val Split | 85% / 15% |

### Combined Loss Function

```python
def dice_loss(pred, target, smooth=1e-6):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def combined_loss(pred, target):
    bce = F.binary_cross_entropy(pred, target)
    return 0.5 * dice_loss(pred, target) + 0.5 * bce
```

### Training Performance (Epoch-wise)

| Epoch | Train Loss | Val Loss | Train Dice | Val Dice |
|---|---|---|---|---|
| 1 | 0.6236 | 0.5646 | 0.7608 | 0.8140 |
| 5 | 0.3870 | 0.3640 | 0.8877 | 0.8968 |
| 10 | 0.1496 | 0.1340 | 0.9227 | 0.9254 |
| 15 | 0.0703 | 0.0674 | 0.9324 | 0.9394 |
| **19** | **0.0545** | **0.0468** | **0.9330** | **0.9451** |
| 20 | 0.0502 | 0.0499 | 0.9372 | 0.9415 |

> 📷 **Figure:** Training & validation loss/Dice curves (see `images/training_curves.png`)

![Training Curves](images/training_curves.png)

---

## 📊 Evaluation Metrics

### Validation Metrics (Best Checkpoint — Epoch 19)

| Metric | Value | Interpretation |
|---|---|---|
| Accuracy | 0.9979 | 99.79% of all pixels correctly classified |
| Precision | 0.8981 | 89.81% of predicted tumor pixels are actual tumor |
| Recall | 0.8267 | 82.67% of actual tumor pixels detected |
| IoU (Jaccard) | 0.7699 | 76.99% overlap between prediction and ground truth |
| Dice Score (Mean) | 0.8314 | 83.14% mean Dice across validation slices |
| AUC-ROC | **0.9988** | Near-perfect tumor vs. background discrimination |

### Post-Processing Results (Single Sample)

| Metric | Cleaned Prediction |
|---|---|
| Dice | **0.9537** |
| IoU | 0.9115 |
| Precision | **0.9922** |
| Recall | 0.9181 |
| Accuracy | 0.9983 |

### Confusion Matrix (200,000 subsampled pixels)

```
                Predicted Background    Predicted Tumor
True Background       195,679                  165
True Tumor               302                 3,854
```

> 📷 **Figures:** ROC curve (see `images/roc_curve.png`) · Confusion matrix (see `images/confusion_matrix.png`) · Prediction vs Ground Truth (see `images/prediction_vs_gt.png`)

![ROC Curve](images/roc_curve.png)
![Confusion Matrix](images/confusion_matrix.png)
![Prediction vs GT](images/prediction_vs_gt.png)

---

## 🖼️ Visualizations

### Grad-CAM Explainability

Grad-CAM at the UNet bottleneck confirms that the model attends to genuine tumor anatomy — **not** imaging artifacts.

> 📷 See `images/gradcam_heatmap.png`

![Grad-CAM](images/gradcam_heatmap.png)

### 3D Volumetric Reconstruction

Interactive 3D tumor surface using **marching cubes** + **Plotly** with per-region color coding:

| Color | Region |
|---|---|
| 🟡 Yellow | Edema |
| 🔴 Red | Necrotic Core |
| 🔵 Cyan | Enhancing Tumor |

> 📷 See `images/3d_visualization.png`

![3D Visualization](images/3d_visualization.png)

### Automated Severity Classification

Patients are categorized into severity tiers based on predicted tumor burden:

| Severity | Tumor Involvement | Patients (N=369) |
|---|---|---|
| Low | < 20% | 5 (1.4%) |
| Medium | 20–50% | 267 (72.4%) |
| **High** | **> 50%** | **97 (26.3%)** |

> 📷 See `images/severity_distribution.png` · `images/tumor_region_pie.png` · `images/dice_score_distribution.png`

![Severity Distribution](images/severity_distribution.png)

---

## 📚 Comparison with Literature

| Method | Architecture | Dice (Whole Tumor) | Dataset |
|---|---|---|---|
| Havaei et al. (2017) | Two-pathway CNN | 0.83 | BraTS 2013 |
| Isensee et al. (2018) | Standard UNet | 0.88 | BraTS 2017 |
| Myronenko (2019) | 3D Enc-Dec + VAE | 0.91 | BraTS 2018 |
| **This Work** | **2D UNet (subset)** | **0.83 (mean) / 0.94 (median)** | **BraTS 2020** |
| Hatamizadeh et al. (2022) | Swin UNETR | 0.93 | BraTS 2021 |

Our model is **competitive with published baselines** despite training on only 1,500 of 57,195 slices on a consumer GPU (GTX 1650, 4.3 GB VRAM).

---

## 📂 Project Structure

```
brain-tumor-detection-unet/
│
├── 📓 Brain_Tumor_Detection_and_3D_Segmentation_using_Deep_Learning.ipynb
├── 📄 README.md
├── 📄 requirements.txt
│
├── 📁 dataset/                    # BraTS2020 .h5 files (not included)
│   ├── volume_0_slice_0.h5
│   └── ...
│
├── 📁 models/
│   ├── best_unet_model.pth        # Best checkpoint (lowest val loss)
│   └── unet_brain_tumor_final.pth # Final epoch checkpoint
│
├── 📁 images/                     # All result figures for README
│   ├── pipeline.png
│   ├── unet_architecture.png
│   ├── sample_mri_slice.png
│   ├── training_curves.png
│   ├── roc_curve.png
│   ├── confusion_matrix.png
│   ├── prediction_vs_gt.png
│   ├── gradcam_heatmap.png
│   ├── 3d_visualization.png
│   ├── severity_distribution.png
│   ├── tumor_region_pie.png
│   └── dice_score_distribution.png
│
└── 📁 outputs/
    └── Brain_Tumor_Full_Patient_Summary.csv
```

---

## 🔮 Future Scope

- **Multi-Class Segmentation:** Extend from binary to 4-class (background, edema, necrotic core, enhancing tumor)
- **3D Volumetric UNet:** Replace 2D slice-by-slice with a 3D U-Net / V-Net for inter-slice context
- **Attention Mechanisms:** Integrate Attention UNet or Swin UNETR transformer bottleneck
- **Full Dataset Training:** Scale to all 57,195 slices with mixed-precision / gradient checkpointing
- **Uncertainty Estimation:** Monte Carlo Dropout for per-pixel confidence maps
- **Federated Learning:** Multi-institutional training without sharing patient data (HIPAA/GDPR compliant)
- **PACS Integration:** DICOM-compatible plugin for hospital radiology workflows (TensorRT / ONNX)

---

## 📖 Citation

If you find this work useful, please cite:

```bibtex
@article{feroz2026multimodal,
  title     = {Multi-Modal Deep Learning Framework for Brain Tumor Detection and
               3D Volumetric Segmentation using Optimized UNet Architectures},
  author    = {Shaik Mohammed Feroz and Guruvammal S},
  journal   = {Journal of Electronics and Information Technology},
  volume    = {26},
  number    = {4},
  year      = {2026},
  publisher = {Scopus/Elsevier},
  issn      = {1009-5896},
  note      = {Paper ID: JEIT/1244}
}
```

---

## 🙏 Acknowledgements

- **BraTS Challenge** organizers for the benchmark dataset
- **JAIN (Deemed-to-be University)**, Faculty of Engineering & Technology
- Guide: **Dr. Guruvammal S**, Dept. of CSE
- M.Tech Program Head: **Dr. Ajay Kumar Singh**

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Made with ❤️ by **Shaik Mohammed Feroz** · [shaikmohammedferoz47@gmail.com](mailto:shaikmohammedferoz47@gmail.com)

⭐ Star this repo if you found it helpful!

</div>
