# ForensicsSAM: Toward Robust and Unified Image Forgery Detection and Localization Resisting to Adversarial Attack
<!--%[![Paper](https://img.shields.io/badge/Paper-PDF-red)](link-to-your-paper)!-->
[![arXiv](https://img.shields.io/badge/arXiv-2508.07402-b31b1b.svg)](https://arxiv.org/pdf/2508.07402)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

PyTorch implementation of the **New and Improved Version** of the paper.

---

## 📌 Abstract
<p align="center">
  <img src="src/intro.png" width="500" /><br>
</p>
Parameter-efficient fine-tuning (PEFT) has emerged as a popular strategy for adapting large vision foundation models—such as the Segment Anything Model (SAM) and LLaVA—to downstream tasks like image forgery detection and localization (IFDL). However, existing PEFT-based approaches often **overlook their vulnerability to adversarial attacks**.  
We show that **highly transferable adversarial images** can be crafted solely via the upstream model—without accessing the downstream model or training data—significantly degrading IFDL performance.

To address this, we propose **ForensicsSAM**, a unified IFDL framework with built-in adversarial robustness, guided by three key ideas:

1. **Shared Forgery Experts**  
   - To compensate for the lack of forgery-relevant knowledge in the frozen image encoder, we insert forgery experts into each transformer block.  
   - These experts are **always active** and **shared** across any input images, enhancing the encoder’s ability to capture forgery artifacts.

2. **Light-weight Adversary Detector**  
   - Learns to capture **structured, task-specific artifacts** in the RGB domain.  
   - Enables reliable detection of adversarial images across various attack methods.

3. **Adaptive Adversary Experts**  
   - Injected into the **global attention layers** and **MLP modules** to progressively correct feature shifts induced by adversarial noise.  
   - **Adaptively activated** by the adversary detector, avoiding unnecessary interference with clean images.

Extensive experiments across multiple benchmarks demonstrate that **ForensicsSAM** not only achieves superior resistance to diverse adversarial attacks, but also delivers **state-of-the-art performance** in both image-level forgery detection and pixel-level forgery localization.

---

## 📂 Project Structure
```
ForensicsSAM-released/
├── adversary_detector/    # Adversary detector module
├── data/                  # Dataset text lists
├── forensics_sam/         # Core ForensicsSAM model implementation
├── mini_dataloader/       # dataloader
├── segment_anything/      # SAM backbone
├── utils/                 # Helper functions and utilities
├── weight/                # Pretrained model weights
├── inference.py           # Inference script
└── README.md              # Project description
```

---

## 📋 Method Overview
<p align="center">
  <img src="src/ForensicsSAM.png" width="800" /><br>
  <em>Figure 1: Overview of the proposed ForensicsSAM framework. Given an input image, ForensicsSAM outputs the image-level detection results (real or forged, clean or adversarial) as well as a pixel-level forgery mask.</em>
</p>

```
weight/
  ├── adversary_detector.pth
  ├── adversary_experts.pth
  ├── forgery_experts.pth
  ├── sam_vit_h_4b8939.pth
```

3. you can download the pre-trained weight from [google drive](https://drive.google.com/file/d/1stLg8bJ1W2E7dVAHC8TYj917REO4sttt/view)
---

## 💻 Training
```bash
python train.py
```

## 💻 Inference
```bash
python inference.py
```

---

## 🙏 Acknowledgement
- This work is built upon the [SAM](https://github.com/facebookresearch/segment-anything).
