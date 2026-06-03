# 🫁 Weakly-Supervised Localization with Grad-CAM on VinDr-PCXR

> *Turning a multi-label chest X-ray classifier into a pseudo-detection model — no bounding box supervision required.*

---

## Overview

Medical AI models are notoriously difficult to trust — radiologists need to know **where** a model is looking, not just **what** it predicts. This project builds an Explainable AI (XAI) pipeline on the [VinDr-PCXR](https://vindr.ai/datasets/pediatric-chest-x-ray) dataset: the first large-scale pediatric chest X-ray dataset with both image-level disease labels and radiologist-drawn bounding boxes.

A **DenseNet-121** classifier is trained on 15 pediatric lung diseases using only image-level labels (no box supervision). At inference, **Grad-CAM** generates class-specific heatmaps that highlight the regions driving each prediction. These heatmaps are then evaluated against radiologist bounding boxes — turning a classifier into a quantitatively evaluated pseudo-detector.

---

## Results

### Classification (15 diseases, multi-label)

| Metric | Value |
|---|---|
| Mean AUC (all classes) | **> 0.80** |
| Loss | Binary Cross-Entropy with class-weighted pos_weight |
| Backbone | DenseNet-121 (ImageNet pretrained) |

### XAI Localization (Grad-CAM vs. radiologist boxes)

| Metric | Value |
|---|---|
| Overall Pointing Game Accuracy | < 0.40 |
| Evaluation strategy | Top-3 predicted disease heatmaps vs. union of GT finding boxes |
| Heatmap threshold | 0.40 |

> **Note on Pointing Game score:** The labels CSV (15 disease classes) and box annotation CSV (36 radiological finding classes) use entirely different taxonomies with zero class overlap — a known characteristic of VinDr-PCXR. The XAI evaluation therefore measures whether disease-level heatmaps activate over finding-level boxes in the same image, making this a cross-taxonomy localization task and a harder, more realistic benchmark than standard Pointing Game setups.

---

## Method

```
DICOM → CLAHE normalisation → 224×224 RGB
         ↓
   DenseNet-121 backbone
   features.denseblock4  ← GradCAM hook
         ↓
   GlobalAveragePool → Dropout(0.5) → FC(15) → Sigmoid
         ↓
   BCEWithLogitsLoss (class-weighted pos_weight)
         ↓
   Grad-CAM heatmap per predicted class
         ↓
   Threshold → binary mask → bounding box → IoU / Pointing Game
```

**Key design decisions:**
- `features.denseblock4` is the GradCAM target layer — the deepest spatial feature map before GAP collapses spatial information
- `pos_weight = neg_count / pos_count` per class to handle severe label imbalance
- On-the-fly DICOM decoding via SimpleITK (dataset > 19GB, no intermediate PNGs saved)
- `aug_smooth=True` + `eigen_smooth=True` in GradCAM for cleaner heatmaps

---

## Dataset

**VinDr-PCXR** — [vindr.ai/datasets/pediatric-chest-x-ray](https://vindr.ai/datasets/pediatric-chest-x-ray)

| Split | Images |
|---|---|
| Train | ~9,000 |
| Test | ~500 |

- **15 disease labels** (image-level, multi-label)
- **36 finding classes** with radiologist bounding boxes
- Access via [PhysioNet](https://physionet.org/content/vindr-pcxr/1.0.0/) (requires credentialed account + DUA)

---

## Repository Structure

```
vindr-pcxr-gradcam/
├── notebook/
│   └── vindr_pcxr_gradcam.ipynb   # Full Kaggle training notebook
├── app/
│   └── app.py                     # Streamlit inference UI
├── assets/
│   └── sample_heatmaps/           # Example Grad-CAM overlays
├── requirements.txt
└── README.md
```

---

## Quickstart

### Requirements
```bash
pip install torch torchvision pytorch-grad-cam albumentations SimpleITK opencv-python scikit-learn
```

### Run on Kaggle
1. Add VinDr-PCXR as a Kaggle dataset
2. Open `notebook/vindr_pcxr_gradcam.ipynb`
3. Enable **GPU T4 x2** accelerator
4. Run all cells — artifacts saved to `/kaggle/working/outputs/`

### Streamlit App
```bash
cd app
streamlit run app.py
# Upload any chest X-ray PNG → get disease probabilities + Grad-CAM overlay
```

---

## Limitations & Future Work

- The cross-taxonomy mismatch between disease labels and finding boxes makes quantitative XAI evaluation non-trivial; a unified annotation scheme would enable cleaner benchmarking
- Pointing Game accuracy could be improved with: test-time augmentation ensembling, larger input resolution (512×512), or fine-tuning on CheXNet weights instead of ImageNet
- Multi-scale Grad-CAM (averaging across multiple DenseNet blocks) may produce tighter localizations
- The model is trained on a Vietnamese pediatric cohort — domain shift to other populations should be evaluated before clinical use

---

## Citation

If you use VinDr-PCXR, please cite the original dataset:

```bibtex
@article{pham2022vindr,
  title     = {VinDr-PCXR: An open, large-scale pediatric chest X-ray dataset for interpretation of common thoracic diseases},
  author    = {Pham, Hieu H. and others},
  journal   = {medRxiv},
  year      = {2022}
}
```

---

## License

This project is released under the MIT License. The VinDr-PCXR dataset has its own access terms via PhysioNet — do not redistribute raw data.
