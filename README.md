# UNet — Breast Ultrasound Tumour Segmentation

Semantic segmentation of breast tumours from ultrasound images using a UNet trained from scratch on the BUSI dataset.

---

## Dataset

🔗[https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)

| Split | Images |
|-------|--------|
| Train | ~546   |
| Val   | ~234   |
| Total | 780 (benign · malignant · normal) |

---

## Current Results *(Epoch 31 / 50)*

| Metric    | Score  |
|-----------|--------|
| DSC       | 0.6116 |
| IoU       | 0.4512 |
| Precision | 0.5820 |
| Recall    | 0.6765 |
| Train Loss| 0.5619 |
| Val Loss  | 0.5497 |

> Train/Val loss gap resolved — model is no longer overfitting.

---

## Bugs

| Bug | Impact |
|-----|--------|
| Double normalization (`max_pixel_value` not set) | Inputs scaled to ~0.004 instead of [0,1] — root cause of DSC 0.22 |
| Dice loss computed globally across batch | Mixed gradients across samples, skewed by large tumours |
| No `pos_weight` in BCE | Model predicted all-foreground (Recall 0.94, Precision 0.12) |
| No dropout in bridge | 3× train/val loss gap |
| Sigmoid missing at inference | Thresholding raw logits — wrong predictions |

---

## Current Limitations

- DSC ~0.61 — below research-grade benchmark (~0.75+) for BUSI
- No preprocessing — raw ultrasound has poor contrast; CLAHE not yet applied
- Training from scratch — no pretrained backbone
- Early stopping patience needs tuning; triggers too early
- Normal class (133 empty-mask images) still creates class imbalance pressure

---

## Roadmap

- [ ] **CLAHE preprocessing** — adaptive histogram equalisation to boost tumour contrast in ultrasound before feeding to the model
- [ ] **Explore other loss functions** — BCE + Dice Loss treats FP and FN equally and our precision lags behind recall, meaning model still leans towards over predicting. 
- [ ] **Test-time augmentation (TTA)** — average predictions over flipped/rotated inputs at inference
- [ ] **Exclude normal class** — train only on benign + malignant, evaluate separately on normal
- [ ] **Post-processing** — connected components to remove spurious small predictions

---

## Visualisations

- Tumor overlay on original ultrasound (solid colour mask blend)  
- Training curves: Loss · DSC · IoU · Precision · Recall  
- Pixel-level confusion matrix (normalised)  
- R² score bar

---

## Stack

`PyTorch` · `Albumentations` · `OpenCV` · `scikit-learn` · `Matplotlib` · `Seaborn`
