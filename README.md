# Image Processing / Vision Course Project — Robustness Study

**Objective:** evaluate how robust low-level and high-level vision algorithms are to image
distortions, then recover lost performance two ways — classical **image enhancement** and
**model fine-tuning**. Run end-to-end on **two datasets** — indoor scenes
([ADE20K](http://sceneparsing.csail.mit.edu/)) and outdoor driving
([KITTI](https://www.cvlibs.net/datasets/kitti/)) — so the findings are validated across domains.

> **Deliverables:** `Project_Report.docx` / `Project_Report.pdf` (report) and two executed
> notebooks: `robustness_project.ipynb` (ADE20K) and `robustness_kitti.ipynb` (KITTI).

---

## 1. Decisions and choices (with links)

| Component | Choice | Why | Reference |
|---|---|---|---|
| **Dataset** | ADE20K (`nateraw/ade20k-tiny`) | Scene-parsing data with **dense GT masks** → enables a real metric (mIoU) and DL fine-tuning with true labels | [HF dataset](https://huggingface.co/datasets/nateraw/ade20k-tiny) · [ADE20K](http://sceneparsing.csail.mit.edu/) |
| **Task 1 (low-level)** | ORB keypoint / corner detection | Classical local-feature detector; tests robustness of hand-crafted features | [OpenCV ORB](https://docs.opencv.org/4.x/d1/d89/tutorial_py_orb.html) |
| **Task 2 (high-level)** | Object detection — YOLOv8n | Fast, widely-used one-stage detector | [Ultralytics](https://docs.ultralytics.com/) |
| **Task 3 (high-level)** | Semantic segmentation — SegFormer-B0 | Transformer segmenter pretrained on ADE20K; has GT here | [Model card](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512) · [paper](https://arxiv.org/abs/2105.15203) |
| **Distortion 1** | Gaussian noise | Sensor noise | [Albumentations `GaussNoise`](https://albumentations.ai/docs/) |
| **Distortion 2** | Severe JPEG | Compression artifacts | [Albumentations `ImageCompression`](https://albumentations.ai/docs/) |
| **Distortion 3** | Low-light | Under-exposure | [Albumentations `RandomBrightnessContrast`](https://albumentations.ai/docs/) |
| **Enhancement** | NLM + bilateral (noise), bilateral-Y (JPEG), gamma + CLAHE (low-light) | Classical restoration matched to each distortion | [OpenCV photo](https://docs.opencv.org/4.x/d1/d79/group__photo__denoise.html) · [CLAHE](https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html) |
| **Fine-tuning** | YOLOv8n (pseudo-labels) + SegFormer-B0 (GT masks) | Adapt the DL models to distorted inputs | — |

---

## 2. Data and annotations (EDA)

A fixed random sample of 4 images (seed = 7) drives the qualitative results. ADE20K masks use
151 raw ids (0 = unlabelled, 1–150 = classes).

**Images with their ground-truth segmentation annotations:**

![Sample images with GT label overlays](outputs/p1_labels.png)

---

## 3. Pipeline overview

| Part | Stage | Output |
|---|---|---|
| 1 | Measure on **clean** images | baseline per task |
| 2 | Apply **distortions**, measure degradation | per-class + per-SNR metrics |
| 3 | **Enhance** (restore) distorted images | Improvement #1 |
| 4 | **Fine-tune** DL models on distorted data | Improvement #2 |

### Coverage — every task gets the full chain

| Task | Clean | Degradation | Enhance (Imp #1) | Fine-tune (Imp #2) |
|---|---|---|---|---|
| ORB (keypoints, low-level) | ✓ | ✓ | ✓ | n/a (classical) |
| YOLO (detection, high-level) | ✓ | ✓ | ✓ | ✓ |
| SegFormer (segmentation, high-level) | ✓ | ✓ | ✓ | ✓ |

---

## 4. Part 1 — Clean baseline

SegFormer prediction vs GT, and the **per-class IoU** (mean IoU = 0.539 over 22 present classes;
ORB ~800 keypoints/image).

Clean / Ground-Truth / Prediction:

![Clean vs GT vs prediction](outputs/p1_seg.png)

**Per-class metric (required: "per class"):**

![Per-class IoU on clean images](outputs/p1_iou.png)

---

## 5. Part 2 — Distortion and degradation

**Input → output, before/after distortion:**

![The three distortions](outputs/p2_distortions.png)

**Annotations under distortion** (clean vs distorted ORB keypoints and YOLO boxes):

![Features and detections under Gaussian noise](outputs/p2_features_GaussNoise.png)

**Degradation metrics** (segmentation mIoU and ORB matching ratio vs clean baseline):

![Degradation bar charts](outputs/p2_degradation.png)

**Per-SNR metric (required: "per SNR")** — YOLO detection recall across a low-light SNR sweep:

![YOLO recall vs SNR](outputs/p2_snr.png)

| Metric | Clean | GaussNoise | SevereJPEG | LowLight |
|---|---|---|---|---|
| Segmentation mIoU | 0.539 | 0.379 | 0.203 | 0.344 |
| ORB matching ratio | 1.000 | 0.573 | 0.298 | 0.087 |

---

## 6. Part 3 — Improvement #1: classical enhancement

**Side-by-side before/after restoration** (clean / distorted / restored):

![Low-light restoration triplets](outputs/p3_restore_LowLight.png)

**Performance comparison — distorted vs enhanced:**

![ORB & YOLO: distorted vs enhanced](outputs/p3_compare.png)

![Segmentation mIoU: distorted vs enhanced](outputs/p3_seg_compare.png)

Enhancement helps **object detection under low-light** (recall 0.30 → 0.62) but **not** the
texture/detail-dependent tasks — denoising removes the structure ORB and SegFormer rely on.

---

## 7. Part 4 — Improvement #2: fine-tuning

YOLOv8n is fine-tuned on distorted images with clean pseudo-labels; SegFormer-B0 is fine-tuned on
distorted images with **real GT masks** (trained on the held-out validation split → no leakage).

| Model | Metric (on noisy images) | Pretrained | Fine-tuned |
|---|---|---|---|
| YOLOv8n | detection recall | 0.60 | **0.85** |
| SegFormer-B0 | mIoU | 0.367 | **0.567** (≈ clean baseline 0.539) |

![YOLO pretrained vs fine-tuned](outputs/p4_finetune_quant.png)
![SegFormer pretrained vs fine-tuned](outputs/p4_segfinetune.png)

Fine-tuned weights: `work/runs/ft/weights/best.pt` (YOLO). All 18 figures are in `outputs/`.

---

## 8. Second dataset — KITTI (cross-domain validation)

To confirm the findings generalize beyond the course example's dataset, the **same study** is repeated
on **KITTI** (outdoor autonomous-driving), the dataset named in the assignment brief. Notebook:
`robustness_kitti.ipynb`. KITTI's semantic benchmark (200 images) provides **real GT for all three
tasks**, so here:

- **Segmentation** uses [SegFormer-B0 (Cityscapes, 19 classes)](https://huggingface.co/nvidia/segformer-b0-finetuned-cityscapes-1024-1024) vs real masks.
- **Detection** is evaluated against **real GT boxes** derived from the `instance` maps (no pseudo-labels).
- **Fine-tuning** of both deep models therefore uses **real labels**.

**Ground-truth annotations (boxes + segmentation):**

![KITTI GT boxes and masks](outputs/k1_gt.png)

**Degradation across all three tasks (vs real GT):**

![KITTI degradation](outputs/k2_degradation.png)

**Per-SNR (low-light sweep) and distorted-vs-enhanced:**

![KITTI YOLO recall vs SNR](outputs/k2_snr.png)
![KITTI distorted vs enhanced](outputs/k3_compare.png)

**Fine-tuning with real GT — pretrained vs fine-tuned on noisy images:**

![KITTI YOLO fine-tune](outputs/k4_yolo.png)
![KITTI SegFormer fine-tune](outputs/k4_seg.png)

### Cross-dataset summary

| Metric | ADE20K (indoor) | KITTI (driving) |
|---|---|---|
| Clean segmentation mIoU | 0.539 | 0.432 |
| Clean YOLO (detection) | — (pseudo-label ref) | **0.614 recall vs real GT** |
| Seg mIoU under noise | 0.379 | 0.350 |
| ORB matching ratio under low-light | 0.087 | 0.096 |
| **YOLO recall on noisy — fine-tuned** | 0.60 → 0.85 | 0.42 → **0.56** |
| **SegFormer mIoU on noisy — fine-tuned** | 0.37 → 0.57 | 0.35 → **0.46** |

Both datasets show the **same pattern**: distortions degrade every task; classical enhancement helps
detection (esp. noise/JPEG on KITTI) but not the texture/detail tasks; and **fine-tuning is the most
reliable recovery path** on both. This cross-domain agreement (indoor vs outdoor-driving) strengthens
the study's conclusions.

## 9. Conclusions

- All three distortions degrade all three tasks; low-light and noise are the most damaging.
- Classical enhancement is **not universally helpful** — it aids detection under low-light but can
  hurt keypoint matching and segmentation.
- **Fine-tuning is the most reliable robustness path**, nearly restoring clean-level accuracy on
  noisy inputs for both DL models.
- *Caveat:* the datasets/samples are small, so per-image numbers carry run-to-run variance; the
  qualitative conclusions are stable and **agree across both datasets**.

---

## 10. Repository structure

```
robustness_project.ipynb   # ADE20K notebook (Parts 1–4), executed
robustness_kitti.ipynb     # KITTI notebook (cross-domain), executed
Project_Report.docx/.pdf   # written report
build_notebook.py          # regenerates the ADE20K notebook
build_kitti_notebook.py    # regenerates the KITTI notebook
report.js                  # regenerates the report
outputs/                   # all figures (p1_*…p4_* ADE20K, k1_*…k4_* KITTI)
work/  work_kitti/         # fine-tune datasets + YOLO runs (…/weights/best.pt)
kitti/                     # KITTI data (from data_semantics.zip)
requirements.txt           # frozen environment
```

## 11. Setup & run (Windows, NVIDIA GPU)

The GPU is an NVIDIA Blackwell-architecture card — it needs the **CUDA 12.8** PyTorch build; plain
`pip install torch` will not run on it.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python.exe -m pip install ultralytics transformers datasets albumentations opencv-python matplotlib nbformat nbconvert ipykernel hf_xet

# run end-to-end
.venv\Scripts\python.exe build_notebook.py
.venv\Scripts\python.exe -m ipykernel install --user --name cvproj
.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=cvproj robustness_project.ipynb
```

Or open either notebook and **Run All**. The KITTI notebook expects `kitti/` (download
`data_semantics.zip` from the KITTI site and extract to `kitti/`).

## 12. Notes

- Helper functions not specified in the brief (ORB matching ratio via Lowe ratio test,
  `detection_recall` with IoU ≥ 0.5, `darken`, the SNR sweep) are implemented and documented in the notebooks.
- An Albumentations-2.x compatibility shim keeps the original `var_limit` / `quality_lower` parameters working.
- KITTI GT is read with PIL (not cv2) because importing ultralytics monkeypatches `cv2.imread` and
  breaks 16-bit `IMREAD_UNCHANGED` reads of the instance/semantic maps.
