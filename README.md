# HemoFusion-Net

**A hybrid CNN–Transformer network for intracranial hemorrhage segmentation and detection**

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/data-PhysioNet%20CT--ICH-orange.svg)](https://physionet.org/content/ct-ich/)

A complete, reproducible study on the PhysioNet **CT-ICH** cohort (75 patients,
2,814 annotated axial slices, 318 with hemorrhage). One command produces every
number, table and figure the paper needs: the model comparison, the statistical
tests, the ablation grid, the explainability analysis, and a single Excel
workbook holding all of it.

> **Author contribution.** **Risha** and **Bitto** contributed equally to this
> work and are joint first authors.

---

## Table of contents

1. [What the model is](#1-what-the-model-is)
2. [Results](#2-results)
3. [Two findings that shaped the pipeline](#3-two-findings-that-shaped-the-pipeline)
4. [Quick start](#4-quick-start)
5. [Repository layout](#5-repository-layout)
6. [Experimental protocol](#6-experimental-protocol)
7. [Metrics and statistics](#7-metrics-and-statistics)
8. [Class-wise analysis](#8-class-wise-analysis)
9. [Ablation grid](#9-ablation-grid)
10. [Figures](#10-figures)
11. [Citation](#11-citation)
12. [License](#12-license)

---

## 1. What the model is

`HemoFusion-Net` runs **two ImageNet-pretrained encoders in parallel** over the
same input and fuses their pyramids scale by scale:

| Stream | Backbone | What it contributes |
|---|---|---|
| CNN | ResNet-34 | sharp local texture of fresh blood |
| Transformer | PVTv2-B1 | global context to separate blood from calcification, falx and partial-volume artefact |

**Fusion is bidirectional.** At the three high-resolution stages a
`GatedCrossFusion` block lets each stream compute the channel *and* spatial gates
applied to the other. At the bottleneck (stride 32, only 64 tokens) full
multi-head cross-attention is affordable, so `CrossAttentionFusion` gives the two
streams a genuine global exchange.

The decoder is an attention-gated U-Net with an `MRHDC` (multi-rate dilated)
bottleneck, `MSPool` + `CoordAttention` refined skips, and deep supervision. A
second head predicts slice-level hemorrhage and the five ICH subtypes plus skull
fracture, so **detection and delineation are learned jointly**.

Input is **2.5D and multi-window**: the brain (40/80), subdural (80/200) and bone
(600/2800) windows of the current slice plus the brain window of the slices above
and below — 5 channels. `timm` adapts each backbone's first convolution
automatically.

```
Loss = 0.4·Dice + 0.4·Tversky(β=0.7) + 0.2·FocalBCE + 0.3·BoundaryDice
     + 0.4·BCE(slice) + 0.2·BCE(subtypes)
```

The boundary term is a differentiable morphological-gradient Dice, which attacks
HD95 directly — region losses are blind to it.

> **On model size.** HemoFusion-Net is **43.8 M** parameters at 384 px, with only
> **9.0 M** of them randomly initialised — the rest is pretrained. That split
> matters more than the total here: the cohort offers ~200 hemorrhage-positive
> training slices, and an earlier 66.7 M version (ResNet-50 + PVTv2-B2, wider
> fusion) carried 17.1 M random parameters and *lost to a plain U-Net*. Halving
> the random capacity raised best validation Dice from 0.427 to 0.500. Sizing was
> chosen on the inner validation split (`outputs/logs/ab_tuning_results.csv`),
> never on test.

---

## 2. Results

Test fold: 7 patients, 597 slices, 72 hemorrhage-positive. Per-slice metrics are
computed on lesion-bearing slices only; brackets are 95 % bootstrap CIs.
Surface metrics are in millimetres, from each patient's NIfTI spacing.

| Model | Params (M) | Dice | IoU | HD95 ↓ | ASSD ↓ | NSD(2 mm) | Dice (patient) | AUROC |
|---|---|---|---|---|---|---|---|---|
| **HemoFusion-Net (proposed)** | 43.8 | 0.289 | 0.229 | **35.94** | **12.59** | **0.542** | **0.360** | **0.890** |
| U-Net++ | 9.4 | 0.264 | 0.218 | 40.85 | 17.21 | 0.522 | 0.257 | 0.826 |
| Attention U-Net | 8.1 | 0.290 | 0.231 | 44.76 | 18.62 | 0.483 | 0.270 | 0.820 |
| U-Net | 8.0 | **0.295** | **0.236** | 46.01 | 21.34 | 0.462 | 0.267 | 0.802 |

**Read this honestly.** On per-slice region overlap the four models are
indistinguishable — U-Net's Dice of 0.295 is nominally the highest, and every
paired Wilcoxon test against the proposed model is non-significant after Holm
correction (`T13_paired_tests_per_patient.csv`, all *p*<sub>holm</sub> = 1.0).
With **7 test patients** the study is not powered to separate models on Dice,
and the repository reports that rather than hiding it.

Where the proposed model does separate is **boundary quality and detection**:
HD95 falls 22 % against U-Net (46.0 → 35.9 mm), ASSD falls 41 % (21.3 → 12.6 mm),
NSD(2 mm) rises from 0.462 to 0.542, and detection AUROC rises from 0.802 to
0.890 at 98.9 % specificity. This is the behaviour the boundary loss and the
joint classification head were added to produce. Confirming it would need a
larger cohort or full 5-fold cross-validation (`--preset full`, one flag away).

Every number above regenerates from `outputs/predictions/*.csv` without
retraining — see [INSTRUCTIONS.md](INSTRUCTIONS.md).

**Cohort.** 2,814 slices / 75 patients; 318 hemorrhagic slices (11.3 %) in 36
patients; median lesion 620 px; subtype slice counts — Epidural 173,
Intraparenchymal 73, Subdural 56, Intraventricular 24, Subarachnoid 18, plus
Fracture 196.

---

## 3. Two findings that shaped the pipeline

**Skull stripping is off by default, and that is a measured decision.**
This cohort contains epidural and subdural hemorrhage, which lies against the
inner skull table — exactly what an intracranial mask erases. Auditing all 318
annotated slices (`T00_skull_strip_cost.csv`):

- the skull ring is **open on 23.3 %** of them, so morphological stripping simply
  fails there;
- forcing the strip anyway retains only **98.75 %** of lesion pixels, and costs
  **> 5 % of the lesion on 36 slices** and **> 25 % on 6**.

A slice that loses a quarter of its lesion can never exceed ≈ 0.86 Dice, so
stripping imposes a permanent ceiling. Bone information is supplied through the
explicit bone-window channel instead. Ablation `A13` measures the alternative.

**A silent fp16 overflow was corrupting training.** A 256×256 sigmoid map sums to
65,536, just above fp16's maximum of 65,504, so Dice/Tversky produced `inf` then
`NaN`. Because `clip_grad_norm_` ran *after* `scaler.unscale_()`, it rescaled
every gradient to `NaN` where `GradScaler` could no longer see it, and the
poisoned step landed. The forward pass now stays in mixed precision while the
loss is computed in fp32 (`MultiTaskLoss.to_fp32`), and steps with non-finite
gradients are skipped and counted.

---

## 4. Quick start

```bash
git clone https://github.com/kowshir-bitto/hemofusion-net.git
cd hemofusion-net
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Download the [PhysioNet CT-ICH dataset](https://physionet.org/content/ct-ich/)
and place it **beside** the repository, then reproduce the reported study
(~2.5 h on one modern GPU):

```bash
python -u run_pipeline.py --preset paper --workers 6
```

Full setup, dataset layout, every command-line flag and troubleshooting live in
**[INSTRUCTIONS.md](INSTRUCTIONS.md)**.

---

## 5. Repository layout

```
.
├── run_pipeline.py              # the driver — every stage
├── ab_tune.py                   # validation-only A/B harness for the recipe
├── make_docx.py                 # renders the results markdown into .docx
├── requirements.txt             # dependency ranges
├── requirements-lock.txt        # exact versions of the reported run
├── src/
│   ├── config.py                # one dataclass drives every experiment
│   ├── preprocess.py            # NIfTI -> cached npz, windowing, CLAHE, strip
│   ├── strip_analysis.py        # the skull-strip audit above
│   ├── dataset.py               # 2.5D assembly, augmentation, patient-wise folds
│   ├── models/
│   │   ├── blocks.py            # MSPool, MRHDC, CoordAttention, fusion blocks
│   │   ├── hemofusion.py        # the proposed model (fully ablatable)
│   │   ├── baselines.py         # U-Net, Attention U-Net, U-Net++, DeepLabV3+, …
│   │   └── hemoclr.py           # prior single-stream model (not in this study)
│   ├── losses.py  metrics.py  engine.py  ssl_pretrain.py
│   ├── stats.py                 # Wilcoxon, bootstrap, DeLong, McNemar, Friedman…
│   ├── xai.py                   # Grad-CAM / ++ / LayerCAM / Eigen-CAM + faithfulness
│   ├── viz.py  viz_class.py     # every figure
│   └── report.py                # CSV + Excel export
├── notebooks/                   # narrative reproduction of the results
├── tests/                       # unit tests for the metrics and statistics
└── outputs/
    ├── ICH_Hybrid_Results_paper.xlsx   # every table, one sheet each
    ├── RESULTS_paper.md                # paste-ready results section
    ├── PAPER_Methods_Results_Discussion.md/.docx
    ├── tables/       *.csv             # the same tables, flat
    ├── figures/      *.png + *.pdf     # 300 dpi and vector
    ├── predictions/  *.csv             # per-slice metrics — statistics re-runnable
    └── logs/         *.csv             # per-epoch history
```

`outputs/cache/` and `outputs/models/` are generated at runtime and are
deliberately not versioned.

---

## 6. Experimental protocol

The reported study (`--preset paper`) uses **one patient-wise hold-out split and
30 epochs**, comparing the proposed model against four baselines: U-Net,
Attention U-Net, U-Net++ and PVTv2-UNet. 5-fold cross-validation and the ablation
grid remain implemented and are one flag away (`--preset full`,
`--stages ...,ablation`) but are not part of this run.

- **Patient-wise stratified split.** Patients are stratified by hemorrhage burden
  (none / low / high) and partitioned five ways; fold 0 is the test set, giving
  roughly 60 % train / 20 % validation / 20 % test *patients*. No patient's slices
  ever appear in two splits.
- **An inner validation split** (18 % of training patients) selects the
  checkpoint, the probability threshold and the classification operating point.
  The test fold is touched exactly once, for reporting.
- **SimCLR pretraining uses training-fold patients only**, so no test image
  contributes to the representation.
- **Class balance by sampler, not by file duplication.** A
  `WeightedRandomSampler` targets 50 % hemorrhagic slices per batch; augmentation
  is online. (The predecessor wrote 10 augmented copies of every positive slice
  to disk, which cannot vary across epochs.)
- **Identical treatment for every model** — same input channels, augmentation,
  loss, schedule, TTA and threshold search. Only the architecture differs.
- **Seeded.** `seed = 42` across NumPy, Python and PyTorch; the exact
  configuration of the reported run is frozen in `outputs/config_paper.json` and
  `outputs/tables/T25_run_configuration.csv`.

---

## 7. Metrics and statistics

**Segmentation** — Dice, IoU, precision, recall, specificity, volume similarity,
and surface metrics in millimetres from the per-patient NIfTI spacing: HD95
(symmetric 95th percentile), ASSD, NSD(2 mm). Reported per slice (lesion-bearing
slices only), per patient (volumetric), and dataset-aggregated.

**Classification** — accuracy, sensitivity, specificity, precision, NPV, F1,
balanced accuracy, MCC, Cohen's κ, AUROC, AUPRC, plus per-subtype AUROC and a
calibration/ECE analysis.

**Tests** — the design is paired (every model sees identical slices) and Dice is
bounded and skewed, so:

| Question | Test |
|---|---|
| Are the distributions normal? | Shapiro–Wilk (screen, reported not assumed) |
| Proposed vs each baseline | Wilcoxon signed-rank, Holm-corrected per metric family |
| How big is the difference? | rank-biserial, Cliff's δ, paired Cohen's d, paired-bootstrap CI |
| All models at once | Friedman + Nemenyi critical-difference diagram |
| Detection AUC | DeLong test for correlated ROC curves |
| Detection decisions | McNemar (exact when discordant pairs < 25) |
| Volume agreement | ICC(2,1), Pearson r, Bland–Altman limits |

**Explainability is quantified, not just pictured** — Grad-CAM, Grad-CAM++ and
LayerCAM at three network depths, scored by pointing-game hit rate, share of CAM
energy inside the lesion, CAM–lesion IoU/Dice, saliency ratio, and a deletion
curve (erase the most salient pixels and watch Dice collapse; a flat curve means
the heatmap is decorative).

---

## 8. Class-wise analysis

Everything that can be broken down by class is, using one vocabulary throughout —
the five ICH subtypes, skull fracture, and hemorrhage-free:

- `F01a` / `F01b` — example slices per class, before and after preprocessing
- `F02_class_distribution` — slice and patient counts, lesion area per subtype, and
  the label co-occurrence matrix that justifies treating classification as
  multi-label rather than multi-class
- `T04_class_distribution`, `T07b_classwise_segmentation` — the numbers behind them
- `F18_classwise_dice` / `F18_classwise_iou` — per-subtype performance, every model
- `F24_gradcam_by_class`, `F25_qualitative_by_class` — explanations and predictions,
  one representative test slice per class

The rare subtypes carry very few test slices (Subdural 8, Subarachnoid 9,
Intraventricular 10), so their per-class figures are noisy by construction; the
slice count is printed on every axis so this cannot be read past.

---

## 9. Ablation grid

*Implemented and runnable; not part of the reported study.*

`A0` full model, then one element removed per row: transformer stream, CNN
stream, cross-modal fusion (→ plain concat), MRHDC bottleneck, coordinate
attention, attention gates, deep supervision, classification branch, 2.5D
context, boundary loss, SimCLR pretraining, the composite loss (→ Dice only),
CLAHE; plus skull stripping switched *on*. Each variant is paired against `A0` on
identical test slices with a Wilcoxon test (`T20_ablation_significance`).

```bash
python -u run_pipeline.py --preset paper --stages ablation --ablation-epochs 15
```

---

## 10. Figures

`F01` dataset composition · `F01a`/`F01b` class examples raw/preprocessed ·
`F02_class_distribution` class distribution · `F02_cv_folds` split composition ·
`F03` preprocessing chain · `F03a`–`F03g` one figure per augmentation ·
`F03z` full augmentation pipeline · `F04` augmentation overview · `F05` model
comparison with CIs · `F06` per-slice Dice distribution · `F07` paired
per-patient Dice · `F08`/`F09` training dynamics · `F10` ROC + PR ·
`F11` confusion matrix · `F12` calibration · `F13` per-subtype AUROC ·
`F14` per-fold variability (CV runs only) · `F15` critical-difference diagram ·
`F16` Bland–Altman · `F17` significance · `F18` class-wise performance ·
`F19` threshold sensitivity · `F20`/`F21` qualitative panels · `F22`/`F23` CAM
figures · `F24` class-wise Grad-CAM/++ · `F25` class-wise predictions.

Colour is assigned from a CVD-validated categorical order (worst adjacent
ΔE 9.1, normal-vision ΔE 22.9 on a light surface); two slots fall below 3:1
contrast, so every chart using them also carries direct value labels and has a
matching CSV. The proposed model keeps one fixed accent colour in every figure.

---

## 11. Citation

If you use this code, please cite both the dataset and this repository.

**Dataset**

```bibtex
@misc{hssayeni2020ctich,
  author       = {Hssayeni, Murtadha D. and Croock, M. S. and Salman, A. D. and
                  Al-khafaji, H. F. and Yahya, Z. A. and Ghoraani, Behnaz},
  title        = {Computed Tomography Images for Intracranial Hemorrhage
                  Detection and Segmentation},
  version      = {1.3.1},
  year         = {2020},
  publisher    = {PhysioNet},
  doi          = {10.13026/4nae-zg36},
  url          = {https://physionet.org/content/ct-ich/}
}
```

**This work**

```bibtex
@software{hemofusion_net,
  author  = {Risha and Bitto, Abu Kowshir},
  title   = {HemoFusion-Net: A Hybrid CNN--Transformer Network for
             Intracranial Hemorrhage Segmentation and Detection},
  year    = {2026},
  url     = {https://github.com/kowshir-bitto/hemofusion-net},
  note    = {Risha and Bitto contributed equally}
}
```

---

## 12. License

Code is released under the [MIT License](LICENSE).

The PhysioNet CT-ICH dataset is **not** redistributed here and carries its own
terms (ODC Attribution License) — download it from
[physionet.org/content/ct-ich](https://physionet.org/content/ct-ich/).
Pretrained encoder weights are fetched at runtime by `timm` and remain subject to
their upstream licences.

> **Research use only.** This software is not a medical device and must not be
> used for clinical diagnosis or treatment decisions.
