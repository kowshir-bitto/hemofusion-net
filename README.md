# HemoFusion-Net

## What the model is

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

## Quick start

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

---

## Repository layout

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
    ├── tables/       *.csv             # the same tables, flat
    ├── figures/      *.png + *.pdf     # 300 dpi and vector
    ├── predictions/  *.csv             # per-slice metrics — statistics re-runnable
    └── logs/         *.csv             # per-epoch history
```

`outputs/cache/` and `outputs/models/` are generated at runtime and are
deliberately not versioned.

---

## License

Code is released under the [MIT License](LICENSE).

The PhysioNet CT-ICH dataset is **not** redistributed here and carries its own
terms (ODC Attribution License) — download it from
[physionet.org/content/ct-ich](https://physionet.org/content/ct-ich/).
Pretrained encoder weights are fetched at runtime by `timm` and remain subject to
their upstream licences.

> **Research use only.** This software is not a medical device and must not be
> used for clinical diagnosis or treatment decisions.
