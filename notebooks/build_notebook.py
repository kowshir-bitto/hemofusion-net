#!/usr/bin/env python
"""Generate the narrative report notebook.

The notebook reads the artefacts the pipeline wrote rather than recomputing
them, so it opens in seconds and always reflects the most recent run.  Regenerate
with:  python notebooks/build_notebook.py
"""
import json
import os

NB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ICH_Hybrid_Report.ipynb")

cells = []


def _lines(text: str):
    """nbformat expects every source line to keep its trailing newline."""
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": _lines(text.strip())})


def code(text):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": _lines(text.strip("\n"))})


md("""
# HemoFusion-Net — results report

Hybrid CNN–Transformer segmentation **and** slice-level detection of intracranial
hemorrhage on the PhysioNet CT-ICH cohort.

This notebook is a **reader** for the artefacts produced by `run_pipeline.py`.
It recomputes nothing heavy, so it opens instantly and always shows the latest
run. To regenerate the underlying results:

```bash
./.venv/bin/python -u run_pipeline.py --preset quick  --workers 6 \\
    --stages data,train,results,stats,qualitative,xai
./.venv/bin/python -u run_pipeline.py --preset quick  --workers 6 \\
    --stages ablation --ablation-epochs 15
```

Sections
1. Setup
2. Dataset
3. Why the pipeline does not skull-strip
4. Preprocessing and augmentation
5. Cross-validation protocol
6. Training dynamics
7. Segmentation results
8. Detection (classification) results
9. Statistical analysis
10. Ablation study
11. Explainability
12. Qualitative results
13. Everything in one workbook
""")

md("## 1. Setup")
code("""
import os, sys, glob, json
import numpy as np, pandas as pd
from IPython.display import Image, Markdown, display

sys.path.insert(0, os.path.abspath(".."))
pd.set_option("display.width", 200, "display.max_columns", 80)

ROOT    = os.path.abspath("..")
OUT     = os.path.join(ROOT, "outputs")
TABLES  = os.path.join(OUT, "tables")
FIGURES = os.path.join(OUT, "figures")

def table(name, n=None, cols=None):
    \"\"\"Load one result table by its T.. prefix.\"\"\"
    hits = sorted(glob.glob(os.path.join(TABLES, f"{name}*.csv")))
    if not hits:
        print(f"[missing] {name} — run the pipeline stage that produces it")
        return pd.DataFrame()
    df = pd.read_csv(hits[0])
    if cols:
        df = df[[c for c in cols if c in df.columns]]
    return df.head(n) if n else df

def fig(name, width=1050):
    hits = sorted(glob.glob(os.path.join(FIGURES, f"{name}*.png")))
    if not hits:
        print(f"[missing figure] {name}")
        return
    for h in hits:
        display(Markdown(f"**{os.path.basename(h)}**"))
        display(Image(filename=h, width=width))

print("tables :", len(glob.glob(os.path.join(TABLES, '*.csv'))))
print("figures:", len(glob.glob(os.path.join(FIGURES, '*.png'))))
""")

md("""
## 2. Dataset

75 patients with matching NIfTI masks (subjects 59–65 have no raw data in this
release), ~30 slices each at 5 mm thickness, annotated by two radiologists with
consensus.
""")
code("""
display(table("T02_dataset_summary"))
display(table("T01_patient_inventory", 10))
fig("F01_dataset_overview")
""")

md("""
## 3. Why the pipeline does not skull-strip

This is a design decision backed by measurement, not a default.

The cohort contains **epidural and subdural** hemorrhage, which lies against the
inner skull table — precisely what an intracranial mask removes. Auditing all
annotated slices shows morphological stripping both *fails often* and *destroys
lesion when it works*, so a slice that loses part of its lesion carries a
permanent Dice ceiling. Bone information is instead supplied through an explicit
bone-window input channel, and stripping is evaluated as ablation `A13`.
""")
code("""
display(table("T00_skull_strip_cost"))

per = table("T00b_skull_strip_per_slice")
if len(per):
    worst = per.nsmallest(8, "lesion_retained_if_forced")[
        ["patient", "slice", "lesion_px", "head_fraction_kept",
         "strip_leaked", "lesion_retained_if_forced"]]
    display(Markdown("**Slices that would suffer most if stripping were forced**"))
    display(worst)
""")

md("## 4. Preprocessing and augmentation")
code("""
fig("F03_preprocessing")
fig("F04_augmentation")
""")

md("""
## 5. Cross-validation protocol

Patient-wise stratified folds — patients are grouped by hemorrhage burden
(none / low / high) so each fold carries comparable positive signal. With only 75
patients an unlucky split would otherwise dominate the result. An inner
validation split taken from the training patients selects the checkpoint and the
operating threshold; the test fold is used once.
""")
code("""
display(table("T03_cv_fold_composition"))
fig("F02_cv_folds")
""")

md("## 6. Training dynamics")
code("""
fig("F08_training_curves_proposed")
fig("F09_training_curves_all")

hist = pd.read_csv(os.path.join(OUT, "logs", "quick_history.csv")) \\
       if os.path.exists(os.path.join(OUT, "logs", "quick_history.csv")) else pd.DataFrame()
if len(hist):
    summ = (hist.groupby("tag")
                .agg(epochs=("epoch", "max"),
                     best_val_dice=("val_dice", "max"),
                     best_val_auroc=("val_auroc", "max"),
                     final_train_loss=("train_total", "last"),
                     skipped_steps=("skipped_steps", "sum"))
                .sort_values("best_val_dice", ascending=False))
    display(summ.round(4))
""")

md("""
## 7. Segmentation results

`Dice`, `IoU`, `Precision`, `Recall`, `NSD` are per-slice means over
lesion-bearing test slices; `HD95` and `ASSD` are in **millimetres**, derived
from each patient's own in-plane spacing. `Dice_patient` is volumetric — tp/fp/fn
pooled across all of a patient's slices, which is the number a clinician reads.
""")
code("""
display(Markdown("### Main comparison (metric with 95% bootstrap CI)"))
display(table("T07_main_comparison_table"))

display(Markdown("### Full metric set"))
display(table("T05_results_mean_over_folds",
              cols=["display", "Dice", "IoU", "Precision", "Recall", "HD95", "ASSD",
                    "NSD", "Dice_patient", "Dice_agg", "params_M", "train_minutes"]).round(4))

fig("F05_model_comparison")
fig("F06_dice_distribution")
fig("F07_paired_patient_dice")
fig("F19_threshold_sensitivity")
""")

md("""
## 8. Detection results

The second head predicts whether a slice contains hemorrhage, plus the five ICH
subtypes and skull fracture as a multi-label problem. Thresholds come from the
inner validation split — never from test.
""")
code("""
display(table("T08_classification_slice_level",
              cols=["display", "AUROC", "AUPRC", "Sensitivity", "Specificity",
                    "Precision", "NPV", "F1", "BalancedAcc", "MCC", "Kappa",
                    "threshold", "TP", "FP", "FN", "TN"]).round(4))
fig("F10_roc_pr")
fig("F11_confusion_matrix")
fig("F12_calibration")

display(Markdown("### Per-subtype performance (proposed model)"))
sub = table("T09_classification_per_subtype")
if len(sub):
    display(sub[sub.tag == "hemofusion"][
        ["class", "n_pos", "AUROC", "AUPRC", "Sensitivity", "Specificity", "F1"]].round(4))
fig("F13_subtype_auc")
""")

md("""
## 9. Statistical analysis

The design is **paired** — every model is evaluated on identical test slices — so
paired tests are both valid and much more powerful than independent ones. Dice is
bounded and left-skewed, so the primary test is non-parametric; normality is
screened and reported rather than assumed.
""")
code("""
display(Markdown("### Normality screen (Shapiro-Wilk) — motivates the non-parametric choice"))
display(table("T11_normality_tests",
              cols=["display", "variable", "n", "mean", "median", "skew",
                    "shapiro_W", "shapiro_p", "normal_at_0.05"]).round(4))
""")

code("""
display(Markdown("### Proposed vs each baseline — Wilcoxon signed-rank, Holm-corrected"))
t = table("T12_paired_tests_per_slice")
if len(t):
    for met in ["Dice", "IoU", "HD95"]:
        d = t[t.metric == met]
        if not len(d):
            continue
        display(Markdown(f"**{met}**  (Δ = proposed − baseline; "
                         f"{'higher' if met not in ('HD95','ASSD') else 'lower'} is better)"))
        display(d[["display_b", "mean_a", "mean_b", "mean_diff", "ci_low", "ci_high",
                   "p_wilcoxon", "p_holm", "signif_label", "rank_biserial",
                   "n_wins", "n_losses"]].round(5).reset_index(drop=True))
fig("F17_significance")
""")

code("""
display(Markdown("### Patient-level test on volumetric Dice"))
display(table("T13_paired_tests_per_patient",
              cols=["display_b", "n_pairs", "mean_a", "mean_b", "mean_diff",
                    "p_wilcoxon", "p_holm", "signif_label", "rank_biserial"]).round(5))

display(Markdown("### Omnibus across all models — Friedman + Nemenyi"))
display(table("T14_friedman_ranks"))
display(table("T15_nemenyi_posthoc",
              cols=["display_a", "display_b", "rank_diff",
                    "critical_difference", "significant"]).round(4))
fig("F15_critical_difference")
""")

code("""
display(Markdown("### Detection AUC — DeLong test for correlated ROC curves"))
display(table("T16_delong_auc_tests",
              cols=["display_b", "auc_a", "auc_b", "z", "p_delong",
                    "p_holm", "signif_label", "n"]).round(5))

display(Markdown("### Detection decisions — McNemar"))
display(table("T17_mcnemar_tests",
              cols=["display_b", "acc_a", "acc_b", "a_correct_b_wrong",
                    "a_wrong_b_correct", "statistic", "p_mcnemar", "p_holm",
                    "signif_label", "exact"]).round(5))
""")

code("""
display(Markdown("### Hemorrhage-volume agreement — ICC(2,1) and Bland–Altman"))
display(table("T18_volume_agreement",
              cols=["display", "ICC", "pearson_r", "pearson_p", "bias",
                    "sd_diff", "loa_low", "loa_high", "n"]).round(4))
fig("F16_bland_altman")
""")

md("""
## 10. Ablation study

One element removed per row, everything else held fixed, every variant trained
under the identical schedule and evaluated on the identical test slices. The
significance table pairs each variant against the full model with a Wilcoxon
test, so a difference that is within noise is visible as such.
""")
code("""
display(table("T19_ablation_study",
              cols=["tag", "label", "Dice", "IoU", "HD95", "Dice_patient",
                    "cls_AUROC", "delta_Dice", "delta_pct", "params_M"]).round(4))
fig("F18_ablation")

display(Markdown("### Each variant vs the full model (paired Wilcoxon)"))
display(table("T20_ablation_significance",
              cols=["metric", "variant_label", "mean_a", "mean_b", "mean_diff",
                    "ci_low", "ci_high", "p_wilcoxon", "p_holm",
                    "signif_label", "rank_biserial"]).round(5))
""")

md("""
## 11. Explainability

Grad-CAM, Grad-CAM++ and LayerCAM at three network depths. The scalar being
explained is a *segmentation* quantity — the mean logit inside the model's own
predicted region — so the maps are not derived from the ground truth.

The heatmaps are also **scored**, because a picture of a heatmap proves nothing:

- `pointing_hit` — does the CAM peak land inside the reference lesion?
- `energy_in_gt` — what share of CAM mass falls inside it?
- `cam_iou` / `cam_dice` — overlap of the thresholded CAM with the lesion
- `saliency_ratio` — mean CAM inside vs outside the lesion
- `deletion_auc` / `dice_drop` — erase the most salient pixels progressively;
  a faithful explanation makes Dice collapse, a decorative one does not.
""")
code("""
display(Markdown("### Averaged explainability metrics by method and depth"))
display(table("T23_xai_summary",
              cols=["model", "method", "layer", "pointing_hit", "energy_in_gt",
                    "cam_iou", "cam_dice", "saliency_ratio", "deletion_auc",
                    "dice_drop"]).round(4))

fig("F22_xai_gradcam_hemofusion")
fig("F22_xai_gradcam_pp_hemofusion")
fig("F22_xai_layercam_hemofusion")
fig("F23_xai_method_comparison_hemofusion")
""")

md("## 12. Qualitative results")
code("""
fig("F20_qualitative_proposed")
fig("F21_qualitative_all_models")
""")

md("""
## 13. Everything in one workbook

`outputs/ICH_Hybrid_Results_*.xlsx` holds every table as its own sheet, with a
`00_contents` index. `outputs/RESULTS_*.md` is a paste-ready results section, and
`outputs/predictions/*.csv` carries the per-slice metrics so any statistic in this
notebook can be recomputed or re-tested independently.
""")
code("""
display(table("T24_figure_manifest"))
for f in sorted(glob.glob(os.path.join(OUT, "*.xlsx"))) + sorted(glob.glob(os.path.join(OUT, "*.md"))):
    print(f"{os.path.getsize(f)/1024:9.1f} KB  {os.path.relpath(f, ROOT)}")
""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open(NB, "w") as fh:
    json.dump(nb, fh, indent=1)
print(f"wrote {NB}  ({len(cells)} cells)")
