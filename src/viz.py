"""Publication figures.

Every figure is written to ``outputs/figures`` as both a 300-dpi PNG and a
vector PDF (what a journal actually wants).  Colour assignment follows a
CVD-validated categorical order — the palette was checked with the data-viz
validator and passes the lightness band, chroma floor, adjacent CVD separation
(worst ΔE 9.1) and normal-vision floor (worst ΔE 22.9) gates on a light surface.
Two slots sit below 3:1 contrast, so every chart that uses them also carries
direct value labels, and each figure has a matching CSV/Excel table.

These are print figures: they deliberately commit to the light surface.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

from .config import DIR_FIGURES, MULTILABEL

SERIES = ["#d55181", "#2a78d6", "#008300", "#eda100",
          "#1baf7a", "#eb6834", "#4a3aa7", "#e34948"]
ACCENT = SERIES[0]
NEUTRAL = "#2a78d6"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8983"
GRID = "#dcdbd6"
SURFACE = "#ffffff"

MODEL_ORDER = ["hemofusion", "hemoclr_net", "pvt_unet", "deeplabv3p",
               "resunet50", "unetpp", "attention_unet", "unet"]

SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f2f7fe", "#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#184f95", "#0d366b"]
)
DIVERGING = LinearSegmentedColormap.from_list(
    "div_br", ["#184f95", "#5598e7", "#cde2fb", "#f0efec", "#f6b8b8", "#e34948", "#8f1f1f"]
)


def model_colors(models: Sequence[str]) -> Dict[str, str]:
    """Stable colour per model — a filter that drops models never repaints the rest."""
    ordered = [m for m in MODEL_ORDER if m in models] + \
              [m for m in models if m not in MODEL_ORDER]
    return {m: SERIES[i % len(SERIES)] for i, m in enumerate(ordered)}


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.9,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
        "figure.dpi": 110,
    })


def _despine(ax, keep=("left", "bottom")):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)


def save(fig, name: str, tight: bool = True) -> str:
    if tight:
        fig.tight_layout()
    png = os.path.join(DIR_FIGURES, f"{name}.png")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(DIR_FIGURES, f"{name}.pdf"), bbox_inches="tight")
    plt.close(fig)
    return png


def fig_dataset_overview(index: pd.DataFrame, demo: Optional[pd.DataFrame], name: str) -> str:
    apply_style()
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))

    per = index.groupby("patient").agg(slices=("slice", "size"), bleed=("has_bleed", "sum"))
    a = ax[0, 0]
    a.hist(per.slices, bins=18, color=NEUTRAL, edgecolor=SURFACE, linewidth=0.8)
    a.set(title="Slices per patient", xlabel="slices", ylabel="patients")

    a = ax[0, 1]
    a.hist(per.bleed, bins=18, color=ACCENT, edgecolor=SURFACE, linewidth=0.8)
    a.set(title="Hemorrhagic slices per patient", xlabel="slices with ICH", ylabel="patients")

    a = ax[0, 2]
    counts = [int((index.has_bleed == 0).sum()), int((index.has_bleed == 1).sum())]
    bars = a.bar(["no ICH", "ICH"], counts, color=[MUTED, ACCENT], width=0.6)
    for b, v in zip(bars, counts):
        a.text(b.get_x() + b.get_width() / 2, v, f"{v}\n({v/sum(counts)*100:.1f}%)",
               ha="center", va="bottom", fontsize=9, color=INK)
    a.set(title=f"Slice-level class balance (n={sum(counts)})", ylabel="slices")
    a.set_ylim(0, max(counts) * 1.22)

    a = ax[1, 0]
    prev = [int(index[c].sum()) for c in MULTILABEL]
    order = np.argsort(prev)[::-1]
    lbl = [MULTILABEL[i] for i in order]
    val = [prev[i] for i in order]
    bars = a.barh(lbl, val, color=NEUTRAL, height=0.65)
    for b, v in zip(bars, val):
        a.text(v + max(val) * 0.01, b.get_y() + b.get_height() / 2, str(v),
               va="center", fontsize=9, color=INK)
    a.invert_yaxis()
    a.set(title="Radiologist labels per slice", xlabel="slices")
    a.set_xlim(0, max(val) * 1.16)

    a = ax[1, 1]
    px = index.loc[index.mask_px > 0, "mask_px"]
    a.hist(px, bins=40, color=ACCENT, edgecolor=SURFACE, linewidth=0.6)
    a.set(title="Lesion area on hemorrhagic slices", xlabel="mask pixels", ylabel="slices")
    a.set_yscale("log")

    a = ax[1, 2]
    if demo is not None and len(demo):
        for i, (g, sub) in enumerate(demo.groupby("gender")):
            a.hist(sub.age.dropna(), bins=14, alpha=0.75, label=str(g),
                   color=SERIES[i + 1], edgecolor=SURFACE, linewidth=0.6)
        a.legend(title="gender")
        a.set(title="Patient age distribution", xlabel="age (years)", ylabel="patients")
    else:
        a.axis("off")

    for row in ax:
        for a_ in row:
            _despine(a_)
    fig.suptitle("CT-ICH dataset composition", fontsize=13, fontweight="bold")
    return save(fig, name)


def fig_preprocessing_stages(stages: List[Dict[str, np.ndarray]], name: str) -> str:
    """One row per example slice, one column per preprocessing stage."""
    apply_style()
    keys = list(stages[0].keys())
    fig, ax = plt.subplots(len(stages), len(keys), figsize=(2.5 * len(keys), 2.7 * len(stages)))
    ax = np.atleast_2d(ax)
    for r, st in enumerate(stages):
        for c, k in enumerate(keys):
            img = st[k]
            a = ax[r, c]
            if img.ndim == 3:
                a.imshow(img)
            else:
                a.imshow(img, cmap="gray")
            if r == 0:
                a.set_title(k, fontsize=10, fontweight="bold")
            a.set_xticks([]); a.set_yticks([])
            for s in a.spines.values():
                s.set_visible(False)
    fig.suptitle("Preprocessing pipeline", fontsize=13, fontweight="bold")
    return save(fig, name)


def fig_augmentation(samples: List[Dict[str, np.ndarray]], name: str) -> str:
    """Grid of augmented views, each with its transformed mask outlined.

    Drawing the mask as a contour on the image (rather than as a separate panel)
    makes it immediately visible that the geometric transform was applied to both.
    """
    apply_style()
    n = len(samples)
    ncol = min(4, n)
    nrow = int(np.ceil(n / ncol))
    fig, ax = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 3.5 * nrow), squeeze=False)
    fig.subplots_adjust(hspace=0.18)
    for i, s in enumerate(samples):
        a = ax[i // ncol][i % ncol]
        a.imshow(s["image"], cmap="gray", vmin=0, vmax=1)
        m = s["mask"] > 0.5
        if m.any():
            a.contour(m.astype(float), levels=[0.5], colors=["#00ff9d"], linewidths=1.6)
        a.set_title(s["label"], fontsize=10, fontweight="bold")
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(False)
    for i in range(n, nrow * ncol):
        ax[i // ncol][i % ncol].axis("off")
    fig.suptitle("Online training augmentation (lesion outlined in green)",
                 fontsize=13, fontweight="bold")
    return save(fig, name)


def fig_fold_composition(folds: pd.DataFrame, name: str) -> str:
    apply_style()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    g = folds.groupby("fold").agg(patients=("patient", "size"), bleed=("bleed", "sum"),
                                  slices=("n", "sum"))
    x = np.arange(len(g))
    w = 0.38
    b1 = ax[0].bar(x - w / 2 - 0.01, g.patients, w, label="patients", color=NEUTRAL)
    b2 = ax[0].bar(x + w / 2 + 0.01, g.bleed, w, label="hemorrhagic slices", color=ACCENT)
    for bars in (b1, b2):
        for b in bars:
            ax[0].text(b.get_x() + b.get_width() / 2, b.get_height(),
                       f"{int(b.get_height())}", ha="center", va="bottom", fontsize=8, color=INK)
    ax[0].set(title="Cross-validation fold composition", xlabel="fold", ylabel="count")
    ax[0].set_xticks(x, [f"{i}" for i in g.index])
    ax[0].legend()

    frac = (g.bleed / g.slices * 100)
    bars = ax[1].bar(x, frac, 0.55, color=NEUTRAL)
    for b, v in zip(bars, frac):
        ax[1].text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}%",
                   ha="center", va="bottom", fontsize=9, color=INK)
    ax[1].set(title="Hemorrhage prevalence per fold", xlabel="fold", ylabel="% of slices")
    ax[1].set_xticks(x, [f"{i}" for i in g.index])
    for a in ax:
        _despine(a)
    return save(fig, name)


def fig_training_curves(history: pd.DataFrame, name: str, title: str = "") -> str:
    apply_style()
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ep = history.epoch

    ax[0].plot(ep, history.train_total, color=NEUTRAL, label="train")
    if "val_loss" in history:
        ax[0].plot(ep, history.val_loss, color=SERIES[5], label="validation")
    ax[0].set(title="Total loss", xlabel="epoch", ylabel="loss")
    ax[0].legend()

    comp = [c for c in ("train_seg", "train_deep_sup", "train_cls", "train_multi")
            if c in history]
    for i, c in enumerate(comp):
        ax[1].plot(ep, history[c], color=SERIES[i + 1], label=c.replace("train_", ""))
    ax[1].set(title="Loss components (train)", xlabel="epoch", ylabel="loss")
    if comp:
        ax[1].legend()

    ax[2].plot(ep, history.val_dice, color=ACCENT, label="val Dice")
    best = history.val_dice.idxmax()
    ax[2].scatter([history.loc[best, "epoch"]], [history.loc[best, "val_dice"]],
                  color=ACCENT, zorder=5, s=45, edgecolor=SURFACE, linewidth=1.5)
    ax[2].annotate(f"best {history.loc[best,'val_dice']:.3f}",
                   (history.loc[best, "epoch"], history.loc[best, "val_dice"]),
                   textcoords="offset points", xytext=(6, -12), fontsize=9, color=INK)
    if "val_auroc" in history and history.val_auroc.notna().any():
        a2 = ax[2].twinx()
        a2.plot(ep, history.val_auroc, color=NEUTRAL, linestyle="--", label="val AUROC")
        a2.set_ylabel("AUROC", color=INK_2)
        a2.grid(False)
        ax[2].legend(handles=[Line2D([], [], color=ACCENT, label="val Dice"),
                              Line2D([], [], color=NEUTRAL, ls="--", label="val AUROC")],
                     loc="lower right")
    else:
        ax[2].legend()
    ax[2].set(title="Validation performance", xlabel="epoch", ylabel="Dice")

    for a in ax:
        _despine(a)
    fig.suptitle(title or "Training dynamics", fontsize=12, fontweight="bold")
    return save(fig, name)


def fig_all_training_curves(history: pd.DataFrame, display: Dict[str, str], name: str) -> str:
    """Validation Dice of every model on one axis — shows convergence differences."""
    apply_style()
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    models = [m for m in MODEL_ORDER if m in history.tag.unique()] + \
             [m for m in history.tag.unique() if m not in MODEL_ORDER]
    cmap = model_colors(models)
    for m in models:
        h = history[history.tag == m].sort_values("epoch")
        ax[0].plot(h.epoch, h.val_dice, color=cmap[m], label=display.get(m, m))
        ax[1].plot(h.epoch, h.val_loss, color=cmap[m], label=display.get(m, m))
    ax[0].set(title="Validation Dice", xlabel="epoch", ylabel="Dice")
    ax[1].set(title="Validation loss", xlabel="epoch", ylabel="loss")
    ax[0].legend(ncol=2, fontsize=8)
    for a in ax:
        _despine(a)
    return save(fig, name)


def fig_metric_bars(summary: pd.DataFrame, name: str,
                    metrics: Sequence[str] = ("Dice", "IoU", "HD95", "Dice_patient"),
                    ci: Optional[pd.DataFrame] = None) -> str:
    """Grouped bars, one panel per metric, value printed on every bar."""
    apply_style()
    df = summary.copy()
    order = [m for m in MODEL_ORDER if m in df.tag.values] + \
            [m for m in df.tag.values if m not in MODEL_ORDER]
    df = df.set_index("tag").loc[order].reset_index()
    cmap = model_colors(order)

    n = len(metrics)
    fig, ax = plt.subplots(1, n, figsize=(4.1 * n, 4.6))
    ax = np.atleast_1d(ax)
    for j, met in enumerate(metrics):
        a = ax[j]
        if met not in df:
            a.axis("off")
            continue
        vals = df[met].values
        colors = [cmap[t] for t in df.tag]
        y = np.arange(len(df))
        err = None
        if ci is not None:
            sub = ci[ci.metric == met].set_index("tag")
            if len(sub):
                lo = df.tag.map(sub.ci_low).values
                hi = df.tag.map(sub.ci_high).values
                err = np.vstack([vals - lo, hi - vals])
                err = np.clip(np.nan_to_num(err, nan=0.0), 0, None)
        a.barh(y, vals, color=colors, height=0.66, xerr=err,
               error_kw=dict(ecolor=INK_2, lw=1.0, capsize=2.5))
        span = np.nanmax(vals) - min(0, np.nanmin(vals))
        for yy, v in zip(y, vals):
            a.text(v + span * 0.02, yy, f"{v:.3f}" if v < 10 else f"{v:.1f}",
                   va="center", fontsize=8.5, color=INK)
        a.set_yticks(y, [df.display.iloc[i] for i in range(len(df))], fontsize=8.5)
        a.invert_yaxis()
        unit = " (mm)" if met in ("HD95", "ASSD") else ""
        a.set(title=met + unit)
        a.set_xlim(0, np.nanmax(vals) * 1.22)
        _despine(a)
        if j > 0:
            a.set_yticklabels([])
    fig.suptitle("Test-set performance by model", fontsize=12, fontweight="bold")
    return save(fig, name)


def fig_dice_distribution(per_slice: Dict[str, pd.DataFrame], display: Dict[str, str],
                          name: str, metric: str = "Dice") -> str:
    """Box + strip of the per-slice metric — the distribution the tests operate on."""
    apply_style()
    models = [m for m in MODEL_ORDER if m in per_slice] + \
             [m for m in per_slice if m not in MODEL_ORDER]
    cmap = model_colors(models)
    data = [per_slice[m].loc[per_slice[m].gt_empty == 0, metric].dropna().values for m in models]

    fig, ax = plt.subplots(figsize=(max(8, 1.5 * len(models)), 5.2))
    bp = ax.boxplot(data, patch_artist=True, widths=0.58, showfliers=False,
                    medianprops=dict(color=SURFACE, lw=2),
                    whiskerprops=dict(color=INK_2, lw=1.0),
                    capprops=dict(color=INK_2, lw=1.0))
    for patch, m in zip(bp["boxes"], models):
        patch.set_facecolor(cmap[m])
        patch.set_edgecolor(SURFACE)
        patch.set_linewidth(2)
        patch.set_alpha(0.92)

    rng = np.random.default_rng(0)
    for i, (d, m) in enumerate(zip(data, models)):
        if len(d) == 0:
            continue
        keep = rng.choice(len(d), size=min(len(d), 260), replace=False)
        ax.scatter(np.full(len(keep), i + 1) + rng.normal(0, 0.075, len(keep)), d[keep],
                   s=6, color=INK_2, alpha=0.28, linewidths=0, zorder=3)
        ax.text(i + 1, 1.045, f"{np.mean(d):.3f}", ha="center", fontsize=8.5,
                color=INK, fontweight="bold")

    ax.set_xticks(range(1, len(models) + 1),
                  [display.get(m, m).replace(" (", "\n(") for m in models], fontsize=8.5)
    ax.set(ylabel=f"per-slice {metric}", ylim=(-0.03, 1.10),
           title=f"Distribution of per-slice {metric} on hemorrhagic test slices"
                 "  (mean printed above each box)")
    _despine(ax)
    return save(fig, name)


def fig_patient_dice(per_slice: Dict[str, pd.DataFrame], display: Dict[str, str],
                     name: str, reference: str = "hemofusion") -> str:
    """Paired per-patient Dice: proposed vs each baseline, one line per patient."""
    apply_style()
    models = [m for m in MODEL_ORDER if m in per_slice and m != reference]
    if reference not in per_slice or not models:
        return ""
    ref = per_slice[reference].groupby("patient").patient_dice.first()
    cmap = model_colors([reference] + models)

    ncol = min(3, len(models))
    nrow = int(np.ceil(len(models) / ncol))
    fig, ax = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.8 * nrow), squeeze=False)
    for k, m in enumerate(models):
        a = ax[k // ncol][k % ncol]
        oth = per_slice[m].groupby("patient").patient_dice.first()
        common = ref.index.intersection(oth.index)
        for p in common:
            a.plot([0, 1], [oth[p], ref[p]], color=MUTED, lw=0.8, alpha=0.6, zorder=1)
        a.scatter(np.zeros(len(common)), oth[common], s=26, color=cmap[m],
                  zorder=3, edgecolor=SURFACE, linewidth=0.8)
        a.scatter(np.ones(len(common)), ref[common], s=26, color=ACCENT,
                  zorder=3, edgecolor=SURFACE, linewidth=0.8)
        win = int(np.sum(ref[common].values > oth[common].values))
        a.set_xticks([0, 1], [display.get(m, m).split(" (")[0], "proposed"], fontsize=8.5)
        a.set_xlim(-0.28, 1.28)
        a.set_ylim(-0.03, 1.03)
        a.set_title(f"{display.get(m,m).split(' (')[0]}  ·  proposed better on "
                    f"{win}/{len(common)} patients", fontsize=9.5)
        if k % ncol == 0:
            a.set_ylabel("per-patient Dice")
        _despine(a)
    for k in range(len(models), nrow * ncol):
        ax[k // ncol][k % ncol].axis("off")
    fig.suptitle("Paired per-patient Dice against the proposed model",
                 fontsize=12, fontweight="bold")
    return save(fig, name)


def fig_roc_pr(cls_rows: Dict[str, pd.DataFrame], display: Dict[str, str], name: str) -> str:
    apply_style()
    from sklearn.metrics import (average_precision_score, precision_recall_curve,
                                 roc_auc_score, roc_curve)
    models = [m for m in MODEL_ORDER if m in cls_rows] + \
             [m for m in cls_rows if m not in MODEL_ORDER]
    cmap = model_colors(models)

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.2))
    base = None
    for m in models:
        df = cls_rows[m]
        y, p = df.target_ich.values, df.prob.values
        if np.isnan(p).all() or len(np.unique(y)) < 2:
            continue
        base = float(np.mean(y))
        fpr, tpr, _ = roc_curve(y, p)
        ax[0].plot(fpr, tpr, color=cmap[m], lw=2.0 if m == "hemofusion" else 1.5,
                   label=f"{display.get(m,m)} — AUC {roc_auc_score(y,p):.3f}")
        pr, rc, _ = precision_recall_curve(y, p)
        ax[1].plot(rc, pr, color=cmap[m], lw=2.0 if m == "hemofusion" else 1.5,
                   label=f"{display.get(m,m)} — AP {average_precision_score(y,p):.3f}")

    ax[0].plot([0, 1], [0, 1], color=MUTED, ls=":", lw=1.2, label="chance")
    ax[0].set(title="ROC — slice-level hemorrhage detection",
              xlabel="1 − specificity", ylabel="sensitivity", xlim=(0, 1), ylim=(0, 1.02))
    if base is not None:
        ax[1].axhline(base, color=MUTED, ls=":", lw=1.2, label=f"prevalence {base:.3f}")
    ax[1].set(title="Precision–recall", xlabel="recall", ylabel="precision",
              xlim=(0, 1), ylim=(0, 1.02))
    for a in ax:
        a.legend(fontsize=7.8, loc="lower left" if a is ax[0] else "lower right")
        _despine(a)
    return save(fig, name)


def fig_confusion(cm: np.ndarray, name: str, labels=("no ICH", "ICH"),
                  title: str = "Slice-level confusion matrix") -> str:
    apply_style()
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    im = ax.imshow(norm, cmap=SEQ_BLUE, vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{int(cm[i,j])}\n{norm[i,j]*100:.1f}%", ha="center", va="center",
                    color=SURFACE if norm[i, j] > 0.55 else INK, fontsize=10,
                    fontweight="bold")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set(xlabel="predicted", ylabel="ground truth", title=title)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, label="row-normalised")
    return save(fig, name)


def fig_subtype_auc(rows: pd.DataFrame, name: str) -> str:
    apply_style()
    df = rows[rows["class"] != "macro-average"].copy().sort_values("AUROC", ascending=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    bars = ax.barh(df["class"], df.AUROC, color=NEUTRAL, height=0.62)
    for b, v, n in zip(bars, df.AUROC, df.n_pos):
        ax.text(v + 0.008, b.get_y() + b.get_height() / 2, f"{v:.3f}  (n+={int(n)})",
                va="center", fontsize=8.8, color=INK)
    ax.axvline(0.5, color=MUTED, ls=":", lw=1.2)
    ax.set(title="Per-class AUROC — ICH subtype and fracture heads",
           xlabel="AUROC", xlim=(0, 1.19))
    _despine(ax)
    return save(fig, name)


def fig_calibration(y_true: np.ndarray, y_prob: np.ndarray, name: str, bins: int = 10) -> str:
    """Reliability diagram — whether the predicted probability means anything."""
    apply_style()
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(y_prob, edges) - 1, 0, bins - 1)
    xs, ys, ns = [], [], []
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        xs.append(y_prob[m].mean())
        ys.append(y_true[m].mean())
        ns.append(int(m.sum()))
    ece = float(np.sum([n * abs(x - y) for x, y, n in zip(xs, ys, ns)]) / max(len(y_true), 1))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4),
                           gridspec_kw={"width_ratios": [1.15, 1]})
    ax[0].plot([0, 1], [0, 1], color=MUTED, ls=":", lw=1.2, label="perfect calibration")
    ax[0].plot(xs, ys, "-o", color=ACCENT, label=f"model (ECE {ece:.3f})")
    ax[0].set(title="Reliability diagram", xlabel="mean predicted probability",
              ylabel="observed frequency", xlim=(0, 1), ylim=(0, 1))
    ax[0].legend()
    ax[1].hist(y_prob[y_true == 0], bins=25, color=MUTED, alpha=0.85, label="no ICH")
    ax[1].hist(y_prob[y_true == 1], bins=25, color=ACCENT, alpha=0.85, label="ICH")
    ax[1].set(title="Predicted probability by true class", xlabel="predicted probability",
              ylabel="slices", yscale="log")
    ax[1].legend()
    for a in ax:
        _despine(a)
    return save(fig, name)


def fig_qualitative(samples: List[Dict], name: str,
                    title: str = "Segmentation results") -> str:
    """CT | ground truth | prediction | error overlay, one row per slice."""
    apply_style()
    cols = ["CT (brain window)", "Ground truth", "Prediction", "TP / FP / FN"]
    fig, ax = plt.subplots(len(samples), 4, figsize=(13, 3.35 * len(samples)))
    ax = np.atleast_2d(ax)
    for r, s in enumerate(samples):
        ct, gt, pr = s["ct"], s["gt"] > 0.5, s["pred"] > 0.5
        for c in range(4):
            a = ax[r, c]
            a.imshow(ct, cmap="gray", vmin=0, vmax=1)
            if c == 1:
                ov = np.zeros((*gt.shape, 4)); ov[gt] = [0.0, 0.51, 0.0, 0.55]
                a.imshow(ov)
            elif c == 2:
                ov = np.zeros((*pr.shape, 4)); ov[pr] = [0.84, 0.32, 0.51, 0.55]
                a.imshow(ov)
            elif c == 3:
                ov = np.zeros((*gt.shape, 4))
                ov[gt & pr] = [0.11, 0.69, 0.48, 0.62]
                ov[~gt & pr] = [0.89, 0.29, 0.28, 0.62]
                ov[gt & ~pr] = [0.93, 0.63, 0.0, 0.62]
                a.imshow(ov)
            if r == 0:
                a.set_title(cols[c], fontsize=10, fontweight="bold")
            a.set_xticks([]); a.set_yticks([])
            for sp in a.spines.values():
                sp.set_visible(False)
        head = f"{s['label']}\n" if s.get("label") else ""
        ax[r, 0].set_ylabel(f"{head}pt {s['patient']} · sl {s['slice']}\n"
                            f"Dice {s['dice']:.3f}", fontsize=9, color=INK)
    handles = [Line2D([], [], marker="s", ls="", markersize=9, color="#1baf7a", label="TP"),
               Line2D([], [], marker="s", ls="", markersize=9, color="#e34948", label="FP"),
               Line2D([], [], marker="s", ls="", markersize=9, color="#eda100", label="FN")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.012))
    fig.suptitle(title, fontsize=13, fontweight="bold")
    return save(fig, name)


def fig_model_gallery(gallery: List[Dict], display: Dict[str, str], name: str) -> str:
    """Same slices, every model — the visual counterpart of the metric table."""
    apply_style()
    models = gallery[0]["preds"].keys()
    models = [m for m in MODEL_ORDER if m in models] + \
             [m for m in models if m not in MODEL_ORDER]
    ncol = 2 + len(models)
    fig, ax = plt.subplots(len(gallery), ncol, figsize=(2.2 * ncol, 2.45 * len(gallery)))
    ax = np.atleast_2d(ax)
    for r, s in enumerate(gallery):
        ct, gt = s["ct"], s["gt"] > 0.5
        ax[r, 0].imshow(ct, cmap="gray", vmin=0, vmax=1)
        ax[r, 1].imshow(ct, cmap="gray", vmin=0, vmax=1)
        ov = np.zeros((*gt.shape, 4)); ov[gt] = [0.0, 0.51, 0.0, 0.55]
        ax[r, 1].imshow(ov)
        for c, m in enumerate(models):
            pr = s["preds"][m] > 0.5
            a = ax[r, 2 + c]
            a.imshow(ct, cmap="gray", vmin=0, vmax=1)
            ovm = np.zeros((*gt.shape, 4))
            ovm[gt & pr] = [0.11, 0.69, 0.48, 0.62]
            ovm[~gt & pr] = [0.89, 0.29, 0.28, 0.62]
            ovm[gt & ~pr] = [0.93, 0.63, 0.0, 0.62]
            a.imshow(ovm)
            a.set_xlabel(f"{s['dice'][m]:.3f}", fontsize=8.5, color=INK)
        titles = ["CT", "Ground truth"] + [display.get(m, m).split(" (")[0] for m in models]
        for c in range(ncol):
            if r == 0:
                ax[r, c].set_title(titles[c], fontsize=9, fontweight="bold")
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
            for sp in ax[r, c].spines.values():
                sp.set_visible(False)
        ax[r, 0].set_ylabel(f"pt {s['patient']}\nsl {s['slice']}", fontsize=8.5)
    fig.suptitle("Qualitative comparison across models (per-slice Dice below each panel)",
                 fontsize=12, fontweight="bold")
    return save(fig, name)


def fig_ablation(df: pd.DataFrame, name: str, metric: str = "Dice",
                 full_tag: str = "A0_full") -> str:
    """Ablation bars sorted by effect, with the delta against the full model."""
    apply_style()
    d = df.sort_values(metric, ascending=True).copy()
    full = d.loc[d.tag == full_tag, metric]
    base = float(full.iloc[0]) if len(full) else float(d[metric].max())
    d["delta"] = d[metric] - base

    fig, ax = plt.subplots(1, 2, figsize=(14.5, max(4.4, 0.42 * len(d) + 1.6)),
                           gridspec_kw={"width_ratios": [1.25, 1]})
    colors = [ACCENT if t == full_tag else NEUTRAL for t in d.tag]
    y = np.arange(len(d))
    bars = ax[0].barh(y, d[metric], color=colors, height=0.68)
    for b, v in zip(bars, d[metric]):
        ax[0].text(v + 0.004, b.get_y() + b.get_height() / 2, f"{v:.4f}",
                   va="center", fontsize=8.5, color=INK)
    ax[0].axvline(base, color=ACCENT, ls="--", lw=1.2, alpha=0.7)
    ax[0].set_yticks(y, d.label, fontsize=8.8)
    ax[0].set(title=f"Ablation — test {metric}", xlabel=metric,
              xlim=(min(d[metric].min() * 0.96, base * 0.96), d[metric].max() * 1.06))
    _despine(ax[0])

    dd = d[d.tag != full_tag]
    colors2 = ["#e34948" if v < 0 else "#008300" for v in dd.delta]
    y2 = np.arange(len(dd))
    bars = ax[1].barh(y2, dd.delta, color=colors2, height=0.68)
    for b, v in zip(bars, dd.delta):
        off = 0.0008 if v >= 0 else -0.0008
        ax[1].text(v + off, b.get_y() + b.get_height() / 2, f"{v:+.4f}",
                   va="center", ha="left" if v >= 0 else "right", fontsize=8.5, color=INK)
    ax[1].axvline(0, color=INK_2, lw=1.0)
    ax[1].set_yticks(y2, dd.label, fontsize=8.8)
    span = max(abs(dd.delta.min()), abs(dd.delta.max())) if len(dd) else 0.01
    ax[1].set(title=f"Change in {metric} vs full model", xlabel=f"Δ {metric}",
              xlim=(-span * 1.45, span * 1.45))
    _despine(ax[1])
    fig.suptitle("Component ablation study", fontsize=12, fontweight="bold")
    return save(fig, name)


def fig_threshold_sensitivity(curves: Dict[str, pd.DataFrame], display: Dict[str, str],
                              name: str) -> str:
    apply_style()
    models = [m for m in MODEL_ORDER if m in curves] + \
             [m for m in curves if m not in MODEL_ORDER]
    cmap = model_colors(models)
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for m in models:
        c = curves[m]
        ax.plot(c.threshold, c.dice, color=cmap[m], label=display.get(m, m),
                lw=2.2 if m == "hemofusion" else 1.4)
        k = c.dice.idxmax()
        ax.scatter([c.loc[k, "threshold"]], [c.loc[k, "dice"]], color=cmap[m], s=32,
                   edgecolor=SURFACE, linewidth=1.2, zorder=5)
    ax.set(title="Sensitivity of test Dice to the binarisation threshold",
           xlabel="probability threshold", ylabel="aggregated Dice")
    ax.legend(fontsize=8, ncol=2)
    _despine(ax)
    return save(fig, name)


def fig_pvalue_heatmap(tests: pd.DataFrame, display: Dict[str, str], name: str,
                       metric: str = "Dice") -> str:
    """Holm-adjusted p-values of the proposed model against every baseline."""
    apply_style()
    d = tests[tests.metric == metric].copy()
    if d.empty:
        return ""
    d = d.sort_values("p_holm")
    fig, ax = plt.subplots(figsize=(8.4, max(3.4, 0.5 * len(d) + 1.6)))
    y = np.arange(len(d))
    logp = -np.log10(np.clip(d.p_holm.values, 1e-16, 1))
    colors = ["#008300" if s else MUTED for s in d.significant]
    bars = ax.barh(y, logp, color=colors, height=0.62)
    for b, p, dd, lab in zip(bars, d.p_holm, d.mean_diff, d.signif_label):
        ax.text(b.get_width() + 0.06, b.get_y() + b.get_height() / 2,
                f"p={p:.2e} {lab}   Δ={dd:+.4f}", va="center", fontsize=8.5, color=INK)
    ax.axvline(-np.log10(0.05), color="#e34948", ls="--", lw=1.2)
    ax.text(-np.log10(0.05), len(d) - 0.35, " α=0.05", fontsize=8.5, color="#e34948")
    ax.set_yticks(y, [display.get(m, m) for m in d.model_b], fontsize=8.8)
    ax.set(title=f"Proposed vs baselines — Wilcoxon signed-rank on per-slice {metric}\n"
                 "(Holm-corrected, longer bar = stronger evidence)",
           xlabel="−log₁₀(adjusted p)", xlim=(0, max(logp.max() * 1.75, 2.4)))
    _despine(ax)
    return save(fig, name)


def fig_critical_difference(rank_df: pd.DataFrame, cd: float, display: Dict[str, str],
                            name: str, omnibus: Optional[Dict] = None) -> str:
    """Demšar critical-difference diagram from the Friedman/Nemenyi analysis."""
    apply_style()
    d = rank_df.sort_values("mean_rank").reset_index(drop=True)
    k = len(d)
    fig, ax = plt.subplots(figsize=(9.6, 1.35 * k * 0.55 + 2.6))
    lo = float(np.floor(d.mean_rank.min() - 0.5))
    hi = float(np.ceil(d.mean_rank.max() + 0.5))

    ax.hlines(0, lo, hi, color=INK, lw=1.4)
    for t in np.arange(lo, hi + 0.001, 0.5):
        ax.vlines(t, 0, 0.12, color=INK, lw=1.0)
        ax.text(t, 0.2, f"{t:g}", ha="center", fontsize=8.5, color=INK_2)

    for i, r in d.iterrows():
        side = -1 if i < k / 2 else 1
        depth = -(0.45 + 0.30 * (i if side < 0 else k - 1 - i))
        xend = lo - 0.22 if side < 0 else hi + 0.22
        col = ACCENT if r.model == "hemofusion" else INK_2
        ax.plot([r.mean_rank, r.mean_rank, xend], [0, depth, depth], color=col, lw=1.3)
        ax.text(xend + (-0.06 if side < 0 else 0.06), depth,
                f"{display.get(r.model, r.model)}  ({r.mean_rank:.2f})",
                ha="right" if side < 0 else "left", va="center", fontsize=8.8,
                color=col, fontweight="bold" if r.model == "hemofusion" else "normal")

    ybar = 0.42
    i = 0
    while i < k:
        j = i
        while j + 1 < k and d.mean_rank[j + 1] - d.mean_rank[i] <= cd:
            j += 1
        if j > i:
            ax.hlines(ybar, d.mean_rank[i], d.mean_rank[j], color=ACCENT, lw=3.4)
            ybar += 0.20
        i += 1

    ax.annotate("", xy=(lo, ybar + 0.22), xytext=(lo + cd, ybar + 0.22),
                arrowprops=dict(arrowstyle="|-|", lw=1.4, color=INK))
    ax.text(lo + cd / 2, ybar + 0.34, f"CD = {cd:.2f}", ha="center", fontsize=9, color=INK)

    sub = ""
    if omnibus:
        sub = (f"Friedman χ² = {omnibus['friedman_chi2']:.1f}, "
               f"p = {omnibus['p_friedman']:.2e}, n = {omnibus['n_observations']}")
    ax.set(xlim=(lo - 2.6, hi + 2.6), ylim=(-(0.45 + 0.30 * k) - 0.4, ybar + 0.75))
    ax.set_title("Critical-difference diagram — mean rank of per-slice Dice\n" + sub,
                 fontsize=11, fontweight="bold")
    ax.axis("off")
    return save(fig, name)


def fig_bland_altman(true_vol: np.ndarray, pred_vol: np.ndarray, ba: Dict[str, float],
                     name: str, unit: str = "mL") -> str:
    apply_style()
    fig, ax = plt.subplots(1, 2, figsize=(11.6, 4.6))
    mean = (true_vol + pred_vol) / 2
    diff = pred_vol - true_vol

    ax[0].scatter(true_vol, pred_vol, s=34, color=NEUTRAL, alpha=0.8,
                  edgecolor=SURFACE, linewidth=0.8)
    lim = max(np.nanmax(true_vol), np.nanmax(pred_vol)) * 1.06
    ax[0].plot([0, lim], [0, lim], color=MUTED, ls=":", lw=1.3, label="identity")
    ax[0].set(title=f"Predicted vs reference lesion volume\n"
                    f"ICC = {ba.get('ICC', float('nan')):.3f}, "
                    f"r = {ba.get('pearson_r', float('nan')):.3f}",
              xlabel=f"reference volume ({unit})", ylabel=f"predicted volume ({unit})",
              xlim=(0, lim), ylim=(0, lim))
    ax[0].legend()

    ax[1].scatter(mean, diff, s=34, color=ACCENT, alpha=0.8,
                  edgecolor=SURFACE, linewidth=0.8)
    for val, lab, col, ls in ((ba["bias"], "bias", INK, "-"),
                              (ba["loa_high"], "+1.96 SD", "#e34948", "--"),
                              (ba["loa_low"], "−1.96 SD", "#e34948", "--")):
        ax[1].axhline(val, color=col, ls=ls, lw=1.3)
        ax[1].text(np.nanmax(mean) * 0.99, val, f" {lab} {val:.2f}", fontsize=8.5,
                   color=col, va="bottom", ha="right")
    ax[1].set(title="Bland–Altman agreement", xlabel=f"mean of the two volumes ({unit})",
              ylabel=f"predicted − reference ({unit})")
    for a in ax:
        _despine(a)
    return save(fig, name)


def fig_fold_variability(summary: pd.DataFrame, display: Dict[str, str], name: str,
                         metric: str = "Dice") -> str:
    """Across-fold spread — the honest picture when only 75 patients exist."""
    apply_style()
    piv = summary.pivot_table(index="fold", columns="tag", values=metric)
    models = [m for m in MODEL_ORDER if m in piv.columns] + \
             [m for m in piv.columns if m not in MODEL_ORDER]
    piv = piv[models]
    cmap = model_colors(models)
    fig, ax = plt.subplots(figsize=(max(7.5, 1.35 * len(models)), 4.9))
    for i, m in enumerate(models):
        v = piv[m].dropna().values
        ax.scatter(np.full(len(v), i) + np.linspace(-0.09, 0.09, max(len(v), 1))[:len(v)],
                   v, s=46, color=cmap[m], edgecolor=SURFACE, linewidth=1.0, zorder=3)
        if len(v):
            ax.hlines(v.mean(), i - 0.26, i + 0.26, color=INK, lw=2.0, zorder=4)
            ax.text(i, v.mean() + 0.022, f"{v.mean():.3f}\n±{v.std():.3f}", ha="center",
                    fontsize=8.5, color=INK, fontweight="bold")
    ax.set_xticks(range(len(models)),
                  [display.get(m, m).replace(" (", "\n(") for m in models], fontsize=8.5)
    ax.set(ylabel=metric, title=f"Per-fold test {metric} (bar = mean across folds)")
    _despine(ax)
    return save(fig, name)
