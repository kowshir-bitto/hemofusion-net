"""Class-wise figures: examples, distribution, per-augmentation panels, CAMs.

Everything here is organised by the seven radiologist labels — the five ICH
subtypes, skull fracture, and hemorrhage-free — so the same class vocabulary runs
through the example figures, the result tables and the explainability panels.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import DIR_FIGURES, MULTILABEL, SUBTYPES, Config
from .viz import (ACCENT, GRID, INK, INK_2, MUTED, NEUTRAL, SEQ_BLUE, SERIES,
                  SURFACE, apply_style, model_colors, save, _despine, MODEL_ORDER)

CLASSES: List[str] = SUBTYPES + ["Fracture", "No_Hemorrhage"]

PRETTY: Dict[str, str] = {
    "Intraventricular": "Intraventricular",
    "Intraparenchymal": "Intraparenchymal",
    "Subarachnoid": "Subarachnoid",
    "Epidural": "Epidural",
    "Subdural": "Subdural",
    "Fracture": "Skull fracture",
    "No_Hemorrhage": "No hemorrhage",
}


def class_mask(index: pd.DataFrame, cls: str) -> pd.Series:
    """Boolean selector for one class label."""
    if cls == "No_Hemorrhage":
        return (index.target_ich == 0)
    return index[cls] == 1


def class_counts(index: pd.DataFrame) -> pd.DataFrame:
    """Slice and patient counts per class — the table behind the distribution figure."""
    rows = []
    for c in CLASSES:
        sel = class_mask(index, c)
        sub = index[sel]
        rows.append({
            "class": PRETTY[c],
            "slices": int(sel.sum()),
            "slice_pct": round(100.0 * sel.sum() / max(len(index), 1), 2),
            "patients": int(sub.patient.nunique()),
            "mean_lesion_px": round(float(sub.loc[sub.mask_px > 0, "mask_px"].mean()), 1)
            if (sub.mask_px > 0).any() else 0.0,
            "median_lesion_px": round(float(sub.loc[sub.mask_px > 0, "mask_px"].median()), 1)
            if (sub.mask_px > 0).any() else 0.0,
        })
    return pd.DataFrame(rows)


def pick_class_examples(index: pd.DataFrame, n_per_class: int = 3,
                        prefer_pure: bool = True, seed: int = 0
                        ) -> Dict[str, pd.DataFrame]:
    """Representative slices for each class.

    ``prefer_pure`` favours slices carrying exactly one ICH subtype, so the
    example actually illustrates that subtype instead of a mixed bleed; the
    largest lesions are then preferred because they are legible at figure size.
    """
    rng = np.random.default_rng(seed)
    picks: Dict[str, pd.DataFrame] = {}
    n_sub = index[SUBTYPES].sum(axis=1)

    for c in CLASSES:
        sub = index[class_mask(index, c)].copy()
        if sub.empty:
            picks[c] = sub
            continue
        if c == "No_Hemorrhage":
            sub = sub.sample(frac=1.0, random_state=seed).drop_duplicates("patient")
            picks[c] = sub.head(n_per_class)
            continue
        if prefer_pure and c in SUBTYPES:
            pure = sub[n_sub.reindex(sub.index) == 1]
            if len(pure) >= n_per_class:
                sub = pure
        sub = sub.sort_values("mask_px", ascending=False)
        sub = sub.drop_duplicates("patient")
        picks[c] = sub.head(n_per_class)
    return picks


def fig_class_examples(panels: Dict[str, List[Dict]], name: str, title: str,
                       overlay: bool = True) -> str:
    """Rows = class, columns = example slices.

    Each panel dict carries ``img`` (H,W or H,W,3), ``mask`` and a caption.
    """
    apply_style()
    classes = [c for c in CLASSES if panels.get(c)]
    ncol = max(len(v) for v in panels.values() if v)
    fig, ax = plt_subplots(len(classes), ncol)

    for r, c in enumerate(classes):
        for j in range(ncol):
            a = ax[r][j]
            a.set_xticks([]); a.set_yticks([])
            for sp in a.spines.values():
                sp.set_visible(False)
            if j >= len(panels[c]):
                a.axis("off")
                continue
            p = panels[c][j]
            img = p["img"]
            a.imshow(img, cmap=None if img.ndim == 3 else "gray",
                     vmin=None if img.ndim == 3 else 0,
                     vmax=None if img.ndim == 3 else 1)
            m = p.get("mask")
            if overlay and m is not None and np.any(m > 0.5):
                ov = np.zeros((*m.shape, 4))
                ov[m > 0.5] = [0.84, 0.32, 0.51, 0.45]
                a.imshow(ov)
                a.contour(m > 0.5, levels=[0.5], colors=["#d55181"], linewidths=1.3)
            a.set_xlabel(p.get("caption", ""), fontsize=7.5, color=INK_2)
        ax[r][0].set_ylabel(PRETTY[c], fontsize=9.5, fontweight="bold", color=INK)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    return save(fig, name)


def plt_subplots(nrow: int, ncol: int):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(nrow, ncol, figsize=(2.35 * ncol, 2.62 * nrow), squeeze=False)
    return fig, ax


def fig_class_distribution(index: pd.DataFrame, name: str) -> str:
    """Four views of how the seven classes are distributed."""
    apply_style()
    import matplotlib.pyplot as plt

    cc = class_counts(index)
    fig, ax = plt.subplots(2, 2, figsize=(13.5, 8.6))

    a = ax[0, 0]
    d = cc.sort_values("slices")
    bars = a.barh(d["class"], d.slices, color=NEUTRAL, height=0.64)
    for b, v, p in zip(bars, d.slices, d.slice_pct):
        a.text(v + max(d.slices) * 0.012, b.get_y() + b.get_height() / 2,
               f"{v}  ({p:.1f}%)", va="center", fontsize=8.8, color=INK)
    a.set(title="Slices per class", xlabel="slices", xlim=(0, max(d.slices) * 1.30))
    a.set_xscale("log")

    a = ax[0, 1]
    d = cc.sort_values("patients")
    bars = a.barh(d["class"], d.patients, color=SERIES[4], height=0.64)
    for b, v in zip(bars, d.patients):
        a.text(v + max(d.patients) * 0.012, b.get_y() + b.get_height() / 2, str(v),
               va="center", fontsize=8.8, color=INK)
    a.set(title="Patients per class", xlabel="patients",
          xlim=(0, max(d.patients) * 1.18))

    a = ax[1, 0]
    data, labels = [], []
    for c in SUBTYPES:
        v = index.loc[class_mask(index, c) & (index.mask_px > 0), "mask_px"].values
        if v.size:
            data.append(v)
            labels.append(PRETTY[c])
    if data:
        bp = a.boxplot(data, patch_artist=True, widths=0.6, showfliers=False,
                       medianprops=dict(color=SURFACE, lw=2),
                       whiskerprops=dict(color=INK_2, lw=1.0),
                       capprops=dict(color=INK_2, lw=1.0))
        for patch, col in zip(bp["boxes"], SERIES[1:]):
            patch.set_facecolor(col); patch.set_edgecolor(SURFACE); patch.set_linewidth(2)
        for i, v in enumerate(data):
            a.text(i + 1, np.median(v), f"{int(np.median(v))}", ha="center",
                   va="bottom", fontsize=8, color=INK, fontweight="bold")
        a.set_xticks(range(1, len(labels) + 1), labels, rotation=18, ha="right", fontsize=8.5)
    a.set(title="Lesion area by subtype (median printed)", ylabel="mask pixels",
          yscale="log")

    a = ax[1, 1]
    mat = np.zeros((len(MULTILABEL), len(MULTILABEL)), dtype=int)
    for i, ci in enumerate(MULTILABEL):
        for j, cj in enumerate(MULTILABEL):
            mat[i, j] = int(((index[ci] == 1) & (index[cj] == 1)).sum())
    im = a.imshow(mat, cmap=SEQ_BLUE)
    for i in range(len(MULTILABEL)):
        for j in range(len(MULTILABEL)):
            if mat[i, j]:
                a.text(j, i, mat[i, j], ha="center", va="center", fontsize=8,
                       color=SURFACE if mat[i, j] > mat.max() * 0.55 else INK)
    names = [PRETTY[c] for c in MULTILABEL]
    a.set_xticks(range(len(names)), names, rotation=40, ha="right", fontsize=8)
    a.set_yticks(range(len(names)), names, fontsize=8)
    a.set_title("Label co-occurrence (slices)")
    a.grid(False)
    fig.colorbar(im, ax=a, fraction=0.046)

    for row in ax:
        for a_ in row:
            _despine(a_)
    fig.suptitle("Class distribution across the CT-ICH dataset",
                 fontsize=13, fontweight="bold")
    return save(fig, name)


def fig_single_augmentation(base_img: np.ndarray, base_mask: np.ndarray,
                            variants: List[Tuple[np.ndarray, np.ndarray]],
                            aug_name: str, description: str, name: str) -> str:
    """Original plus N sampled draws of one augmentation.

    The mask is drawn as a contour on every panel, which is what makes it
    visible that a geometric transform was applied to image and label together.
    """
    apply_style()
    import matplotlib.pyplot as plt

    n = len(variants) + 1
    fig, ax = plt.subplots(1, n, figsize=(2.5 * n, 3.05), squeeze=False)
    panels = [(base_img, base_mask, "original")] + \
             [(v[0], v[1], f"sample {i+1}") for i, v in enumerate(variants)]

    for j, (img, msk, cap) in enumerate(panels):
        a = ax[0][j]
        a.imshow(img, cmap=None if img.ndim == 3 else "gray")
        if msk is not None and np.any(msk > 0.5):
            a.contour(msk > 0.5, levels=[0.5], colors=["#d55181"], linewidths=1.4)
        a.set_title(cap, fontsize=9, fontweight="bold" if j == 0 else "normal",
                    color=INK if j == 0 else INK_2)
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(False)
    fig.suptitle(f"{aug_name} — {description}", fontsize=12, fontweight="bold")
    return save(fig, name)


def classwise_segmentation(per_slice: Dict[str, pd.DataFrame], index: pd.DataFrame,
                           metric: str = "Dice") -> pd.DataFrame:
    """Mean per-slice metric for each (model, class) pair on hemorrhagic slices."""
    key = ["patient", "slice"]
    lab = index.set_index(key)
    rows = []
    for model, df in per_slice.items():
        d = df[df.gt_empty == 0].set_index(key)
        common = d.index.intersection(lab.index)
        d = d.loc[common]
        for c in SUBTYPES:
            sel = lab.loc[common, c] == 1
            v = d.loc[sel.values, metric].dropna().values if sel.any() else np.array([])
            rows.append({"model": model, "class": PRETTY[c], "n_slices": int(v.size),
                         metric: float(np.mean(v)) if v.size else np.nan,
                         f"{metric}_std": float(np.std(v)) if v.size else np.nan})
    return pd.DataFrame(rows)


def fig_classwise_metric(table: pd.DataFrame, display: Dict[str, str], name: str,
                         metric: str = "Dice") -> str:
    """Grouped bars: one cluster per class, one bar per model."""
    apply_style()
    import matplotlib.pyplot as plt

    models = [m for m in MODEL_ORDER if m in table.model.unique()] + \
             [m for m in table.model.unique() if m not in MODEL_ORDER]
    cmap = model_colors(models)
    classes = [PRETTY[c] for c in SUBTYPES if PRETTY[c] in set(table["class"])]

    fig, ax = plt.subplots(figsize=(max(9.5, 1.9 * len(classes)), 5.2))
    w = 0.8 / max(len(models), 1)
    x = np.arange(len(classes))
    for i, m in enumerate(models):
        sub = table[table.model == m].set_index("class").reindex(classes)
        vals = sub[metric].values
        off = (i - (len(models) - 1) / 2) * w
        bars = ax.bar(x + off, np.nan_to_num(vals), w * 0.92, color=cmap[m],
                      label=display.get(m, m))
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.2f}",
                        ha="center", fontsize=6.8, color=INK, rotation=90)
    counts = table[table.model == models[0]].set_index("class").reindex(classes).n_slices
    ax.set_xticks(x, [f"{c}\n(n={int(v)})" for c, v in zip(classes, counts.fillna(0))],
                  fontsize=8.5)
    ax.set(ylabel=f"mean per-slice {metric}",
           title=f"Class-wise segmentation performance ({metric})",
           ylim=(0, min(1.0, np.nanmax(table[metric].values) * 1.32)))
    ax.legend(ncol=min(len(models), 5), fontsize=8)
    _despine(ax)
    return save(fig, name)


def fig_cam_by_class(entries: List[Dict], name: str, title: str) -> str:
    """One row per class: CT, ground truth, and each CAM as heatmap + overlay.

    ``entries`` items carry ``cls``, ``ct``, ``gt``, ``pred`` and a ``cams`` dict
    of ``{method: array}``.
    """
    apply_style()
    import matplotlib.pyplot as plt

    methods = list(entries[0]["cams"].keys())
    cols = ["CT slice", "Ground truth"] + \
           [f"{m}\nheatmap" for m in methods] + [f"{m}\noverlay" for m in methods]
    ncol = len(cols)
    fig, ax = plt.subplots(len(entries), ncol, figsize=(2.28 * ncol, 2.55 * len(entries)),
                           squeeze=False)

    for r, e in enumerate(entries):
        ct, gt = e["ct"], e["gt"]
        ax[r][0].imshow(ct, cmap="gray", vmin=0, vmax=1)

        ax[r][1].imshow(ct, cmap="gray", vmin=0, vmax=1)
        if np.any(gt > 0.5):
            ov = np.zeros((*gt.shape, 4)); ov[gt > 0.5] = [0.0, 0.51, 0.0, 0.5]
            ax[r][1].imshow(ov)
            ax[r][1].contour(gt > 0.5, levels=[0.5], colors=["#008300"], linewidths=1.2)

        for k, meth in enumerate(methods):
            cam = e["cams"][meth]
            ax[r][2 + k].imshow(cam, cmap="jet", vmin=0, vmax=1)
            a = ax[r][2 + len(methods) + k]
            a.imshow(ct, cmap="gray", vmin=0, vmax=1)
            a.imshow(cam, cmap="jet", alpha=0.45, vmin=0, vmax=1)
            if np.any(gt > 0.5):
                a.contour(gt > 0.5, levels=[0.5], colors=["#ffffff"], linewidths=1.1)

        for c in range(ncol):
            if r == 0:
                ax[r][c].set_title(cols[c], fontsize=8.8, fontweight="bold")
            ax[r][c].set_xticks([]); ax[r][c].set_yticks([])
            for sp in ax[r][c].spines.values():
                sp.set_visible(False)
        ax[r][0].set_ylabel(f"{PRETTY[e['cls']]}\npt {e['patient']} · sl {e['slice']}",
                            fontsize=8.5, fontweight="bold", color=INK)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    return save(fig, name)


def fig_xai_quantitative(rows: pd.DataFrame, name: str) -> str:
    """Do the heatmaps actually land on the lesion? — per method and per class."""
    apply_style()
    import matplotlib.pyplot as plt

    d = rows.dropna(subset=["energy_in_gt"])
    if d.empty:
        return ""
    methods = sorted(d.method.unique())
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))

    for j, (met, lab) in enumerate([("pointing_hit", "Pointing-game accuracy"),
                                    ("energy_in_gt", "CAM energy inside lesion"),
                                    ("saliency_ratio", "Saliency ratio (in/out)")]):
        a = ax[j]
        vals = [d.loc[d.method == m, met].dropna().values for m in methods]
        means = [float(np.mean(v)) if len(v) else np.nan for v in vals]
        bars = a.bar(range(len(methods)), means, 0.6,
                     color=[SERIES[i + 1] for i in range(len(methods))])
        for b, v, arr in zip(bars, means, vals):
            if np.isfinite(v):
                a.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}\n(n={len(arr)})",
                       ha="center", va="bottom", fontsize=8.5, color=INK)
        a.set_xticks(range(len(methods)), methods, fontsize=9)
        a.set(title=lab)
        a.set_ylim(0, max([m for m in means if np.isfinite(m)] + [0.1]) * 1.3)
        _despine(a)
    fig.suptitle("Quantitative explainability — a heatmap is only useful if it "
                 "points at the lesion", fontsize=12, fontweight="bold")
    return save(fig, name)
