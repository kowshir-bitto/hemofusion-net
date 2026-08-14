"""Segmentation and classification metrics.

Surface metrics are implemented on top of ``scipy`` Euclidean distance
transforms rather than pulled from a helper library, so the exact convention
used in the paper is visible and reproducible:

* **HD95** — ``max`` of the two directed 95th-percentile surface distances
  (the standard symmetric definition),
* **ASSD** — mean over the union of both directed distance sets,
* **NSD(tau)** — normalised surface Dice, the fraction of boundary points within
  a tolerance ``tau`` mm of the other boundary.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.ndimage import binary_erosion, label as cc_label
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
)

EPS = 1e-7


def _border(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    return mask & ~binary_erosion(mask, iterations=1, border_value=0)


def surface_metrics(pred: np.ndarray, gt: np.ndarray, spacing: float = 1.0,
                    tau: float = 2.0) -> Dict[str, float]:
    """HD95 / ASSD / NSD in the units of ``spacing`` (mm when spacing is mm/px)."""
    from scipy.ndimage import distance_transform_edt

    pb, gb = _border(pred.astype(bool)), _border(gt.astype(bool))
    if not pb.any() or not gb.any():
        return {"HD95": np.nan, "ASSD": np.nan, "NSD": np.nan}

    dt_to_gt = distance_transform_edt(~gb, sampling=spacing)
    dt_to_pred = distance_transform_edt(~pb, sampling=spacing)
    d_pg = dt_to_gt[pb]
    d_gp = dt_to_pred[gb]

    hd95 = float(max(np.percentile(d_pg, 95), np.percentile(d_gp, 95)))
    assd = float(np.concatenate([d_pg, d_gp]).mean())
    nsd = float((np.sum(d_pg <= tau) + np.sum(d_gp <= tau)) / (len(d_pg) + len(d_gp)))
    return {"HD95": hd95, "ASSD": assd, "NSD": nsd}


def slice_seg_metrics(pred: np.ndarray, gt: np.ndarray, spacing: float = 1.0,
                      with_surface: bool = True) -> Dict[str, float]:
    """Overlap + surface metrics for a single binary slice.

    Slices with an empty ground truth are scored by convention (perfect when the
    prediction is also empty) but flagged via ``gt_empty`` so the reported
    lesion-level averages can exclude them, as is standard for ICH.
    """
    p = pred.astype(bool)
    g = gt.astype(bool)
    tp = float(np.sum(p & g))
    fp = float(np.sum(p & ~g))
    fn = float(np.sum(~p & g))
    tn = float(np.sum(~p & ~g))

    out: Dict[str, float] = {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "gt_px": float(g.sum()), "pred_px": float(p.sum()),
        "gt_empty": int(not g.any()),
    }

    if not g.any():
        perfect = not p.any()
        out.update(Dice=1.0 if perfect else 0.0, IoU=1.0 if perfect else 0.0,
                   Precision=1.0 if perfect else 0.0, Recall=1.0,
                   Specificity=tn / (tn + fp + EPS), VS=1.0 if perfect else 0.0,
                   HD95=np.nan, ASSD=np.nan, NSD=np.nan)
        return out

    out["Dice"] = 2 * tp / (2 * tp + fp + fn + EPS)
    out["IoU"] = tp / (tp + fp + fn + EPS)
    out["Precision"] = tp / (tp + fp + EPS)
    out["Recall"] = tp / (tp + fn + EPS)
    out["Specificity"] = tn / (tn + fp + EPS)
    out["VS"] = 1.0 - abs(out["pred_px"] - out["gt_px"]) / (out["pred_px"] + out["gt_px"] + EPS)
    out.update(surface_metrics(p, g, spacing) if with_surface
               else {"HD95": np.nan, "ASSD": np.nan, "NSD": np.nan})
    return out


def aggregate_seg(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """Lesion-level means (positive slices only) plus dataset-level overlap.

    Both views matter: the per-slice mean is what clinicians read, while the
    aggregated (``*_agg``) figures pool tp/fp/fn over the whole test set and so
    are insensitive to how many slices happen to contain a tiny bleed.
    """
    arr = {k: np.array([r.get(k, np.nan) for r in rows], dtype=float) for k in
           ("Dice", "IoU", "Precision", "Recall", "Specificity", "VS", "HD95", "ASSD", "NSD",
            "tp", "fp", "fn", "tn", "gt_empty", "gt_px", "pred_px")}
    pos = arr["gt_empty"] == 0

    out: Dict[str, float] = {}
    for k in ("Dice", "IoU", "Precision", "Recall", "Specificity", "VS", "HD95", "ASSD", "NSD"):
        v = arr[k][pos]
        out[k] = float(np.nanmean(v)) if v.size and not np.all(np.isnan(v)) else float("nan")
        out[k + "_std"] = float(np.nanstd(v)) if v.size and not np.all(np.isnan(v)) else float("nan")

    tp, fp, fn = arr["tp"].sum(), arr["fp"].sum(), arr["fn"].sum()
    out["Dice_agg"] = float(2 * tp / (2 * tp + fp + fn + EPS))
    out["IoU_agg"] = float(tp / (tp + fp + fn + EPS))
    out["Precision_agg"] = float(tp / (tp + fp + EPS))
    out["Recall_agg"] = float(tp / (tp + fn + EPS))
    out["n_slices"] = int(len(rows))
    out["n_pos_slices"] = int(pos.sum())
    return out


def patient_volumes(rows: Sequence[Dict[str, float]], patients: Sequence[int]
                    ) -> Dict[int, Dict[str, float]]:
    """Pool tp/fp/fn and reference volume over every slice of each patient."""
    acc: Dict[int, Dict[str, float]] = {}
    for r, p in zip(rows, patients):
        a = acc.setdefault(int(p), {"tp": 0.0, "fp": 0.0, "fn": 0.0,
                                    "gt_px": 0.0, "pred_px": 0.0})
        a["tp"] += r["tp"]; a["fp"] += r["fp"]; a["fn"] += r["fn"]
        a["gt_px"] += r["gt_px"]; a["pred_px"] += r["pred_px"]
    return acc


def patient_dice(rows: Sequence[Dict[str, float]], patients: Sequence[int]
                 ) -> Dict[int, float]:
    """Volumetric (per-patient) Dice — tp/fp/fn pooled across a patient's slices.

    Patients with no hemorrhage anywhere get ``NaN``, not 0.  Their volumetric
    Dice is genuinely undefined (the denominator is zero whenever the model
    correctly predicts nothing), and scoring a correct empty prediction as total
    failure would penalise every model in proportion to how many hemorrhage-free
    patients happen to fall in the fold — 39 of the 75 patients here.  The
    false-positive burden on those patients is reported separately by
    :func:`lesion_free_burden`, which is the quantity that actually matters for
    them.
    """
    out: Dict[int, float] = {}
    for p, a in patient_volumes(rows, patients).items():
        out[p] = (2 * a["tp"] / (2 * a["tp"] + a["fp"] + a["fn"])
                  if a["gt_px"] > 0 else float("nan"))
    return out


def lesion_free_burden(rows: Sequence[Dict[str, float]], patients: Sequence[int]
                       ) -> Dict[str, float]:
    """False-positive load on patients who have no hemorrhage at all.

    ``clean_rate`` is the share of those patients on whom the model predicts
    nothing whatsoever — the per-patient specificity a radiologist cares about.
    """
    vols = patient_volumes(rows, patients)
    free = {p: a for p, a in vols.items() if a["gt_px"] == 0}
    if not free:
        return {"n_lesion_free_patients": 0, "clean_rate": float("nan"),
                "mean_fp_px": float("nan")}
    fp = np.array([a["fp"] for a in free.values()], dtype=float)
    return {"n_lesion_free_patients": int(len(free)),
            "clean_rate": float(np.mean(fp == 0)),
            "mean_fp_px": float(np.mean(fp)),
            "median_fp_px": float(np.median(fp))}


def remove_small_components(mask: np.ndarray, min_size: int) -> np.ndarray:
    """Delete connected components below ``min_size`` pixels.

    Isolated few-pixel blobs are almost always noise or partial-volume artefact;
    dropping them raises precision at essentially no recall cost.
    """
    if min_size <= 0 or not mask.any():
        return mask
    lab, n = cc_label(mask)
    if n == 0:
        return mask
    keep = np.zeros(n + 1, dtype=bool)
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    keep[1:] = sizes[1:] >= min_size
    return keep[lab]


def binary_cls_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                       threshold: float = 0.5) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    ok = np.isfinite(y_true) & np.isfinite(y_prob)
    y_true = y_true[ok].astype(int)
    y_prob = y_prob[ok]
    y_pred = (y_prob >= threshold).astype(int)

    out: Dict[str, float] = {"threshold": float(threshold), "n": int(y_true.size),
                             "n_pos": int(y_true.sum())}
    if y_true.size == 0 or len(np.unique(y_true)) == 0:
        return out

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    out.update(TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp))
    out["Accuracy"] = accuracy_score(y_true, y_pred)
    out["Sensitivity"] = tp / (tp + fn + EPS)
    out["Specificity"] = tn / (tn + fp + EPS)
    out["Precision"] = tp / (tp + fp + EPS)
    out["NPV"] = tn / (tn + fn + EPS)
    out["F1"] = 2 * tp / (2 * tp + fp + fn + EPS)
    out["BalancedAcc"] = balanced_accuracy_score(y_true, y_pred)
    out["MCC"] = matthews_corrcoef(y_true, y_pred) if len(np.unique(y_pred)) > 1 else 0.0
    out["Kappa"] = cohen_kappa_score(y_true, y_pred)

    if len(np.unique(y_true)) > 1:
        out["AUROC"] = roc_auc_score(y_true, y_prob)
        out["AUPRC"] = average_precision_score(y_true, y_prob)
    else:
        out["AUROC"] = float("nan")
        out["AUPRC"] = float("nan")
    return out


def multilabel_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                       names: Sequence[str], threshold: float = 0.5) -> List[Dict[str, float]]:
    """Per-class metrics for the ICH-subtype / fracture head."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    rows: List[Dict[str, float]] = []
    for j, nm in enumerate(names):
        m = binary_cls_metrics(y_true[:, j], y_prob[:, j], threshold)
        m["class"] = nm
        rows.append(m)
    macro = {"class": "macro-average"}
    keys = [k for k in rows[0] if k not in ("class",)]
    for k in keys:
        vals = [r[k] for r in rows if not (isinstance(r[k], float) and np.isnan(r[k]))]
        macro[k] = float(np.mean(vals)) if vals else float("nan")
    rows.append(macro)
    return rows


def best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray,
                      grid: Optional[np.ndarray] = None) -> float:
    """Operating point chosen on the *inner validation* split, never on test."""
    grid = np.linspace(0.05, 0.95, 37) if grid is None else grid
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    best, best_t = -1.0, 0.5
    for t in grid:
        pred = (y_prob >= t).astype(int)
        tp = float(np.sum((pred == 1) & (y_true == 1)))
        fp = float(np.sum((pred == 1) & (y_true == 0)))
        fn = float(np.sum((pred == 0) & (y_true == 1)))
        f1 = 2 * tp / (2 * tp + fp + fn + EPS)
        if f1 > best:
            best, best_t = f1, float(t)
    return best_t
