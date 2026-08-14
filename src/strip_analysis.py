"""Quantify what morphological skull stripping costs on this cohort.

Epidural and subdural hemorrhage lie against the inner skull table, so any
intracranial mask risks erasing exactly the lesion the model must find.  This
module measures that loss over every annotated slice, which is what turns
"we did not skull-strip" from a shrug into a justified design decision.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List

import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import CT_DIR, MASK_DIR
from .preprocess import apply_window, intracranial_mask


def analyse_stripping(sample_limit: int | None = None) -> Dict[str, pd.DataFrame]:
    """Per-slice and summary tables of lesion retention under stripping."""
    files = sorted(f for f in os.listdir(CT_DIR) if f.endswith(".nii"))
    rows: List[Dict[str, float]] = []

    for fn in tqdm(files, desc="strip analysis", disable=not sys.stderr.isatty()):
        pid = int(fn[:-4])
        mp = os.path.join(MASK_DIR, fn)
        if not os.path.exists(mp):
            continue
        nii = nib.load(os.path.join(CT_DIR, fn))
        vol = nii.get_fdata()
        msk = nib.load(mp).get_fdata()
        n = min(vol.shape[2], msk.shape[2])
        for s in range(n):
            gt = msk[:, :, s] > 0
            if not gt.any():
                continue
            hu = vol[:, :, s]
            inner, frac = intracranial_mask(hu)
            leaked = frac > 0.90
            kept = float((gt & inner).sum()) / float(gt.sum())
            rows.append({
                "patient": pid, "slice": s, "lesion_px": float(gt.sum()),
                "head_fraction_kept": frac, "strip_leaked": int(leaked),
                "lesion_retained": 1.0 if leaked else kept,
                "lesion_retained_if_forced": kept,
            })
            if sample_limit and len(rows) >= sample_limit:
                break

    per = pd.DataFrame(rows)
    if per.empty:
        return {"per_slice": per, "summary": per}

    def pixel_weighted(col: str) -> float:
        return float((per[col] * per.lesion_px).sum() / per.lesion_px.sum())

    summary = pd.DataFrame([
        {"statistic": "annotated slices analysed", "value": len(per)},
        {"statistic": "lesion pixels total", "value": float(per.lesion_px.sum())},
        {"statistic": "slices where the skull ring leaked (no strip possible)",
         "value": int(per.strip_leaked.sum())},
        {"statistic": "% slices where the ring leaked",
         "value": round(100 * per.strip_leaked.mean(), 2)},
        {"statistic": "lesion pixels retained if stripping is forced (%)",
         "value": round(100 * pixel_weighted("lesion_retained_if_forced"), 2)},
        {"statistic": "slices losing >5% of lesion if stripping is forced",
         "value": int((per.lesion_retained_if_forced < 0.95).sum())},
        {"statistic": "slices losing >25% of lesion if stripping is forced",
         "value": int((per.lesion_retained_if_forced < 0.75).sum())},
        {"statistic": "worst-case slice lesion retention (%)",
         "value": round(100 * per.lesion_retained_if_forced.min(), 2)},
        {"statistic": "decision",
         "value": "skull stripping disabled by default; retained as ablation A13"},
    ])
    return {"per_slice": per, "summary": summary}
