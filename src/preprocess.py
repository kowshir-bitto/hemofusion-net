"""NIfTI -> cached numpy tensors.

For every patient one compressed ``.npz`` is written containing

    img : (S, H, W, 3) uint8   multi-window stack  [brain+CLAHE, subdural, bone]
    msk : (S, H, W)    uint8   binary ICH mask (0/255)

plus a single ``slice_index.csv`` that carries the slice-level labels taken from
``hemorrhage_diagnosis_raw_ct.csv``.  Caching once keeps the 16+ training runs
of the study I/O-free.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, Tuple

import cv2
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import (binary_closing, binary_dilation, binary_fill_holes,
                           label as cc_label)
from tqdm import tqdm

from .config import CT_DIR, DEMOGRAPHICS_CSV, MASK_DIR, SLICE_LABEL_CSV, Config, MULTILABEL


def apply_window(hu: np.ndarray, centre: float, width: float) -> np.ndarray:
    """Map a Hounsfield-unit slice onto [0, 1] with a radiological window."""
    lo, hi = centre - width / 2.0, centre + width / 2.0
    return ((np.clip(hu, lo, hi) - lo) / max(hi - lo, 1e-6)).astype(np.float32)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    lab, n = cc_label(mask)
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return lab == int(np.argmax(sizes))


def intracranial_mask(hu: np.ndarray) -> Tuple[np.ndarray, float]:
    """Extract the intracranial region by exploiting the closed skull ring.

    ``head`` is the largest air-thresholded component (which discards the scanner
    table and the de-identification arc present in this release), ``bone`` is
    dense skull dilated by one pixel to bridge diploe gaps, and the intracranial
    region is the largest component of ``head and not bone``, hole-filled so the
    hemorrhage and ventricles remain inside.

    Returns the mask and the fraction of the head it covers.  A fraction near 1
    means the ring was open and the mask leaked out to the scalp — the caller
    should treat that as a failed strip.

    Note: this dataset contains epidural and subdural hemorrhage, which lie
    against the inner skull table, so *any* stripping risks deleting lesion.
    Table T00 in the report quantifies that loss; stripping is therefore off by
    default and studied as an ablation rather than applied blindly.
    """
    head = binary_fill_holes(_largest_component(hu > -250))
    bone = binary_dilation(hu > 150, iterations=1)
    inner = binary_fill_holes(_largest_component(head & ~bone))
    inner = binary_closing(inner, np.ones((5, 5), bool))
    frac = float(inner.sum()) / max(float(head.sum()), 1.0)
    if frac < 0.15:
        return head, frac
    return inner, frac


def skull_strip(img_u8: np.ndarray, hu: np.ndarray) -> np.ndarray:
    """Zero everything outside the intracranial region."""
    inner, frac = intracranial_mask(hu)
    if frac > 0.90:
        return img_u8
    out = img_u8.copy()
    out[~inner] = 0
    return out


def enhance_clahe(img_u8: np.ndarray, clip: float = 2.0, grid: int = 8) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(img_u8)


def build_slice(hu: np.ndarray, cfg: Config) -> np.ndarray:
    """HU slice -> (H, W, 3) uint8 multi-window stack at ``cfg.img_size``."""
    (c0, w0), (c1, w1), (c2, w2) = cfg.windows

    brain = (apply_window(hu, c0, w0) * 255).astype(np.uint8)
    if cfg.skull_strip:
        brain = skull_strip(brain, hu)
    if cfg.clahe:
        brain = enhance_clahe(brain)

    sub = (apply_window(hu, c1, w1) * 255).astype(np.uint8)
    bone = (apply_window(hu, c2, w2) * 255).astype(np.uint8)

    stack = np.stack([brain, sub, bone], axis=-1)
    return cv2.resize(stack, (cfg.img_size, cfg.img_size), interpolation=cv2.INTER_LINEAR)


def load_slice_labels() -> pd.DataFrame:
    """Slice-level radiologist labels, one row per annotated slice."""
    df = pd.read_csv(SLICE_LABEL_CSV, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"Fracture_Yes_No": "Fracture"})
    df["PatientNumber"] = df["PatientNumber"].astype(int)
    df["SliceNumber"] = df["SliceNumber"].astype(int)
    for c in MULTILABEL + ["No_Hemorrhage"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    df["ICH_label"] = (df[list(MULTILABEL[:-1])].sum(axis=1) > 0).astype(int)
    return df


def load_demographics() -> pd.DataFrame:
    df = pd.read_csv(DEMOGRAPHICS_CSV, encoding="utf-8-sig", skiprows=[1])
    df.columns = [str(c).strip() for c in df.columns]
    pid = [c for c in df.columns if "Patient" in c][0]
    age = [c for c in df.columns if "Age" in c][0]
    out = df[[pid, age, "Gender"]].copy()
    out.columns = ["patient", "age", "gender"]
    out = out.dropna(subset=["patient"])
    out["patient"] = out["patient"].astype(int)
    out["age"] = pd.to_numeric(out["age"], errors="coerce")
    return out.reset_index(drop=True)


def build_cache(cfg: Config, force: bool = False) -> pd.DataFrame:
    """Write one ``.npz`` per patient and return the slice index table."""
    os.makedirs(cfg.cache_dir, exist_ok=True)
    index_csv = os.path.join(cfg.cache_dir, "slice_index.csv")
    if os.path.exists(index_csv) and not force:
        return pd.read_csv(index_csv)

    labels = load_slice_labels()
    files = sorted(f for f in os.listdir(CT_DIR) if f.endswith(".nii"))
    rows = []

    for fname in tqdm(files, desc="preprocess", disable=not sys.stderr.isatty()):
        pid = int(fname[:-4])
        mpath = os.path.join(MASK_DIR, fname)
        if not os.path.exists(mpath):
            continue

        nii = nib.load(os.path.join(CT_DIR, fname))
        ct = nii.get_fdata().astype(np.float32)
        mk = nib.load(mpath).get_fdata()
        n = min(ct.shape[2], mk.shape[2])

        zx, zy = nii.header.get_zooms()[:2]
        spacing_mm = float(np.mean([zx * ct.shape[0] / cfg.img_size,
                                    zy * ct.shape[1] / cfg.img_size]))

        plabels = labels[labels.PatientNumber == pid].sort_values("SliceNumber").reset_index(drop=True)

        imgs = np.zeros((n, cfg.img_size, cfg.img_size, 3), np.uint8)
        msks = np.zeros((n, cfg.img_size, cfg.img_size), np.uint8)

        for s in range(n):
            imgs[s] = build_slice(ct[:, :, s], cfg)
            m = (mk[:, :, s] > 0).astype(np.uint8)
            msks[s] = cv2.resize(m, (cfg.img_size, cfg.img_size),
                                 interpolation=cv2.INTER_NEAREST) * 255

            row: Dict[str, object] = {
                "patient": pid,
                "slice": s,
                "n_slices": n,
                "spacing_mm": spacing_mm,
                "mask_px": int((msks[s] > 127).sum()),
                "has_bleed": int((msks[s] > 127).sum() > 0),
            }
            if s < len(plabels):
                lr = plabels.iloc[s]
                row["ICH_label"] = int(lr["ICH_label"])
                for c in MULTILABEL:
                    row[c] = int(lr[c])
            else:
                row["ICH_label"] = row["has_bleed"]
                for c in MULTILABEL:
                    row[c] = 0
            rows.append(row)

        np.savez_compressed(os.path.join(cfg.cache_dir, f"{pid:03d}.npz"), img=imgs, msk=msks)

    idx = pd.DataFrame(rows)
    idx["target_ich"] = ((idx.ICH_label == 1) | (idx.has_bleed == 1)).astype(int)
    idx.to_csv(index_csv, index=False)
    return idx


def load_patient(cfg: Config, pid: int) -> Tuple[np.ndarray, np.ndarray]:
    d = np.load(os.path.join(cfg.cache_dir, f"{pid:03d}.npz"))
    return d["img"], d["msk"]
