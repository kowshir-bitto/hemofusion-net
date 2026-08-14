"""Class-wise stages: example galleries, distribution, per-augmentation panels.

Kept apart from ``run_pipeline`` so the class vocabulary lives in one place and
these figures can be regenerated on their own without retraining anything.
"""
from __future__ import annotations

import os
from typing import Dict, List, Tuple

import albumentations as A
import numpy as np
import pandas as pd

from . import report, viz_class as vc
from .config import CT_DIR, Config, MULTILABEL, SUBTYPES
from .dataset import VolumeStore
from .preprocess import apply_window, build_slice, enhance_clahe, skull_strip

AUGMENTATIONS: List[Tuple[str, str, A.BasicTransform]] = [
    ("Horizontal flip", "left-right mirror; valid on axial head CT because the "
     "skull is broadly symmetric, and it doubles the effective lesion laterality",
     A.HorizontalFlip(p=1.0)),
    ("Rotation", "+/-15 deg, covering the head-tilt variation seen between scans",
     A.Affine(rotate=(-15, 15), p=1.0)),
    ("Scaling", "0.9-1.1x zoom, for differences in head size and field of view",
     A.Affine(scale=(0.9, 1.1), p=1.0)),
    ("Translation", "+/-6 % shift, so the network cannot rely on absolute position",
     A.Affine(translate_percent=(-0.06, 0.06), p=1.0)),
    ("Brightness / contrast", "+/-12 %, mimicking scanner and reconstruction "
     "differences that change apparent blood density",
     A.RandomBrightnessContrast(brightness_limit=0.12, contrast_limit=0.12, p=1.0)),
    ("Gaussian noise", "low-dose acquisition noise", A.GaussNoise(std_range=(0.02, 0.08), p=1.0)),
    ("Coarse dropout", "small occlusions that force the model to use context "
     "rather than a single bright blob",
     A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(0.03, 0.10),
                     hole_width_range=(0.03, 0.10), p=1.0)),
]


def _raw_brain(pid: int, sl: int, cfg: Config) -> np.ndarray:
    """Brain-window view of the untouched NIfTI slice (no strip, no CLAHE)."""
    import cv2
    import nibabel as nib

    ct = nib.load(os.path.join(CT_DIR, f"{pid:03d}.nii")).get_fdata()
    hu = ct[:, :, min(sl, ct.shape[2] - 1)].astype(np.float32)
    c, w = cfg.windows[0]
    img = apply_window(hu, c, w)
    return cv2.resize(img, (cfg.img_size, cfg.img_size), interpolation=cv2.INTER_LINEAR)


def stage_class_examples(cfg: Config, index: pd.DataFrame, fig_captions: Dict[str, str],
                         n_per_class: int = 3) -> Dict[str, pd.DataFrame]:
    """Raw and preprocessed example galleries, plus the class distribution."""
    picks = vc.pick_class_examples(index, n_per_class=n_per_class)
    pids = sorted({int(p) for df in picks.values() for p in df.patient})
    store = VolumeStore(cfg, pids)

    raw_panels: Dict[str, List[Dict]] = {}
    pre_panels: Dict[str, List[Dict]] = {}
    for cls, df in picks.items():
        raw_panels[cls], pre_panels[cls] = [], []
        for _, r in df.iterrows():
            pid, sl = int(r.patient), int(r["slice"])
            msk = store.mask(pid, sl)
            cap = f"pt {pid} · sl {sl}"
            raw_panels[cls].append({"img": _raw_brain(pid, sl, cfg), "mask": msk,
                                    "caption": cap})
            pre_panels[cls].append({"img": store.vols[pid][0][sl, :, :, 0] / 255.0,
                                    "mask": msk, "caption": cap})

    vc.fig_class_examples(
        raw_panels, "F01a_class_examples_raw",
        "Raw CT slices by class (brain window only, lesion outlined)")
    fig_captions["F01a_class_examples_raw"] = (
        "Representative raw axial CT slices for each of the seven radiologist "
        "labels, shown in the brain window (40/80 HU) with no further processing. "
        "The reference hemorrhage mask is outlined in magenta. Slices carrying a "
        "single ICH subtype were preferred so each row illustrates that subtype "
        "rather than a mixed bleed.")

    strip_note = ("brain windowing + CLAHE" if not cfg.skull_strip
                  else "brain windowing + skull stripping + CLAHE")
    vc.fig_class_examples(
        pre_panels, "F01b_class_examples_preprocessed",
        f"The same slices after preprocessing ({strip_note})")
    fig_captions["F01b_class_examples_preprocessed"] = (
        f"The identical slices after the preprocessing chain applied in this study: "
        f"{strip_note}, followed by the three-window stack (brain 40/80, subdural "
        f"80/200, bone 600/2800 HU) and 2.5D stacking of the adjacent slices. "
        + ("Skull stripping is deliberately NOT applied: extra-axial epidural and "
           "subdural bleeds lie directly against the inner skull table, and forcing "
           "a morphological strip removed more than 25 % of the lesion on 6 of the "
           "318 annotated slices (Table T00). Keeping the skull costs the model "
           "nothing that CLAHE cannot recover, and costs no lesion pixels."
           if not cfg.skull_strip else
           "CLAHE restores blood-to-parenchyma contrast after the bright skull is "
           "removed from the intensity range."))

    vc.fig_class_distribution(index, "F02_class_distribution")
    fig_captions["F02_class_distribution"] = (
        "Class distribution. Slice counts are shown on a log axis because the "
        "classes are severely imbalanced, and the co-occurrence matrix shows how "
        "often subtypes appear together on the same slice — the reason "
        "classification is treated as multi-label rather than multi-class.")

    counts = vc.class_counts(index)
    report.register("T04_class_distribution", counts,
                    "Slice and patient counts per class, with lesion-area statistics. "
                    "Percentages are of all 2,814 annotated slices.")
    del store
    return {"class_counts": counts, "picks": picks}


def stage_augmentation_gallery(cfg: Config, index: pd.DataFrame,
                               fig_captions: Dict[str, str], n_samples: int = 4,
                               seed: int = 7) -> None:
    """One figure per augmentation, on a slice with a clearly visible lesion."""
    cand = index[(index.mask_px > 200)].sort_values("mask_px", ascending=False)
    if cand.empty:
        cand = index[index.mask_px > 0].sort_values("mask_px", ascending=False)
    r = cand.iloc[len(cand) // 6]
    pid, sl = int(r.patient), int(r["slice"])

    store = VolumeStore(cfg, [pid])
    img = store.vols[pid][0][sl]
    msk = store.mask(pid, sl)
    base = img[:, :, 0] / 255.0

    for i, (name, desc, tf) in enumerate(AUGMENTATIONS, start=1):
        aug = A.Compose([tf])
        variants = []
        for k in range(n_samples):
            import random
            random.seed(seed + 100 * i + k)
            np.random.seed(seed + 100 * i + k)
            out = aug(image=img, mask=msk)
            variants.append((out["image"][:, :, 0] / 255.0, out["mask"]))
        stem = f"F03{chr(96+i)}_aug_{name.split()[0].lower().replace('/','_')}"
        vc.fig_single_augmentation(base, msk, variants, name, desc, stem)
        fig_captions[stem] = (
            f"{name} augmentation — {desc}. The leftmost panel is the unaugmented "
            f"slice; the rest are independent draws. The magenta contour is the "
            f"reference mask carried through the same transform, confirming that "
            f"geometric operations are applied to image and label together.")

    from .dataset import train_transform
    tfm = train_transform(cfg)
    combo = []
    ch = cfg.context_slices
    for k in range(n_samples):
        stack = store.slice_channels(pid, sl, cfg.context_slices)
        o = tfm(image=stack, mask=msk)
        v = o["image"][ch].numpy()
        v = (v - v.min()) / (np.ptp(v) + 1e-6)
        combo.append((v, o["mask"].numpy()))
    vc.fig_single_augmentation(
        base, msk, combo, "Full training pipeline",
        "all augmentations composed, as sampled during training", "F03z_aug_combined")
    fig_captions["F03z_aug_combined"] = (
        "The complete training-time augmentation pipeline, with every transform "
        "composed and sampled as it is during training. Panels are shown after "
        "normalisation, which is why the greyscale range differs from the raw view.")
    del store
