"""Dataset, 2.5D channel assembly, augmentation and patient-wise CV splits."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import albumentations as A
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .config import IMAGENET_MEAN, IMAGENET_STD, MULTILABEL, Config
from .preprocess import load_patient


class VolumeStore:
    """Holds every patient volume in RAM (~0.5 GB at 256 px) so the 16 training
    runs of the study never touch disk twice."""

    def __init__(self, cfg: Config, patients: List[int]):
        self.cfg = cfg
        self.vols: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        for p in patients:
            self.vols[int(p)] = load_patient(cfg, int(p))

    def slice_channels(self, pid: int, s: int, context: int) -> np.ndarray:
        """Assemble the network input for one slice.

        Channel order: ``[brain(-k) ... brain(-1), brain, subdural, bone,
        brain(+1) ... brain(+k)]`` — the current slice's three windows stay in
        channels ``k..k+2`` so that a 3-channel model is a strict sub-case.
        """
        img, _ = self.vols[pid]
        n = img.shape[0]
        pre = [img[np.clip(s - k, 0, n - 1), :, :, 0] for k in range(context, 0, -1)]
        post = [img[np.clip(s + k, 0, n - 1), :, :, 0] for k in range(1, context + 1)]
        cur = [img[s, :, :, c] for c in range(3)]
        return np.stack(pre + cur + post, axis=-1)

    def mask(self, pid: int, s: int) -> np.ndarray:
        return (self.vols[pid][1][s] > 127).astype(np.float32)


def _norm_stats(c: int) -> Tuple[List[float], List[float]]:
    """Per-channel ImageNet statistics, tiled for the 2.5D channel layout."""
    mean = [IMAGENET_MEAN[i % 3] for i in range(c)]
    std = [IMAGENET_STD[i % 3] for i in range(c)]
    return mean, std


def train_transform(cfg: Config) -> A.Compose:
    mean, std = _norm_stats(cfg.in_chans)
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Affine(scale=(0.9, 1.1), translate_percent=(-0.06, 0.06),
                     rotate=(-15, 15), p=0.7),
            A.RandomBrightnessContrast(brightness_limit=0.12, contrast_limit=0.12, p=0.4),
            A.GaussNoise(std_range=(0.02, 0.08), p=0.25),
            A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(0.03, 0.10),
                            hole_width_range=(0.03, 0.10), p=0.15),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )


def eval_transform(cfg: Config) -> A.Compose:
    mean, std = _norm_stats(cfg.in_chans)
    return A.Compose([A.Normalize(mean=mean, std=std, max_pixel_value=255.0), ToTensorV2()])


def ssl_transform(cfg: Config) -> A.Compose:
    """Stronger view generator for SimCLR contrastive pretraining."""
    mean, std = _norm_stats(cfg.in_chans)
    return A.Compose(
        [
            A.RandomResizedCrop(size=(cfg.img_size, cfg.img_size), scale=(0.5, 1.0), p=1.0),
            A.HorizontalFlip(p=0.5),
            A.Affine(rotate=(-20, 20), p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
            A.GaussNoise(std_range=(0.02, 0.10), p=0.4),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )


class ICHDataset(Dataset):
    """Returns ``(image, mask, ich_label, multilabel)`` for one CT slice."""

    def __init__(self, store: VolumeStore, index: pd.DataFrame, cfg: Config,
                 transform: Optional[A.Compose] = None):
        self.store = store
        self.df = index.reset_index(drop=True)
        self.cfg = cfg
        self.tfm = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        r = self.df.iloc[i]
        pid, s = int(r["patient"]), int(r["slice"])
        img = self.store.slice_channels(pid, s, self.cfg.context_slices)
        msk = self.store.mask(pid, s)

        if self.tfm is not None:
            out = self.tfm(image=img, mask=msk)
            img_t, msk_t = out["image"], out["mask"]
        else:
            img_t = torch.from_numpy(img.transpose(2, 0, 1).astype(np.float32) / 255.0)
            msk_t = torch.from_numpy(msk)

        return (
            img_t.float(),
            msk_t.unsqueeze(0).float(),
            torch.tensor([float(r["target_ich"])]),
            torch.tensor([float(r[c]) for c in MULTILABEL]),
        )


class SimCLRDataset(Dataset):
    """Two independently augmented views of the same slice."""

    def __init__(self, store: VolumeStore, index: pd.DataFrame, cfg: Config):
        self.store, self.df, self.cfg = store, index.reset_index(drop=True), cfg
        self.tfm = ssl_transform(cfg)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        r = self.df.iloc[i]
        img = self.store.slice_channels(int(r["patient"]), int(r["slice"]), self.cfg.context_slices)
        return self.tfm(image=img)["image"].float(), self.tfm(image=img)["image"].float()


def make_folds(index: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Assign every patient to one of ``cfg.n_folds`` test folds.

    Stratification is on the patient's hemorrhage burden (none / low / high
    bleed-slice count) so each fold carries a comparable amount of positive
    signal — critical with only 75 patients.
    """
    per = (index.groupby("patient")
                .agg(bleed=("has_bleed", "sum"), n=("has_bleed", "size"))
                .reset_index())
    frac = per.bleed / per.n
    per["stratum"] = np.select(
        [per.bleed == 0, frac <= 0.15], [0, 1], default=2
    ).astype(int)

    skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
    per["fold"] = -1
    for k, (_, te) in enumerate(skf.split(per.patient, per.stratum)):
        per.loc[te, "fold"] = k
    return per[["patient", "fold", "stratum", "bleed", "n"]]


def split_patients(folds: pd.DataFrame, fold: int, cfg: Config
                   ) -> Tuple[List[int], List[int], List[int]]:
    """(train, inner-val, test) patient ids for one CV fold."""
    test = folds[folds.fold == fold]
    pool = folds[folds.fold != fold]
    tr, va = train_test_split(
        pool.patient.values,
        test_size=cfg.inner_val_frac,
        random_state=cfg.seed + fold,
        stratify=pool.stratum.values,
    )
    return sorted(tr.tolist()), sorted(va.tolist()), sorted(test.patient.tolist())


def balanced_sampler(df: pd.DataFrame, cfg: Config) -> WeightedRandomSampler:
    """Oversample hemorrhagic slices to ``cfg.positive_ratio`` of every batch."""
    pos = df.has_bleed.values.astype(bool)
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    r = cfg.positive_ratio
    w_pos = r / max(n_pos, 1)
    w_neg = (1.0 - r) / max(n_neg, 1)
    weights = np.where(pos, w_pos, w_neg).astype(np.float64)
    return WeightedRandomSampler(torch.as_tensor(weights), num_samples=len(df), replacement=True)


def make_loaders(store: VolumeStore, index: pd.DataFrame, cfg: Config,
                 train_p: List[int], val_p: List[int], test_p: List[int]):
    tr_df = index[index.patient.isin(train_p)].reset_index(drop=True)
    va_df = index[index.patient.isin(val_p)].reset_index(drop=True)
    te_df = index[index.patient.isin(test_p)].reset_index(drop=True)

    common = dict(num_workers=cfg.num_workers, pin_memory=True,
                  persistent_workers=cfg.num_workers > 0)

    train_loader = DataLoader(
        ICHDataset(store, tr_df, cfg, train_transform(cfg)),
        batch_size=cfg.batch_size, sampler=balanced_sampler(tr_df, cfg),
        drop_last=True, **common,
    )
    val_loader = DataLoader(
        ICHDataset(store, va_df, cfg, eval_transform(cfg)),
        batch_size=cfg.eval_batch_size, shuffle=False, **common,
    )
    test_loader = DataLoader(
        ICHDataset(store, te_df, cfg, eval_transform(cfg)),
        batch_size=cfg.eval_batch_size, shuffle=False, **common,
    )
    return train_loader, val_loader, test_loader, tr_df, va_df, te_df
