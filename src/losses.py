"""Composite objective for joint segmentation + slice-level classification.

The segmentation term combines three complementary views of the overlap
problem — soft Dice (region), Tversky (recall-weighted, because a missed bleed
costs more than a false alarm) and Focal BCE (hard-pixel mining for the extreme
foreground/background imbalance of ICH) — plus a morphological boundary term
that directly attacks the Hausdorff distance the region losses ignore.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.s = smooth

    def forward(self, logits, target):
        p = torch.sigmoid(logits).flatten(1)
        t = target.flatten(1)
        inter = (p * t).sum(1)
        dice = (2 * inter + self.s) / (p.sum(1) + t.sum(1) + self.s)
        return 1.0 - dice.mean()


class TverskyLoss(nn.Module):
    """``beta > alpha`` penalises false negatives harder than false positives."""

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1.0):
        super().__init__()
        self.a, self.b, self.s = alpha, beta, smooth

    def forward(self, logits, target):
        p = torch.sigmoid(logits).flatten(1)
        t = target.flatten(1)
        tp = (p * t).sum(1)
        fp = (p * (1 - t)).sum(1)
        fn = ((1 - p) * t).sum(1)
        return (1.0 - (tp + self.s) / (tp + self.a * fp + self.b * fn + self.s)).mean()


class FocalBCELoss(nn.Module):
    def __init__(self, alpha: float = 0.85, gamma: float = 2.0):
        super().__init__()
        self.a, self.g = alpha, gamma

    def forward(self, logits, target):
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        pt = torch.exp(-bce)
        at = target * self.a + (1 - target) * (1 - self.a)
        return (at * (1 - pt) ** self.g * bce).mean()


class BoundaryDiceLoss(nn.Module):
    """Dice between the *contours* of prediction and ground truth.

    Contours come from a differentiable morphological gradient (3x3 max-pool of
    x minus 3x3 max-pool of -x), so no distance transform is needed and the term
    costs two pooling operations.  Optimising it tightens HD95/ASSD, which pure
    region losses are blind to.
    """

    def __init__(self, kernel: int = 3, smooth: float = 1.0):
        super().__init__()
        self.k = kernel
        self.s = smooth

    def _contour(self, x):
        pad = self.k // 2
        dil = F.max_pool2d(x, self.k, stride=1, padding=pad)
        ero = -F.max_pool2d(-x, self.k, stride=1, padding=pad)
        return (dil - ero).clamp(0, 1)

    def forward(self, logits, target):
        pc = self._contour(torch.sigmoid(logits)).flatten(1)
        tc = self._contour(target).flatten(1)
        inter = (pc * tc).sum(1)
        return (1.0 - (2 * inter + self.s) / (pc.sum(1) + tc.sum(1) + self.s)).mean()


class MultiTaskLoss(nn.Module):
    """Weighted sum of segmentation, deep-supervision and classification terms.

    Returns the total plus a detached breakdown so every component can be logged
    per epoch and reported in the loss-ablation table.
    """

    def __init__(self, cfg, pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.cfg = cfg
        self.dice = SoftDiceLoss()
        self.tversky = TverskyLoss(cfg.tversky_alpha, cfg.tversky_beta)
        self.focal = FocalBCELoss()
        self.boundary = BoundaryDiceLoss()
        self.register_buffer(
            "pos_weight", pos_weight if pos_weight is not None else torch.tensor(1.0)
        )

    def seg_term(self, logits, target) -> torch.Tensor:
        c = self.cfg
        loss = (c.w_dice * self.dice(logits, target)
                + c.w_tversky * self.tversky(logits, target)
                + c.w_focal * self.focal(logits, target))
        if c.w_boundary > 0:
            loss = loss + c.w_boundary * self.boundary(logits, target)
        return loss

    @staticmethod
    def to_fp32(out: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Promote model outputs to fp32 before any reduction.

        Region losses sum a sigmoid map over every pixel; at 256x256 that total
        reaches 65536, which overflows fp16 (max 65504) and turns Dice/Tversky
        into inf and then NaN.  The loss is a negligible share of the compute, so
        it always runs in fp32 while the forward pass stays in mixed precision.
        """
        cast: Dict[str, torch.Tensor] = {}
        for k, v in out.items():
            if torch.is_tensor(v):
                cast[k] = v.float()
            elif isinstance(v, (list, tuple)):
                cast[k] = [o.float() for o in v]
            else:
                cast[k] = v
        return cast

    def forward(self, out: Dict[str, torch.Tensor], mask, ich, multi):
        c = self.cfg
        out = self.to_fp32(out)
        mask = mask.float()
        parts: Dict[str, torch.Tensor] = {}

        main = self.seg_term(out["seg"], mask)
        parts["seg"] = main
        total = main

        if "ds" in out and c.w_deep_sup > 0:
            aux = sum(self.seg_term(o, mask) for o in out["ds"]) / len(out["ds"])
            parts["deep_sup"] = aux
            total = (1.0 - c.w_deep_sup) * main + c.w_deep_sup * aux

        if "cls" in out and c.w_cls_binary > 0:
            lb = F.binary_cross_entropy_with_logits(
                out["cls"], ich, pos_weight=self.pos_weight
            )
            parts["cls"] = lb
            total = total + c.w_cls_binary * lb

        if "multi" in out and c.w_cls_multilabel > 0:
            lm = F.binary_cross_entropy_with_logits(out["multi"], multi)
            parts["multi"] = lm
            total = total + c.w_cls_multilabel * lm

        parts["total"] = total
        return total, {k: float(v.detach()) for k, v in parts.items()}
