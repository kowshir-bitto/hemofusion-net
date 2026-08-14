"""Explainability: Grad-CAM, Grad-CAM++, LayerCAM and Eigen-CAM.

Implemented directly rather than pulled from a library for two reasons: the
scalar being explained has to be a *segmentation* quantity (the mean logit inside
the region of interest, not a class score), and the study needs the raw CAM
tensors to compute quantitative faithfulness metrics — a picture of a heatmap on
its own proves nothing.

Quantitative XAI reported alongside the figures
-----------------------------------------------
* **Pointing game** — does the CAM peak fall inside the reference lesion?
* **Energy-based pointing game** — share of CAM mass inside the lesion.
* **CAM-lesion IoU** at the CAM's Otsu-free 0.5 level.
* **Deletion AUC** — Dice as the most salient regions are progressively erased;
  a faithful explanation makes performance collapse quickly.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-8


def resolve_layer(model: nn.Module, path: str) -> Optional[nn.Module]:
    """Resolve a dotted/indexed module path such as ``dc_full.block.1``."""
    cur: nn.Module = model
    for part in path.split("."):
        if part.isdigit():
            try:
                cur = cur[int(part)]
            except (TypeError, IndexError, KeyError):
                return None
        else:
            if not hasattr(cur, part):
                return None
            cur = getattr(cur, part)
    return cur


LAYER_CANDIDATES: Dict[str, List[Tuple[str, str]]] = {
    "hemofusion": [("bottleneck", "bottleneck (1/32)"),
                   ("dc3", "decoder 1/16"),
                   ("dc1", "decoder 1/4"),
                   ("dc_full", "decoder full-res")],
    "hemoclr_net": [("bottleneck", "bottleneck (1/32)"),
                    ("dc3", "decoder 1/16"),
                    ("dc2", "decoder 1/8"),
                    ("dc0", "decoder full-res")],
}
DEFAULT_CANDIDATES = [("aspp", "ASPP"), ("fuse", "fusion"), ("x40", "bottleneck"),
                      ("decs.0", "decoder 1/16"), ("dec.0", "decoder 1/16"),
                      ("final", "final block"), ("head", "head")]


def pick_layers(model: nn.Module, model_name: str, max_layers: int = 3
                ) -> List[Tuple[str, nn.Module, str]]:
    out: List[Tuple[str, nn.Module, str]] = []
    for path, label in LAYER_CANDIDATES.get(model_name, DEFAULT_CANDIDATES):
        mod = resolve_layer(model, path)
        if isinstance(mod, nn.Module):
            out.append((path, mod, label))
        if len(out) >= max_layers:
            break
    if not out:
        convs = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Conv2d)]
        if convs:
            n, m = convs[len(convs) // 2]
            out.append((n, m, n))
    return out


class CAM:
    """Forward/backward hook based CAM producer.

    Use as a context manager so the hooks are always removed, even on error.
    """

    def __init__(self, model: nn.Module, layer: nn.Module):
        self.model = model
        self.layer = layer
        self.acts: Optional[torch.Tensor] = None
        self.grads: Optional[torch.Tensor] = None
        self._handles = []

    def __enter__(self) -> "CAM":
        self._handles.append(self.layer.register_forward_hook(self._fwd))
        self._handles.append(self.layer.register_full_backward_hook(self._bwd))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return False

    def _fwd(self, _m, _i, out):
        self.acts = out if isinstance(out, torch.Tensor) else out[0]

    def _bwd(self, _m, _gi, go):
        self.grads = go[0] if isinstance(go, (tuple, list)) else go

    def _run(self, x: torch.Tensor, score_fn: Callable[[Dict], torch.Tensor]):
        self.model.zero_grad(set_to_none=True)
        out = self.model(x)
        score = score_fn(out)
        score.backward()
        if self.acts is None or self.grads is None:
            raise RuntimeError("hooks captured nothing — check the target layer")
        return self.acts.detach().float(), self.grads.detach().float(), out

    @staticmethod
    def _normalise(cam: torch.Tensor, size: Tuple[int, int]) -> np.ndarray:
        cam = F.relu(cam).unsqueeze(1)
        cam = F.interpolate(cam, size=size, mode="bilinear", align_corners=False)
        cam = cam.squeeze(1)
        flat = cam.flatten(1)
        lo = flat.min(1)[0].view(-1, 1, 1)
        hi = flat.max(1)[0].view(-1, 1, 1)
        return ((cam - lo) / (hi - lo + EPS)).cpu().numpy()

    def gradcam(self, x, score_fn) -> np.ndarray:
        a, g, _ = self._run(x, score_fn)
        w = g.mean(dim=(2, 3), keepdim=True)
        return self._normalise((w * a).sum(1), x.shape[-2:])

    def gradcam_pp(self, x, score_fn) -> np.ndarray:
        """Grad-CAM++ — pixel-wise weighting via second/third-order gradients."""
        a, g, _ = self._run(x, score_fn)
        g2, g3 = g.pow(2), g.pow(3)
        denom = 2 * g2 + a.sum(dim=(2, 3), keepdim=True) * g3
        alpha = g2 / torch.where(denom.abs() < EPS, torch.full_like(denom, EPS), denom)
        w = (alpha * F.relu(g)).sum(dim=(2, 3), keepdim=True)
        return self._normalise((w * a).sum(1), x.shape[-2:])

    def layercam(self, x, score_fn) -> np.ndarray:
        """LayerCAM — element-wise positive gradients; sharper on shallow layers."""
        a, g, _ = self._run(x, score_fn)
        return self._normalise((F.relu(g) * a).sum(1), x.shape[-2:])

    def eigencam(self, x, score_fn) -> np.ndarray:
        """Eigen-CAM — first principal component of the activations, gradient-free."""
        a, _, _ = self._run(x, score_fn)
        b, c, h, w = a.shape
        cams = []
        for i in range(b):
            m = a[i].reshape(c, h * w)
            m = m - m.mean(dim=1, keepdim=True)
            try:
                _, _, v = torch.linalg.svd(m, full_matrices=False)
                cam = v[0].reshape(h, w)
                if cam.sum() < 0:
                    cam = -cam
            except Exception:
                cam = a[i].mean(0)
            cams.append(cam)
        return self._normalise(torch.stack(cams), x.shape[-2:])


def seg_score(region: Optional[torch.Tensor] = None, use_pred: bool = True):
    """Mean segmentation logit inside a region of interest.

    ``region`` fixes the region explicitly (e.g. the reference mask); otherwise
    the model's own confident predictions define it, which keeps the explanation
    free of ground-truth leakage.
    """
    def fn(out: Dict[str, torch.Tensor]) -> torch.Tensor:
        logits = out["seg"]
        if region is not None:
            w = region
        elif use_pred:
            w = (torch.sigmoid(logits) > 0.5).float()
        else:
            w = torch.ones_like(logits)
        if float(w.sum()) < 1.0:
            flat = logits.flatten(1)
            k = max(1, flat.shape[1] // 200)
            return flat.topk(k, dim=1).values.mean()
        return (logits * w).sum() / (w.sum() + EPS)
    return fn


def cls_score(out_key: str = "cls", index: int = 0):
    """Slice-level classification logit — explains the detection head."""
    def fn(out: Dict[str, torch.Tensor]) -> torch.Tensor:
        if out_key not in out:
            return out["seg"].mean()
        return out[out_key][:, index].mean()
    return fn


def cam_localisation_metrics(cam: np.ndarray, gt: np.ndarray,
                             level: float = 0.5) -> Dict[str, float]:
    g = gt.astype(bool)
    out: Dict[str, float] = {"gt_px": float(g.sum())}
    if not g.any():
        return {**out, "pointing_hit": np.nan, "energy_in_gt": np.nan, "cam_iou": np.nan}

    peak = np.unravel_index(int(np.argmax(cam)), cam.shape)
    out["pointing_hit"] = float(bool(g[peak]))
    total = float(cam.sum())
    out["energy_in_gt"] = float(cam[g].sum() / (total + EPS))
    b = cam >= level
    inter = float(np.sum(b & g))
    out["cam_iou"] = inter / (float(np.sum(b | g)) + EPS)
    out["cam_dice"] = 2 * inter / (float(b.sum()) + float(g.sum()) + EPS)
    out["saliency_ratio"] = float(cam[g].mean() / (cam[~g].mean() + EPS))
    return out


@torch.no_grad()
def deletion_curve(model: nn.Module, x: torch.Tensor, gt: np.ndarray, cam: np.ndarray,
                   steps: int = 10, threshold: float = 0.5) -> Dict[str, object]:
    """Erase the most salient pixels progressively and track Dice.

    A steep drop means the CAM really did point at the evidence the model uses;
    a flat curve means the heatmap is decorative.
    """
    order = np.argsort(-cam.ravel())
    n = order.size
    dices, fracs = [], []
    for s in range(steps + 1):
        k = int(n * s / steps)
        xm = x.clone()
        if k > 0:
            m = np.zeros(n, dtype=bool)
            m[order[:k]] = True
            mask = torch.from_numpy(m.reshape(cam.shape)).to(x.device)
            xm[:, :, mask] = 0.0
        prob = torch.sigmoid(model(xm)["seg"].float())[0, 0].cpu().numpy()
        p = prob >= threshold
        g = gt.astype(bool)
        inter = float(np.sum(p & g))
        dices.append(2 * inter / (float(p.sum()) + float(g.sum()) + EPS))
        fracs.append(s / steps)
    auc = float(np.trapezoid(dices, fracs)) if hasattr(np, "trapezoid") \
        else float(np.trapz(dices, fracs))
    return {"fraction_removed": fracs, "dice": dices, "deletion_auc": auc,
            "dice_drop": float(dices[0] - dices[-1])}


def explain_samples(model: nn.Module, model_name: str, samples: Sequence[Dict],
                    device: str, methods=("gradcam", "gradcam_pp", "layercam"),
                    max_layers: int = 3, run_deletion: bool = True
                    ) -> Tuple[List[Dict], List[Dict]]:
    """Compute every CAM variant for every sample/layer combination.

    Returns ``(cam_records, metric_rows)`` where each record carries the raw
    normalised CAM so the figures and the metrics come from identical tensors.
    """
    model.eval()
    layers = pick_layers(model, model_name, max_layers)
    records: List[Dict] = []
    rows: List[Dict] = []

    for s in samples:
        x = s["x"].unsqueeze(0).to(device)
        gt = s["gt"]
        for path, module, label in layers:
            for meth in methods:
                x_in = x.clone().requires_grad_(True)
                try:
                    with CAM(model, module) as engine:
                        cam = getattr(engine, meth)(x_in, seg_score(use_pred=True))[0]
                except Exception as exc:
                    rows.append({"model": model_name, "patient": s["patient"],
                                 "slice": s["slice"], "layer": label, "method": meth,
                                 "error": str(exc)[:120]})
                    continue

                records.append({"patient": s["patient"], "slice": s["slice"],
                                "layer": label, "layer_path": path, "method": meth,
                                "cam": cam, "ct": s["ct"], "gt": gt,
                                "pred": s.get("pred")})
                m = cam_localisation_metrics(cam, gt)
                m.update(model=model_name, patient=s["patient"], slice=s["slice"],
                         layer=label, method=meth)
                if run_deletion and meth == "gradcam_pp" and path == layers[0][0]:
                    d = deletion_curve(model, x, gt, cam)
                    m["deletion_auc"] = d["deletion_auc"]
                    m["dice_drop"] = d["dice_drop"]
                rows.append(m)
    return records, rows
