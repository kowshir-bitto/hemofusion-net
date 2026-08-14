"""Training / inference engine.

One function, :func:`run_experiment`, takes a model name and a fold and returns
everything the report needs: the training history, the operating point selected
on the inner validation split, per-slice segmentation metrics, and per-slice
classification probabilities.  Every experiment in the study — proposed model,
baselines and ablations — goes through this same path, which is what makes the
comparison fair.
"""
from __future__ import annotations

import contextlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import DIR_MODELS, MULTILABEL, Config
from .dataset import ICHDataset, VolumeStore, eval_transform, make_loaders
from .losses import MultiTaskLoss
from .metrics import (
    aggregate_seg,
    best_f1_threshold,
    binary_cls_metrics,
    lesion_free_burden,
    patient_dice,
    remove_small_components,
    slice_seg_metrics,
)
from .models import SSL_CAPABLE, build_model, count_params, has_pretrained_encoder
from .ssl_pretrain import pretrain_simclr

EPS = 1e-7


class ModelEMA:
    """Exponential moving average of the weights.

    With only ~200 hemorrhage-positive training slices the validation Dice swings
    by more than 0.2 between consecutive epochs, so picking a checkpoint off that
    curve is mostly luck — the "best epoch" is whichever one got a favourable
    batch order.  Averaging the weights damps the swing; the averaged weights are
    what gets validated, selected and saved, so the reported model is the stable
    one rather than a lucky snapshot.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}
        self.n = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.n += 1
        d = min(self.decay, (1.0 + self.n) / (10.0 + self.n))
        msd = model.state_dict()
        for k, shadow in self.shadow.items():
            shadow.mul_(d).add_(msd[k].detach().float(), alpha=1.0 - d)

    @contextlib.contextmanager
    def applied(self, model: nn.Module):
        """Temporarily swap the averaged weights into ``model``."""
        msd = model.state_dict()
        backup = {k: msd[k].detach().clone() for k in self.shadow}
        for k, v in self.shadow.items():
            msd[k].copy_(v)
        try:
            yield model
        finally:
            for k, v in backup.items():
                msd[k].copy_(v)


@dataclass
class ExperimentResult:
    model: str
    fold: int
    history: List[Dict] = field(default_factory=list)
    seg_rows: pd.DataFrame = field(default_factory=pd.DataFrame)
    cls_rows: pd.DataFrame = field(default_factory=pd.DataFrame)
    summary: Dict[str, float] = field(default_factory=dict)
    threshold: float = 0.5
    cls_threshold: float = 0.5
    ckpt: str = ""


@torch.no_grad()
def predict_batch(model: nn.Module, x: torch.Tensor, tta: bool, amp: bool
                  ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Sigmoid probabilities for (mask, ich, multilabel).

    TTA averages the identity view with a horizontal flip — the only augmentation
    that is anatomically valid on an axial head CT.
    """
    dev_type = "cuda" if x.is_cuda else "cpu"
    with torch.amp.autocast(dev_type, enabled=(amp and x.is_cuda)):
        out = model(x)
        seg = torch.sigmoid(out["seg"].float())
        cls = torch.sigmoid(out["cls"].float()) if "cls" in out else None
        multi = torch.sigmoid(out["multi"].float()) if "multi" in out else None

        if tta:
            out2 = model(torch.flip(x, dims=[-1]))
            seg = (seg + torch.flip(torch.sigmoid(out2["seg"].float()), dims=[-1])) / 2
            if cls is not None and "cls" in out2:
                cls = (cls + torch.sigmoid(out2["cls"].float())) / 2
            if multi is not None and "multi" in out2:
                multi = (multi + torch.sigmoid(out2["multi"].float())) / 2
    return seg, cls, multi


@torch.no_grad()
def collect_probabilities(model: nn.Module, loader: DataLoader, device: str,
                          cfg: Config, tta: bool) -> Dict[str, np.ndarray]:
    """Run the whole loader once and keep probabilities in memory.

    Probability maps are stored as float16 (~33 MB for 560 slices at 256 px), so
    the threshold sweep and every metric variant are computed from one forward
    pass instead of re-running the network per threshold.
    """
    model.eval()
    segs, masks, cls_p, cls_t, mul_p, mul_t = [], [], [], [], [], []
    for x, m, ich, multi in loader:
        x = x.to(device, non_blocking=True)
        s, c, u = predict_batch(model, x, tta, cfg.amp)
        segs.append(s.squeeze(1).cpu().numpy().astype(np.float16))
        masks.append(m.squeeze(1).numpy().astype(np.uint8))
        cls_t.append(ich.numpy())
        mul_t.append(multi.numpy())
        cls_p.append(c.cpu().numpy() if c is not None else np.full((x.size(0), 1), np.nan))
        mul_p.append(u.cpu().numpy() if u is not None
                     else np.full((x.size(0), len(MULTILABEL)), np.nan))
    return {
        "seg_prob": np.concatenate(segs),
        "mask": np.concatenate(masks),
        "cls_prob": np.concatenate(cls_p).ravel(),
        "cls_true": np.concatenate(cls_t).ravel(),
        "multi_prob": np.concatenate(mul_p),
        "multi_true": np.concatenate(mul_t),
    }


def safe_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """AUROC that tolerates absent heads and non-finite predictions."""
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true, dtype=float).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    ok = np.isfinite(y_true) & np.isfinite(y_prob)
    if ok.sum() < 2 or len(np.unique(y_true[ok])) < 2:
        return float("nan")
    return float(roc_auc_score(y_true[ok].astype(int), y_prob[ok]))


def dice_at_threshold(seg_prob: np.ndarray, mask: np.ndarray, th: float,
                      min_comp: int = 0) -> float:
    """Dataset-aggregated Dice, the quantity the operating point maximises."""
    tp = fp = fn = 0.0
    for p, g in zip(seg_prob, mask):
        b = p.astype(np.float32) >= th
        if min_comp > 0:
            b = remove_small_components(b, min_comp)
        g = g > 0
        tp += np.sum(b & g)
        fp += np.sum(b & ~g)
        fn += np.sum(~b & g)
    return float(2 * tp / (2 * tp + fp + fn + EPS))


def search_operating_point(probs: Dict[str, np.ndarray], cfg: Config
                           ) -> Tuple[float, int, float]:
    """Grid-search (threshold, min-component) on the inner validation split."""
    lo, hi, n = cfg.threshold_grid
    grid = np.linspace(lo, hi, int(n))
    best = (0.5, 0, -1.0)
    for th in grid:
        d = dice_at_threshold(probs["seg_prob"], probs["mask"], float(th), 0)
        if d > best[2]:
            best = (float(th), 0, d)
    th = best[0]
    for mc in (0, cfg.min_component, cfg.min_component * 3):
        d = dice_at_threshold(probs["seg_prob"], probs["mask"], th, mc)
        if d > best[2]:
            best = (th, int(mc), d)
    return best


def score_predictions(probs: Dict[str, np.ndarray], index: pd.DataFrame, th: float,
                      min_comp: int, with_surface: bool = True
                      ) -> Tuple[pd.DataFrame, Dict[str, float]]:
    rows = []
    for i in range(len(index)):
        r = index.iloc[i]
        b = probs["seg_prob"][i].astype(np.float32) >= th
        if min_comp > 0:
            b = remove_small_components(b, min_comp)
        m = slice_seg_metrics(b, probs["mask"][i] > 0,
                              spacing=float(r.get("spacing_mm", 1.0)),
                              with_surface=with_surface)
        m.update(patient=int(r["patient"]), slice=int(r["slice"]))
        rows.append(m)
    df = pd.DataFrame(rows)
    summ = aggregate_seg(rows)
    pd_ = patient_dice(rows, df.patient.values)
    vals = np.array(list(pd_.values()), dtype=float)
    scored = vals[~np.isnan(vals)]
    summ["Dice_patient"] = float(np.mean(scored)) if scored.size else float("nan")
    summ["Dice_patient_std"] = float(np.std(scored)) if scored.size else float("nan")
    summ["n_patients_scored"] = int(scored.size)
    summ.update(lesion_free_burden(rows, df.patient.values))
    df["patient_dice"] = df.patient.map(pd_)
    return df, summ


def _param_groups(model: nn.Module, cfg: Config, frozen: bool):
    enc = getattr(model, "encoder_parameters", None)
    enc_params = enc() if callable(enc) else [
        p for n, p in model.named_parameters()
        if n.startswith("enc") or n.startswith("cnn.") or n.startswith("trans.")
    ]
    enc_ids = {id(p) for p in enc_params}
    dec_params = [p for p in model.parameters() if id(p) not in enc_ids]

    for p in enc_params:
        p.requires_grad_(not frozen)
    if frozen:
        return [{"params": dec_params, "lr": cfg.base_lr}]
    return [
        {"params": enc_params, "lr": cfg.base_lr * cfg.encoder_lr_mult},
        {"params": dec_params, "lr": cfg.base_lr},
    ]


def _build_scheduler(opt, cfg: Config, steps_per_epoch: int, epochs: int):
    warm = max(1, cfg.warmup_epochs) * max(steps_per_epoch, 1)
    total = max(epochs, 1) * max(steps_per_epoch, 1)
    w = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.05, total_iters=warm)
    c = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(total - warm, 1), eta_min=1e-6)
    return torch.optim.lr_scheduler.SequentialLR(opt, [w, c], milestones=[warm])


def run_experiment(model_name: str, cfg: Config, store: VolumeStore, index: pd.DataFrame,
                   train_p: List[int], val_p: List[int], test_p: List[int],
                   fold: int, device: str, tag: Optional[str] = None,
                   ssl_ckpt: Optional[str] = None, verbose: bool = True,
                   reuse_existing: bool = False
                   ) -> ExperimentResult:
    """Train one model and evaluate it on the held-out fold.

    With ``reuse_existing`` the training loop is skipped whenever a checkpoint for
    this (run, tag, fold) already exists, and the model is only re-evaluated. The
    evaluation path is byte-identical either way — the threshold search still runs
    on the inner validation split and the test fold is still scored once — so a
    resumed run and a fresh one produce the same numbers. This is what makes it
    safe to interrupt a long multi-model run and continue it.
    """
    tag = tag or model_name
    t_start = time.time()
    torch.manual_seed(cfg.seed + fold)
    np.random.seed(cfg.seed + fold)

    train_loader, val_loader, test_loader, tr_df, va_df, te_df = make_loaders(
        store, index, cfg, train_p, val_p, test_p
    )

    model = build_model(model_name, cfg).to(device)
    n_ssl = 0
    if cfg.ssl_enabled and ssl_ckpt and model_name in SSL_CAPABLE:
        n_ssl = model.load_ssl_encoder(ssl_ckpt)

    r = min(max(cfg.positive_ratio, 1e-3), 1 - 1e-3)
    criterion = MultiTaskLoss(
        cfg, pos_weight=torch.tensor((1.0 - r) / r, dtype=torch.float32)
    ).to(device)

    res = ExperimentResult(model=model_name, fold=fold)
    res.ckpt = os.path.join(DIR_MODELS, f"{cfg.run_name}_{tag}_f{fold}.pth")

    skip_training = reuse_existing and os.path.exists(res.ckpt)
    if skip_training and verbose:
        print(f"  [{tag} | fold {fold}] reusing {os.path.basename(res.ckpt)} "
              f"— re-evaluating only")

    pretrained = has_pretrained_encoder(model_name)
    freeze_epochs = cfg.freeze_epochs if pretrained else 0
    frozen = freeze_epochs > 0
    opt = torch.optim.AdamW(_param_groups(model, cfg, frozen), weight_decay=cfg.weight_decay)
    sched = _build_scheduler(opt, cfg, len(train_loader), cfg.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda" and cfg.amp))

    if verbose:
        p = count_params(model)
        init = (f"pretrained{f' + {n_ssl} SSL tensors' if n_ssl else ''}"
                if pretrained else "from scratch")
        print(f"  [{tag} | fold {fold}] {p['params_M']:.1f}M params, {init}"
              f" | freeze {freeze_epochs}ep"
              f" | train {len(tr_df)} / val {len(va_df)} / test {len(te_df)} slices")

    ema = ModelEMA(model, cfg.ema_decay) if cfg.ema_decay > 0 else None
    best_score, best_epoch, stale = -1.0, -1, 0
    for ep in range(0 if not skip_training else cfg.epochs, cfg.epochs):
        if frozen and ep >= freeze_epochs:
            frozen = False
            opt = torch.optim.AdamW(_param_groups(model, cfg, False),
                                    weight_decay=cfg.weight_decay)
            sched = _build_scheduler(opt, cfg, len(train_loader), cfg.epochs - ep)

        model.train()
        t0 = time.time()
        agg: Dict[str, float] = {}
        nb, skipped = 0, 0
        for x, m, ich, multi in train_loader:
            x = x.to(device, non_blocking=True)
            m = m.to(device, non_blocking=True)
            ich = ich.to(device, non_blocking=True)
            multi = multi.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=(device == "cuda" and cfg.amp)):
                out = model(x)
            loss, parts = criterion(out, m, ich, multi)

            if not torch.isfinite(loss):
                skipped += 1
                opt.zero_grad(set_to_none=True)
                continue

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gnorm = nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            if torch.isfinite(gnorm):
                scaler.step(opt)
            else:
                skipped += 1
            scaler.update()
            sched.step()
            if ema is not None:
                ema.update(model)
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + v
            nb += 1
        train_stats = {f"train_{k}": v / max(nb, 1) for k, v in agg.items()}
        train_stats["skipped_steps"] = skipped

        with (ema.applied(model) if ema is not None else contextlib.nullcontext()):
            vp = collect_probabilities(model, val_loader, device, cfg, tta=False)
            val_dice = dice_at_threshold(vp["seg_prob"], vp["mask"], 0.5)
            val_auc = safe_auroc(vp["cls_true"], vp["cls_prob"])

            vl, vnb = 0.0, 0
            model.eval()
            with torch.no_grad():
                for x, m, ich, multi in val_loader:
                    x, m = x.to(device), m.to(device)
                    ich, multi = ich.to(device), multi.to(device)
                    with torch.amp.autocast("cuda", enabled=(device == "cuda" and cfg.amp)):
                        out = model(x)
                    _, parts = criterion(out, m, ich, multi)
                    vl += parts["total"]
                    vnb += 1

            rec = {"model": model_name, "tag": tag, "fold": fold, "epoch": ep + 1,
                   "val_loss": vl / max(vnb, 1), "val_dice": val_dice,
                   "val_auroc": val_auc, "lr": opt.param_groups[-1]["lr"],
                   "frozen": int(frozen), "sec": time.time() - t0, **train_stats}
            res.history.append(rec)

            if val_dice > best_score + 1e-5:
                best_score, best_epoch, stale = val_dice, ep + 1, 0
                torch.save(model.state_dict(), res.ckpt)
            else:
                stale += 1

        if verbose and ((ep + 1) % max(1, cfg.epochs // 7) == 0 or ep == 0):
            print(f"    ep {ep+1:3d}/{cfg.epochs}  loss {rec.get('train_total', 0):.4f}"
                  f"  val_loss {rec['val_loss']:.4f}  val_dice {val_dice:.4f}"
                  f"  auc {val_auc:.3f}  ({rec['sec']:.0f}s)"
                  f"{'  [frozen]' if frozen else ''}")

        if stale >= cfg.early_stop_patience:
            if verbose:
                print(f"    early stop at epoch {ep+1} (best {best_score:.4f} @ {best_epoch})")
            break

    model.load_state_dict(torch.load(res.ckpt, map_location=device))
    val_probs = collect_probabilities(model, val_loader, device, cfg, cfg.tta)
    th, min_comp, val_best = search_operating_point(val_probs, cfg)
    res.threshold = th
    res.cls_threshold = (
        best_f1_threshold(val_probs["cls_true"], val_probs["cls_prob"])
        if not np.isnan(val_probs["cls_prob"]).all() else 0.5
    )

    test_probs = collect_probabilities(model, test_loader, device, cfg, cfg.tta)
    res.seg_rows, res.summary = score_predictions(test_probs, te_df, th, min_comp)

    cls_df = te_df[["patient", "slice", "target_ich"]].copy()
    cls_df["prob"] = test_probs["cls_prob"]
    for j, nm in enumerate(MULTILABEL):
        cls_df[f"true_{nm}"] = test_probs["multi_true"][:, j]
        cls_df[f"prob_{nm}"] = test_probs["multi_prob"][:, j]
    res.cls_rows = cls_df

    if not np.isnan(test_probs["cls_prob"]).all():
        res.summary.update({
            f"cls_{k}": v for k, v in binary_cls_metrics(
                test_probs["cls_true"], test_probs["cls_prob"], res.cls_threshold).items()
        })

    res.summary.update({
        "model": model_name, "tag": tag, "fold": fold,
        "threshold": th, "min_component": min_comp, "val_dice_at_op": val_best,
        "cls_threshold": res.cls_threshold,
        "best_epoch": best_epoch, "epochs_run": len(res.history),
        "train_minutes": (time.time() - t_start) / 60.0,
        "pretrained_encoder": int(pretrained), "freeze_epochs": freeze_epochs,
        "ssl_tensors": n_ssl, **count_params(model),
    })

    if verbose:
        print(f"  -> Dice {res.summary['Dice']:.4f} | IoU {res.summary['IoU']:.4f} | "
              f"HD95 {res.summary['HD95']:.2f}mm | Dice_pat {res.summary['Dice_patient']:.4f}"
              + (f" | AUROC {res.summary.get('cls_AUROC', float('nan')):.4f}"
                 if 'cls_AUROC' in res.summary else "")
              + f" | {res.summary['train_minutes']:.1f}min")

    del model
    torch.cuda.empty_cache()
    return res


def prepare_ssl(cfg: Config, store: VolumeStore, index: pd.DataFrame,
                train_p: List[int], fold: int, device: str,
                ssl_log: List[Dict]) -> Optional[str]:
    if not cfg.ssl_enabled:
        return None
    tr = index[index.patient.isin(train_p)].reset_index(drop=True)
    return pretrain_simclr(cfg, store, tr, device, f"{cfg.run_name}_f{fold}", ssl_log)
