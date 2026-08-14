"""SimCLR contrastive pretraining of the CNN encoder.

Only the *training* patients of the current fold are used, so no test-set image
ever contributes to the representation — a detail that is easy to get wrong and
silently inflates results.
"""
from __future__ import annotations

import os
import time
from typing import Dict, List

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import Config, DIR_MODELS
from .dataset import SimCLRDataset, VolumeStore


class SimCLR(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.encoder = timm.create_model(
            cfg.cnn_encoder, pretrained=True, num_classes=0, in_chans=cfg.in_chans
        )
        dim = self.encoder.num_features
        self.projector = nn.Sequential(
            nn.Linear(dim, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Linear(512, 128),
        )

    def forward(self, x):
        h = self.encoder(x)
        return h, self.projector(h)


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """Normalised temperature-scaled cross entropy over 2N in-batch views.

    Computed in fp32: dividing by a small temperature inflates the similarity
    matrix, which is exactly the kind of magnitude fp16 handles badly.
    """
    z1, z2 = z1.float(), z2.float()
    n = z1.size(0)
    z = torch.cat([F.normalize(z1, dim=1), F.normalize(z2, dim=1)], 0)
    sim = (z @ z.T) / temperature
    sim.fill_diagonal_(float("-inf"))
    targets = (torch.arange(2 * n, device=z.device) + n) % (2 * n)
    return F.cross_entropy(sim, targets)


def pretrain_simclr(cfg: Config, store: VolumeStore, train_index: pd.DataFrame,
                    device: str, tag: str, log: List[Dict] | None = None) -> str:
    """Returns the path of the saved encoder checkpoint."""
    path = os.path.join(DIR_MODELS, f"simclr_{tag}.pth")
    if os.path.exists(path):
        print(f"  [ssl] reusing {os.path.basename(path)}")
        return path

    loader = DataLoader(
        SimCLRDataset(store, train_index, cfg),
        batch_size=cfg.ssl_batch_size, shuffle=True, drop_last=True,
        num_workers=cfg.num_workers, pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )
    model = SimCLR(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.ssl_lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.ssl_lr, total_steps=cfg.ssl_epochs * max(len(loader), 1),
        pct_start=0.1,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda" and cfg.amp))

    best = float("inf")
    for ep in range(cfg.ssl_epochs):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        for x1, x2 in loader:
            x1, x2 = x1.to(device, non_blocking=True), x2.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=(device == "cuda" and cfg.amp)):
                _, z1 = model(x1)
                _, z2 = model(x2)
            loss = nt_xent(z1, z2, cfg.ssl_temperature)
            if not torch.isfinite(loss):
                continue
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gnorm = nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            if torch.isfinite(gnorm):
                scaler.step(opt)
            scaler.update()
            sched.step()
            tot += float(loss.detach())
            nb += 1
        avg = tot / max(nb, 1)
        if log is not None:
            log.append({"tag": tag, "epoch": ep + 1, "ssl_loss": avg,
                        "lr": opt.param_groups[0]["lr"], "sec": time.time() - t0})
        if avg < best:
            best = avg
            torch.save({k: v for k, v in model.state_dict().items()
                        if k.startswith("encoder.")}, path)
        if (ep + 1) % max(1, cfg.ssl_epochs // 4) == 0 or ep == 0:
            print(f"  [ssl] epoch {ep+1:3d}/{cfg.ssl_epochs}  NT-Xent {avg:.4f}"
                  f"  ({time.time()-t0:.0f}s)")

    print(f"  [ssl] best NT-Xent {best:.4f} -> {os.path.basename(path)}")
    del model
    torch.cuda.empty_cache()
    return path
