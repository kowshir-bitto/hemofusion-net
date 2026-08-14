"""Validation-only A/B harness for settling the proposed model's recipe.

The first exploratory run left the proposed hybrid behind a plain U-Net, so the
recipe has to be fixed before the final run.  Doing that honestly means the
choice is made on the **inner validation split only** — this script never prints
or returns a test-set number, because a configuration chosen on test is a
configuration whose test score no longer means anything.

    python ab_tune.py --arms capacity,recipe_only --epochs 30

The winner is then written into ``paper_config`` and the final pipeline trains
every model — baselines included — under that same recipe.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import DIR_LOGS, Config, paper_config
from src.dataset import VolumeStore, make_folds, make_loaders, split_patients
from src.engine import (collect_probabilities, prepare_ssl, run_experiment,
                        search_operating_point)
from src.preprocess import build_cache

PROPOSED = "hemofusion"


def arms(base: Config, epochs: int) -> Dict[str, Config]:
    """Each arm isolates one hypothesis for why the hybrid lost."""
    return {
        "capacity": base.clone(epochs=epochs),
        "recipe_only": base.clone(epochs=epochs, cnn_encoder="resnet50",
                                  trans_encoder="pvt_v2_b2"),
        "slim_reg": base.clone(epochs=epochs, fuse_dims=(64, 128, 256, 384),
                               dropout=0.20, weight_decay=3e-4),
        "slim_reg_pos70": base.clone(epochs=epochs, fuse_dims=(64, 128, 256, 384),
                                     dropout=0.20, weight_decay=3e-4,
                                     positive_ratio=0.65),
        "tiny": base.clone(epochs=epochs, cnn_encoder="resnet18",
                           trans_encoder="pvt_v2_b0", fuse_dims=(64, 128, 256, 384)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="capacity,recipe_only")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--model", default=PROPOSED)
    args = ap.parse_args()

    base = paper_config()
    grid = arms(base, args.epochs)
    want = [a.strip() for a in args.arms.split(",")]
    unknown = [a for a in want if a not in grid]
    if unknown:
        raise SystemExit(f"unknown arm(s) {unknown}; available: {sorted(grid)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    index = build_cache(base)
    folds = make_folds(index, base)
    tr, va, te = split_patients(folds, args.fold, base)
    print(f"fold {args.fold}: {len(tr)} train / {len(va)} inner-val patients "
          f"({len(te)} test patients held out and NOT scored here)", flush=True)

    rows: List[Dict] = []
    for name in want:
        cfg = grid[name].clone(run_name=f"ab_{name}")
        store = VolumeStore(cfg, sorted(set(tr) | set(va) | set(te)))
        ssl_ckpt = prepare_ssl(cfg, store, index, tr, args.fold, device, [])
        print(f"\n=== arm '{name}' — {args.model} "
              f"({cfg.cnn_encoder} + {cfg.trans_encoder}) ===", flush=True)
        t0 = time.time()

        r = run_experiment(args.model, cfg, store, index, tr, va, te, args.fold,
                           device, tag=f"ab_{name}", ssl_ckpt=ssl_ckpt,
                           verbose=False)

        hist = pd.DataFrame(r.history)
        rows.append({
            "arm": name,
            "encoders": f"{cfg.cnn_encoder}+{cfg.trans_encoder}",
            "params_M": round(r.summary.get("params_M", float("nan")), 1),
            "val_dice_at_op": round(r.summary.get("val_dice_at_op", float("nan")), 4),
            "val_dice_best": round(float(hist.val_dice.max()), 4),
            "val_dice_last5": round(float(hist.val_dice.tail(5).mean()), 4),
            "val_dice_std_last10": round(float(hist.val_dice.tail(10).std()), 4),
            "val_auroc_best": round(float(np.nanmax(hist.val_auroc.values)), 4),
            "val_loss_min": round(float(hist.val_loss.min()), 4),
            "best_epoch": r.summary.get("best_epoch"),
            "minutes": round((time.time() - t0) / 60, 1),
        })
        hist.to_csv(os.path.join(DIR_LOGS, f"ab_{name}_history.csv"), index=False)
        del store
        torch.cuda.empty_cache()

        out = pd.DataFrame(rows).sort_values("val_dice_at_op", ascending=False)
        out.to_csv(os.path.join(DIR_LOGS, "ab_tuning_results.csv"), index=False)
        print("\n" + out.to_string(index=False), flush=True)

    print("\nSelection is on val_dice_at_op (inner validation). "
          "Test scores are produced once, by the final pipeline run.")


if __name__ == "__main__":
    main()
