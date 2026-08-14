"""Correctness checks for the statistics and metrics that end up in the paper.

Each test pins an implementation against an independent reference (sklearn,
scipy, statsmodels, or a hand-computed value) rather than against itself.

    ./.venv/bin/python tests/test_stats_metrics.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import stats as S
from src import metrics as M

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))


def test_delong_auc():
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    a = rng.normal(y * 1.2, 1.0)
    b = rng.normal(y * 0.5, 1.0)
    out = S.delong_test(y, a, b)
    ref_a, ref_b = roc_auc_score(y, a), roc_auc_score(y, b)
    check("DeLong AUC(a) == sklearn", abs(out["auc_a"] - ref_a) < 1e-9,
          f"{out['auc_a']:.6f} vs {ref_a:.6f}")
    check("DeLong AUC(b) == sklearn", abs(out["auc_b"] - ref_b) < 1e-9,
          f"{out['auc_b']:.6f} vs {ref_b:.6f}")
    check("DeLong detects a real AUC gap", out["p_delong"] < 0.01,
          f"p={out['p_delong']:.2e}")

    same = S.delong_test(y, a, a.copy())
    check("DeLong on identical predictors gives p=1", same["p_delong"] >= 0.999,
          f"p={same['p_delong']:.4f}")

    tied = S.delong_test(y, np.round(a), np.round(b))
    check("DeLong handles tied scores", np.isfinite(tied["p_delong"]))


def test_mcnemar():
    from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar

    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    pa = np.where(rng.random(300) < 0.85, y, 1 - y)
    pb = np.where(rng.random(300) < 0.70, y, 1 - y)
    out = S.mcnemar_test(y, pa, pb)

    n01, n10 = out["a_correct_b_wrong"], out["a_wrong_b_correct"]
    both = int(np.sum((pa == y) & (pb == y)))
    neither = int(np.sum((pa != y) & (pb != y)))
    tbl = [[both, n01], [n10, neither]]
    ref = sm_mcnemar(tbl, exact=out["exact"], correction=not out["exact"])
    check("McNemar p == statsmodels", abs(out["p_mcnemar"] - float(ref.pvalue)) < 1e-9,
          f"{out['p_mcnemar']:.6e} vs {float(ref.pvalue):.6e}")

    y2 = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    small = S.mcnemar_test(y2, y2.copy(), np.array([1, 0, 1, 0, 1, 0, 1, 0, 0, 1]))
    check("McNemar uses the exact test when discordants < 25", small["exact"] is True)
    check("McNemar with no disagreement gives p=1",
          S.mcnemar_test(y2, y2.copy(), y2.copy())["p_mcnemar"] == 1.0)


def test_holm():
    from statsmodels.stats.multitest import multipletests

    p = np.array([0.001, 0.008, 0.039, 0.041, 0.9])
    adj, rej = S.holm_correction(p, 0.05)
    ref_rej, ref_adj, _, _ = multipletests(p, alpha=0.05, method="holm")
    check("Holm adjusted p == statsmodels", np.allclose(adj, ref_adj),
          f"{np.round(adj,6)} vs {np.round(ref_adj,6)}")
    check("Holm reject flags == statsmodels", np.array_equal(rej, ref_rej))
    check("Holm is monotone non-decreasing in sorted p",
          np.all(np.diff(adj[np.argsort(p)]) >= -1e-12))


def test_paired():
    from scipy import stats as sps

    rng = np.random.default_rng(2)
    a = rng.normal(0.70, 0.10, 120)
    b = a - rng.normal(0.04, 0.05, 120)
    r = S.paired_comparison(a, b, "A", "B", "Dice")
    ref = sps.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    check("Wilcoxon p == scipy", abs(r["p_wilcoxon"] - float(ref.pvalue)) < 1e-12)
    check("mean_diff has the right sign", r["mean_diff"] > 0)
    check("rank-biserial is positive when A dominates", r["rank_biserial"] > 0.5,
          f"rb={r['rank_biserial']:.3f}")
    check("wins > losses when A dominates", r["n_wins"] > r["n_losses"],
          f"{r['n_wins']} vs {r['n_losses']}")

    same = S.paired_comparison(a, a.copy(), "A", "A", "Dice")
    check("identical inputs -> p=1, effect 0",
          same["p_wilcoxon"] == 1.0 and same["rank_biserial"] == 0.0)

    dom = S.paired_comparison(np.arange(20) + 1.0, np.arange(20) + 0.5, "A", "B", "Dice")
    check("rank-biserial == 1 when A wins every pair",
          abs(dom["rank_biserial"] - 1.0) < 1e-12, f"{dom['rank_biserial']:.6f}")


def test_bootstrap():
    rng = np.random.default_rng(3)
    x = rng.normal(0.6, 0.2, 500)
    ci = S.bootstrap_ci(x, n_boot=4000, seed=7)
    check("bootstrap point estimate == sample mean",
          abs(ci["point"] - x.mean()) < 1e-12)
    check("bootstrap CI brackets the mean",
          ci["ci_low"] < x.mean() < ci["ci_high"])
    sem = x.std(ddof=1) / np.sqrt(len(x))
    half = (ci["ci_high"] - ci["ci_low"]) / 2
    check("bootstrap CI width ~ 1.96*SEM", abs(half - 1.96 * sem) < 0.4 * 1.96 * sem,
          f"half={half:.5f} vs {1.96*sem:.5f}")
    check("bootstrap is reproducible under a fixed seed",
          S.bootstrap_ci(x, 1000, seed=11) == S.bootstrap_ci(x, 1000, seed=11))

    a = rng.normal(0.7, 0.1, 200)
    d = S.paired_bootstrap_diff(a, a - 0.05, n_boot=3000, seed=5)
    check("paired bootstrap recovers the offset", abs(d["diff"] - 0.05) < 1e-9)
    check("paired bootstrap CI excludes 0 for a real shift", d["ci_low"] > 0)
    check("paired bootstrap on identical inputs -> diff 0",
          abs(S.paired_bootstrap_diff(a, a.copy(), 500)["diff"]) < 1e-12)


def test_icc():
    judge1 = [9, 6, 8, 7, 10, 6]
    judge2 = [2, 1, 4, 1, 5, 2]
    out = S.icc21(judge1, judge2)
    check("ICC(2,1) matches the hand-worked ANOVA value",
          abs(out["ICC"] - 0.12565) < 5e-5, f"ICC={out['ICC']:.5f} (expected 0.12565)")

    perfect = S.icc21([1.0, 2, 3, 4, 5], [1.0, 2, 3, 4, 5])
    check("ICC == 1 for identical raters", abs(perfect["ICC"] - 1.0) < 1e-9,
          f"ICC={perfect['ICC']:.6f}")

    base = [1.0, 2, 3, 4, 5, 6, 7, 8]
    shifted = S.icc21(base, [b + 3.0 for b in base])
    check("a constant rater offset lowers ICC(2,1)", shifted["ICC"] < 0.75,
          f"ICC={shifted['ICC']:.4f} with a +3 offset")

    rng = np.random.default_rng(9)
    noisy = np.array(base) * 1.0
    agree = S.icc21(noisy, noisy + rng.normal(0, 0.15, len(noisy)))
    check("ICC stays high for close, unbiased raters", agree["ICC"] > 0.95,
          f"ICC={agree['ICC']:.4f}")


def test_bland_altman():
    a = np.array([10.0, 12, 14, 16, 18])
    b = a - 2.0
    ba = S.bland_altman(a, b)
    check("Bland-Altman bias == constant offset", abs(ba["bias"] - 2.0) < 1e-12)
    check("Bland-Altman LoA collapse when the difference is constant",
          abs(ba["loa_high"] - ba["loa_low"]) < 1e-9)
    check("Bland-Altman r == 1 for a linear relation",
          abs(ba["pearson_r"] - 1.0) < 1e-9)


def test_seg_metrics():
    gt = np.zeros((64, 64), bool)
    gt[20:40, 20:40] = True

    exact = M.slice_seg_metrics(gt, gt)
    check("Dice == 1 for a perfect match", abs(exact["Dice"] - 1.0) < 1e-6)
    check("IoU == 1 for a perfect match", abs(exact["IoU"] - 1.0) < 1e-6)
    check("HD95 == 0 for a perfect match", exact["HD95"] == 0.0)
    check("NSD == 1 for a perfect match", abs(exact["NSD"] - 1.0) < 1e-9)

    pred = np.zeros((64, 64), bool)
    pred[20:40, 20:30] = True
    half = M.slice_seg_metrics(pred, gt)
    check("Dice on a half-overlap == 2/3", abs(half["Dice"] - 2 / 3) < 1e-4,
          f"{half['Dice']:.6f}")
    check("IoU on a half-overlap == 0.5", abs(half["IoU"] - 0.5) < 1e-4,
          f"{half['IoU']:.6f}")
    check("precision == 1 when the prediction is contained in the mask",
          abs(half["Precision"] - 1.0) < 1e-4)
    check("recall == 0.5 on a half-overlap", abs(half["Recall"] - 0.5) < 1e-4)

    disjoint = np.zeros((64, 64), bool)
    disjoint[0:5, 0:5] = True
    d = M.slice_seg_metrics(disjoint, gt)
    check("Dice == 0 when prediction and mask are disjoint", d["Dice"] < 1e-6)
    check("HD95 is large for a disjoint prediction", d["HD95"] > 10,
          f"HD95={d['HD95']:.2f}")

    empty = M.slice_seg_metrics(np.zeros((64, 64), bool), np.zeros((64, 64), bool))
    check("empty prediction on an empty mask scores 1 and is flagged",
          empty["Dice"] == 1.0 and empty["gt_empty"] == 1)
    fp_only = M.slice_seg_metrics(disjoint, np.zeros((64, 64), bool))
    check("false positive on an empty mask scores 0", fp_only["Dice"] == 0.0)

    s1 = M.surface_metrics(pred, gt, spacing=1.0)
    s2 = M.surface_metrics(pred, gt, spacing=2.0)
    check("HD95 scales linearly with pixel spacing",
          abs(s2["HD95"] - 2 * s1["HD95"]) < 1e-6,
          f"{s1['HD95']:.3f} -> {s2['HD95']:.3f}")


def test_aggregate():
    rows = [
        {"Dice": 0.8, "IoU": 0.7, "tp": 80, "fp": 10, "fn": 10, "tn": 900,
         "gt_empty": 0, "gt_px": 90, "pred_px": 90, "HD95": 2.0, "ASSD": 1.0,
         "NSD": 0.9, "Precision": 0.9, "Recall": 0.9, "Specificity": 1.0, "VS": 1.0},
        {"Dice": 1.0, "IoU": 1.0, "tp": 0, "fp": 0, "fn": 0, "tn": 1000,
         "gt_empty": 1, "gt_px": 0, "pred_px": 0, "HD95": np.nan, "ASSD": np.nan,
         "NSD": np.nan, "Precision": 1.0, "Recall": 1.0, "Specificity": 1.0, "VS": 1.0},
    ]
    agg = M.aggregate_seg(rows)
    check("per-slice mean ignores empty-mask slices", abs(agg["Dice"] - 0.8) < 1e-9,
          f"Dice={agg['Dice']:.4f} (mean over positive slices only)")
    check("aggregated Dice pools tp/fp/fn", abs(agg["Dice_agg"] - 160 / 180) < 1e-6,
          f"Dice_agg={agg['Dice_agg']:.4f}")
    check("positive-slice count is correct", agg["n_pos_slices"] == 1)
    check("NaN surface metrics do not poison the mean", abs(agg["HD95"] - 2.0) < 1e-9)

    pd_ = M.patient_dice(rows, [7, 7])
    check("patient Dice pools across that patient's slices",
          abs(pd_[7] - 160 / 180) < 1e-6, f"{pd_[7]:.4f}")


def test_components_and_cls():
    m = np.zeros((32, 32), bool)
    m[2:4, 2:4] = True
    m[10:20, 10:20] = True
    kept = M.remove_small_components(m, 12)
    check("small components are removed", kept.sum() == 100, f"{kept.sum()} px kept")
    check("large components survive", kept[15, 15])
    check("min_size=0 is a no-op",
          np.array_equal(M.remove_small_components(m, 0), m))

    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(4)
    y = rng.integers(0, 2, 300)
    p = np.clip(rng.normal(y * 0.4 + 0.3, 0.15), 0, 1)
    cm = M.binary_cls_metrics(y, p, 0.5)
    check("AUROC == sklearn", abs(cm["AUROC"] - roc_auc_score(y, p)) < 1e-12)
    check("confusion counts sum to n", cm["TP"] + cm["FP"] + cm["FN"] + cm["TN"] == 300)
    check("sensitivity == TP/(TP+FN)",
          abs(cm["Sensitivity"] - cm["TP"] / (cm["TP"] + cm["FN"])) < 1e-6)
    check("specificity == TN/(TN+FP)",
          abs(cm["Specificity"] - cm["TN"] / (cm["TN"] + cm["FP"])) < 1e-6)

    withnan = M.binary_cls_metrics(y.astype(float), np.where(rng.random(300) < 0.1, np.nan, p))
    check("classification metrics tolerate NaN predictions",
          np.isfinite(withnan["AUROC"]) and withnan["n"] < 300,
          f"n={withnan['n']}")

    th = M.best_f1_threshold(y, p)
    check("threshold search returns a value inside the grid", 0.05 <= th <= 0.95,
          f"th={th:.3f}")


def test_friedman():
    import pandas as pd
    from scipy import stats as sps

    rng = np.random.default_rng(5)
    n = 60
    df = pd.DataFrame({
        "best": rng.normal(0.75, 0.08, n),
        "mid": rng.normal(0.65, 0.08, n),
        "worst": rng.normal(0.50, 0.08, n),
    })
    omni, ranks, nem = S.friedman_nemenyi(df)
    ref = sps.friedmanchisquare(df.best, df.mid, df.worst)
    check("Friedman chi2 == scipy", abs(omni["friedman_chi2"] - float(ref.statistic)) < 1e-9)
    check("Friedman p == scipy", abs(omni["p_friedman"] - float(ref.pvalue)) < 1e-12)
    check("rank 1 goes to the best model", ranks.iloc[0]["model"] == "best",
          f"order={list(ranks.model)}")
    check("mean ranks lie in [1, k]", ranks.mean_rank.between(1, 3).all())
    check("Nemenyi separates best from worst",
          bool(nem[(nem.model_a == "best") & (nem.model_b == "worst")].significant.iloc[0]))


if __name__ == "__main__":
    for fn in (test_delong_auc, test_mcnemar, test_holm, test_paired, test_bootstrap,
               test_icc, test_bland_altman, test_seg_metrics, test_aggregate,
               test_components_and_cls, test_friedman):
        print(f"\n--- {fn.__name__} ---")
        fn()
    print(f"\n{'='*60}\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("failures:")
        for f in FAIL:
            print("  -", f)
    sys.exit(1 if FAIL else 0)
