"""Statistical analysis of the model comparison.

The tests are chosen for what this dataset actually is: paired observations
(every model sees the identical test slices), small patient count, and metric
distributions that are strongly non-normal (Dice is bounded and left-skewed).
Hence non-parametric paired tests as the primary analysis, with normality
formally checked rather than assumed, multiplicity controlled by Holm, and
effect sizes reported alongside every p-value.

Implemented
-----------
* Shapiro-Wilk normality screen
* Wilcoxon signed-rank (primary) and paired t-test (secondary)
* Rank-biserial correlation and Cliff's delta as effect sizes
* Holm-Bonferroni correction across the family of comparisons
* BCa-free percentile bootstrap CIs, paired bootstrap for differences
* Fast DeLong test for correlated ROC curves
* McNemar's test (exact for small discordant counts)
* Friedman omnibus test + Nemenyi critical difference
* ICC(2,1) and Bland-Altman limits of agreement for predicted lesion volume
"""
from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def normality(x: Sequence[float], name: str = "") -> Dict[str, object]:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    out: Dict[str, object] = {"variable": name, "n": int(x.size),
                              "mean": float(np.mean(x)) if x.size else np.nan,
                              "median": float(np.median(x)) if x.size else np.nan,
                              "skew": float(stats.skew(x)) if x.size > 2 else np.nan,
                              "kurtosis": float(stats.kurtosis(x)) if x.size > 3 else np.nan}
    if 3 <= x.size <= 5000:
        w, p = stats.shapiro(x)
        out["shapiro_W"] = float(w)
        out["shapiro_p"] = float(p)
        out["normal_at_0.05"] = bool(p > 0.05)
    else:
        out["shapiro_W"] = np.nan
        out["shapiro_p"] = np.nan
        out["normal_at_0.05"] = None
    return out


def rank_biserial(a: np.ndarray, b: np.ndarray) -> float:
    """Effect size matched to the Wilcoxon signed-rank test.

    +1 means ``a`` beats ``b`` on every non-tied pair, -1 the reverse.
    """
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[~np.isnan(d)]
    d = d[d != 0]
    if d.size == 0:
        return 0.0
    r = stats.rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    """Non-parametric, unpaired dominance measure — reported for context."""
    a = np.asarray(a, float)[~np.isnan(a)]
    b = np.asarray(b, float)[~np.isnan(b)]
    if a.size == 0 or b.size == 0:
        return np.nan
    u = stats.mannwhitneyu(a, b, alternative="two-sided").statistic
    return float(2 * u / (a.size * b.size) - 1)


def cohens_d_paired(a: Sequence[float], b: Sequence[float]) -> float:
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[~np.isnan(d)]
    sd = np.std(d, ddof=1)
    return float(np.mean(d) / sd) if sd > 0 else np.nan


def paired_comparison(a: Sequence[float], b: Sequence[float], name_a: str, name_b: str,
                      metric: str, higher_is_better: bool = True) -> Dict[str, object]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]

    row: Dict[str, object] = {
        "metric": metric, "model_a": name_a, "model_b": name_b, "n_pairs": int(a.size),
        "mean_a": float(np.mean(a)) if a.size else np.nan,
        "mean_b": float(np.mean(b)) if b.size else np.nan,
        "median_a": float(np.median(a)) if a.size else np.nan,
        "median_b": float(np.median(b)) if b.size else np.nan,
        "higher_is_better": higher_is_better,
    }
    row["mean_diff"] = row["mean_a"] - row["mean_b"]

    if a.size < 3 or np.allclose(a, b):
        row.update(wilcoxon_W=np.nan, p_wilcoxon=1.0, p_ttest=1.0,
                   rank_biserial=0.0, cliffs_delta=0.0, cohens_d=0.0,
                   n_wins=0, n_losses=0, n_ties=int(a.size))
        return row

    try:
        w, p = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        row["wilcoxon_W"] = float(w)
        row["p_wilcoxon"] = float(p)
    except ValueError:
        row["wilcoxon_W"] = np.nan
        row["p_wilcoxon"] = 1.0

    t, pt = stats.ttest_rel(a, b)
    row["t_stat"] = float(t)
    row["p_ttest"] = float(pt)
    row["rank_biserial"] = rank_biserial(a, b)
    row["cliffs_delta"] = cliffs_delta(a, b)
    row["cohens_d"] = cohens_d_paired(a, b)
    d = a - b
    row["n_wins"] = int(np.sum(d > 0))
    row["n_losses"] = int(np.sum(d < 0))
    row["n_ties"] = int(np.sum(d == 0))
    return row


def holm_correction(pvals: Sequence[float], alpha: float = 0.05
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """Holm-Bonferroni step-down: returns (adjusted p, reject flags)."""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    adj = np.empty(n)
    running = 0.0
    for i, idx in enumerate(order):
        running = max(running, (n - i) * p[idx])
        adj[idx] = min(running, 1.0)
    return adj, adj < alpha


def stars(p: float) -> str:
    if not np.isfinite(p):
        return "n/a"
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def bootstrap_ci(x: Sequence[float], n_boot: int = 10000, alpha: float = 0.05,
                 seed: int = 42, stat=np.nanmean) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return {"point": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    boots = stat(x[idx], axis=1)
    return {"point": float(stat(x)),
            "ci_low": float(np.percentile(boots, 100 * alpha / 2)),
            "ci_high": float(np.percentile(boots, 100 * (1 - alpha / 2))),
            "se": float(np.std(boots, ddof=1))}


def paired_bootstrap_diff(a: Sequence[float], b: Sequence[float], n_boot: int = 10000,
                          alpha: float = 0.05, seed: int = 42) -> Dict[str, float]:
    """CI of the mean paired difference — resamples *pairs*, preserving the
    correlation that makes the comparison powerful."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    if a.size == 0:
        return {"diff": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_boot": np.nan}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    diffs = (a[idx] - b[idx]).mean(axis=1)
    obs = float(np.mean(a - b))
    p = 2.0 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"diff": obs,
            "ci_low": float(np.percentile(diffs, 100 * alpha / 2)),
            "ci_high": float(np.percentile(diffs, 100 * (1 - alpha / 2))),
            "p_boot": float(min(p, 1.0))}


def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    z = x[order]
    n = x.size
    tr = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n - 1 and z[j + 1] == z[i]:
            j += 1
        tr[i:j + 1] = 0.5 * (i + j) + 1
        i = j + 1
    out = np.empty(n, dtype=float)
    out[order] = tr
    return out


def _fast_delong(preds: np.ndarray, n_pos: int) -> Tuple[np.ndarray, np.ndarray]:
    """Sun & Xu (2014) fast DeLong covariance of AUCs for k predictors."""
    m, n = n_pos, preds.shape[1] - n_pos
    pos, neg = preds[:, :m], preds[:, m:]
    k = preds.shape[0]

    tz = np.empty([k, m + n]); tx = np.empty([k, m]); ty = np.empty([k, n])
    for r in range(k):
        tx[r] = _midrank(pos[r])
        ty[r] = _midrank(neg[r])
        tz[r] = _midrank(preds[r])
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    s = np.cov(v01) / m + np.cov(v10) / n
    return aucs, np.atleast_2d(s)


def delong_test(y_true: Sequence[int], prob_a: Sequence[float], prob_b: Sequence[float]
                ) -> Dict[str, float]:
    """p-value for AUC(a) == AUC(b) on the *same* samples."""
    y = np.asarray(y_true).astype(int).ravel()
    a = np.asarray(prob_a, dtype=float).ravel()
    b = np.asarray(prob_b, dtype=float).ravel()
    ok = ~(np.isnan(a) | np.isnan(b))
    y, a, b = y[ok], a[ok], b[ok]
    if len(np.unique(y)) < 2:
        return {"auc_a": np.nan, "auc_b": np.nan, "z": np.nan, "p_delong": np.nan}

    order = np.argsort(-y)
    y, a, b = y[order], a[order], b[order]
    n_pos = int(y.sum())
    aucs, cov = _fast_delong(np.vstack([a, b]), n_pos)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "z": 0.0, "p_delong": 1.0}
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "z": float(z),
            "p_delong": float(2 * stats.norm.sf(abs(z)))}


def mcnemar_test(y_true: Sequence[int], pred_a: Sequence[int], pred_b: Sequence[int]
                 ) -> Dict[str, float]:
    """Exact binomial test when discordant pairs are few, chi-square otherwise."""
    y = np.asarray(y_true).astype(int).ravel()
    a = (np.asarray(pred_a).astype(int).ravel() == y)
    b = (np.asarray(pred_b).astype(int).ravel() == y)
    n01 = int(np.sum(a & ~b))
    n10 = int(np.sum(~a & b))
    out = {"a_correct_b_wrong": n01, "a_wrong_b_correct": n10,
           "acc_a": float(a.mean()), "acc_b": float(b.mean())}
    n = n01 + n10
    if n == 0:
        out.update(statistic=0.0, p_mcnemar=1.0, exact=True)
        return out
    if n < 25:
        out.update(statistic=float(min(n01, n10)),
                   p_mcnemar=float(stats.binomtest(min(n01, n10), n, 0.5).pvalue),
                   exact=True)
    else:
        chi2 = (abs(n01 - n10) - 1) ** 2 / n
        out.update(statistic=float(chi2), p_mcnemar=float(stats.chi2.sf(chi2, 1)),
                   exact=False)
    return out


def friedman_nemenyi(matrix: pd.DataFrame) -> Tuple[Dict[str, float], pd.DataFrame, pd.DataFrame]:
    """Friedman test over a (observations x models) table, plus Nemenyi post-hoc.

    ``matrix`` rows are paired observations (e.g. test slices or patients) and
    columns are models.  Returns the omnibus result, the mean ranks, and the
    pairwise rank-difference / critical-difference table.
    """
    m = matrix.dropna(axis=0, how="any")
    arrays = [m[c].values for c in m.columns]
    stat, p = stats.friedmanchisquare(*arrays)
    omnibus = {"friedman_chi2": float(stat), "p_friedman": float(p),
               "n_observations": int(len(m)), "k_models": int(m.shape[1])}

    ranks = m.rank(axis=1, ascending=False)
    mean_ranks = ranks.mean(axis=0).sort_values()
    rank_df = pd.DataFrame({"model": mean_ranks.index, "mean_rank": mean_ranks.values})

    n, k = len(m), m.shape[1]
    q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031,
           9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268}
    q = q05.get(k, 3.268)
    cd = q * np.sqrt(k * (k + 1) / (6.0 * n))

    rows = []
    for x, y in itertools.combinations(m.columns, 2):
        diff = abs(mean_ranks[x] - mean_ranks[y])
        rows.append({"model_a": x, "model_b": y, "rank_diff": float(diff),
                     "critical_difference": float(cd), "significant": bool(diff > cd)})
    return omnibus, rank_df, pd.DataFrame(rows)


def icc21(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    """ICC(2,1) — two-way random effects, absolute agreement, single measure."""
    x = np.column_stack([np.asarray(a, float), np.asarray(b, float)])
    x = x[~np.isnan(x).any(axis=1)]
    n, k = x.shape
    if n < 2:
        return {"ICC": np.nan, "n": int(n)}
    grand = x.mean()
    ms_r = k * ((x.mean(axis=1) - grand) ** 2).sum() / (n - 1)
    ms_c = n * ((x.mean(axis=0) - grand) ** 2).sum() / (k - 1)
    ss_t = ((x - grand) ** 2).sum()
    ss_e = ss_t - (ms_r * (n - 1) + ms_c * (k - 1))
    ms_e = ss_e / ((n - 1) * (k - 1))
    denom = ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / n
    return {"ICC": float((ms_r - ms_e) / denom) if denom > 0 else np.nan, "n": int(n)}


def bland_altman(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    """Bias and 95 % limits of agreement between predicted and true volume."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    if a.size == 0:
        return {"bias": np.nan, "loa_low": np.nan, "loa_high": np.nan, "n": 0}
    d = a - b
    bias, sd = float(np.mean(d)), float(np.std(d, ddof=1)) if a.size > 1 else 0.0
    r, p = (stats.pearsonr(a, b) if a.size > 2 else (np.nan, np.nan))
    return {"bias": bias, "sd_diff": sd, "loa_low": bias - 1.96 * sd,
            "loa_high": bias + 1.96 * sd, "pearson_r": float(r), "pearson_p": float(p),
            "n": int(a.size)}


def compare_against_reference(per_slice: Dict[str, pd.DataFrame], reference: str,
                              metrics: Sequence[str] = ("Dice", "IoU", "HD95", "ASSD", "NSD"),
                              key: Sequence[str] = ("patient", "slice"),
                              positive_only: bool = True, alpha: float = 0.05,
                              n_boot: int = 10000) -> pd.DataFrame:
    """Paired tests of every model against ``reference`` on aligned test slices.

    Rows are aligned on (patient, slice) rather than position, so a mismatch in
    ordering can never silently pair the wrong slices.
    """
    if reference not in per_slice:
        raise KeyError(f"reference model '{reference}' not among {list(per_slice)}")

    ref = per_slice[reference]
    if positive_only:
        ref = ref[ref.gt_empty == 0]
    ref = ref.set_index(list(key)).sort_index()

    rows: List[Dict[str, object]] = []
    for name, df in per_slice.items():
        if name == reference:
            continue
        oth = df[df.gt_empty == 0] if positive_only else df
        oth = oth.set_index(list(key)).sort_index()
        common = ref.index.intersection(oth.index)
        for met in metrics:
            if met not in ref.columns or met not in oth.columns:
                continue
            hib = met not in ("HD95", "ASSD")
            r = paired_comparison(ref.loc[common, met].values, oth.loc[common, met].values,
                                  reference, name, met, hib)
            r.update(paired_bootstrap_diff(ref.loc[common, met].values,
                                           oth.loc[common, met].values,
                                           n_boot=n_boot, alpha=alpha))
            rows.append(r)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["p_holm"] = np.nan
    for met, grp in out.groupby("metric"):
        adj, _ = holm_correction(grp["p_wilcoxon"].values, alpha)
        out.loc[grp.index, "p_holm"] = adj
    out["significant"] = out["p_holm"] < alpha
    out["signif_label"] = out["p_holm"].map(stars)
    return out.sort_values(["metric", "p_holm"]).reset_index(drop=True)
