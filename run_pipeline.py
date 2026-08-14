#!/usr/bin/env python
"""End-to-end study driver.

    python run_pipeline.py --preset smoke     # ~25 min, proves every code path
    python run_pipeline.py --preset quick     # single fold, meaningful epochs
    python run_pipeline.py --preset full      # 5-fold patient-wise CV, paper scale

Stages can be run selectively (``--stages train,stats,xai``) and every stage
persists its artefacts, so a later stage can be re-run without retraining.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import report, stages_class, stats, viz, viz_class
from src.config import (
    DIR_FIGURES, DIR_LOGS, DIR_MODELS, DIR_PREDS, DIR_TABLES, MULTILABEL,
    OUT_ROOT, PRESETS, Config,
)
from src.dataset import (
    ICHDataset, VolumeStore, eval_transform, make_folds, split_patients, train_transform,
)
from src.engine import (
    collect_probabilities, dice_at_threshold, prepare_ssl, run_experiment, score_predictions,
)
from src.metrics import binary_cls_metrics, multilabel_metrics, remove_small_components
from src.models import DISPLAY_NAMES, build_model, display_name
from src.preprocess import build_cache, load_demographics
from src.strip_analysis import analyse_stripping
from src.xai import explain_samples

warnings.filterwarnings("ignore")
PROPOSED = "hemofusion"

FIG_CAPTIONS: Dict[str, str] = {}

REUSE_TRAINED = False


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def banner(msg: str) -> None:
    print(f"\n{'='*78}\n  {msg}\n{'='*78}", flush=True)


def smoke_config() -> Config:
    """Fastest possible run that still touches every stage."""
    return Config(
        run_name="smoke", img_size=256, epochs=3, ssl_epochs=2, freeze_epochs=1,
        warmup_epochs=1, folds=[0], ablation_folds=[0], n_bootstrap=400,
        baselines=["unet", "resunet50", "hemoclr_net"],
        early_stop_patience=99,
    )


PRESETS["smoke"] = smoke_config

ABLATIONS: List[Dict] = [
    {"tag": "A0_full", "label": "Full model (proposed)", "kw": {}},
    {"tag": "A1_no_transformer", "label": "− Transformer stream (CNN only)",
     "kw": {"use_trans_stream": False}},
    {"tag": "A2_no_cnn", "label": "− CNN stream (Transformer only)",
     "kw": {"use_cnn_stream": False}},
    {"tag": "A3_concat_fusion", "label": "− Cross-modal fusion (plain concat)",
     "kw": {"use_cross_fusion": False}},
    {"tag": "A4_no_mrhdc", "label": "− MRHDC bottleneck", "kw": {"use_mrhdc": False}},
    {"tag": "A5_no_coordattn", "label": "− Coordinate attention",
     "kw": {"use_coord_attn": False}},
    {"tag": "A6_no_attngate", "label": "− Attention gates on skips",
     "kw": {"use_attn_gate": False}},
    {"tag": "A7_no_deepsup", "label": "− Deep supervision", "kw": {"use_deep_sup": False}},
    {"tag": "A8_no_clshead", "label": "− Classification branch (single task)",
     "kw": {"use_cls_head": False, "w_cls_binary": 0.0, "w_cls_multilabel": 0.0}},
    {"tag": "A9_no_25d", "label": "− 2.5D context (3-channel input)",
     "kw": {"context_slices": 0}},
    {"tag": "A10_no_boundary", "label": "− Boundary loss term", "kw": {"w_boundary": 0.0}},
    {"tag": "A11_no_ssl", "label": "− SimCLR pretraining", "kw": {"ssl_enabled": False}},
    {"tag": "A12_dice_only", "label": "Loss: Dice only",
     "kw": {"w_dice": 1.0, "w_tversky": 0.0, "w_focal": 0.0, "w_boundary": 0.0}},
    {"tag": "A13_skullstrip", "label": "+ Skull stripping (intracranial masking)",
     "kw": {"skull_strip": True}},
    {"tag": "A14_no_clahe", "label": "− CLAHE enhancement", "kw": {"clahe": False}},
]


def stage_data(cfg: Config) -> Dict:
    banner("STAGE 1/7 — dataset preparation and description")
    idx = build_cache(cfg)
    demo = load_demographics()
    log(f"{len(idx)} slices from {idx.patient.nunique()} patients "
        f"({int(idx.has_bleed.sum())} hemorrhagic, "
        f"{idx.has_bleed.mean()*100:.1f}%)")

    folds = make_folds(idx, cfg)
    folds.to_csv(os.path.join(DIR_TABLES, "cv_folds.csv"), index=False)

    per_patient = (idx.groupby("patient")
                   .agg(slices=("slice", "size"), hemorrhagic=("has_bleed", "sum"),
                        lesion_px=("mask_px", "sum"), spacing_mm=("spacing_mm", "first"))
                   .reset_index()
                   .merge(demo, on="patient", how="left")
                   .merge(folds[["patient", "fold", "stratum"]], on="patient", how="left"))
    report.register("T01_patient_inventory", per_patient,
                    "Per-patient slice counts, hemorrhage burden, demographics and CV fold.")

    counts = {"total_slices": len(idx), "patients": idx.patient.nunique(),
              "hemorrhagic_slices": int(idx.has_bleed.sum()),
              "non_hemorrhagic_slices": int((idx.has_bleed == 0).sum()),
              "hemorrhagic_pct": round(float(idx.has_bleed.mean() * 100), 2),
              "patients_with_ICH": int((per_patient.hemorrhagic > 0).sum()),
              "mean_slices_per_patient": round(float(per_patient.slices.mean()), 2),
              "mean_lesion_px_on_positive": round(
                  float(idx.loc[idx.mask_px > 0, "mask_px"].mean()), 1),
              "median_lesion_px_on_positive": float(
                  idx.loc[idx.mask_px > 0, "mask_px"].median()),
              "in_plane_spacing_mm_mean": round(float(idx.spacing_mm.mean()), 4)}
    label_counts = {f"slices_{c}": int(idx[c].sum()) for c in MULTILABEL}
    report.register("T02_dataset_summary",
                    pd.DataFrame([{**counts, **label_counts}]).T.reset_index()
                    .rename(columns={"index": "statistic", 0: "value"}),
                    "Dataset-level counts used in the Materials section.")

    report.register("T03_cv_fold_composition",
                    folds.groupby("fold").agg(patients=("patient", "size"),
                                              slices=("n", "sum"),
                                              hemorrhagic_slices=("bleed", "sum")).reset_index()
                    .assign(hemorrhagic_pct=lambda d: (d.hemorrhagic_slices / d.slices * 100).round(2)),
                    "Patient-wise stratified 5-fold split; strata are hemorrhage burden.")

    f = viz.fig_dataset_overview(idx, demo, "F01_dataset_overview")
    FIG_CAPTIONS["F01_dataset_overview"] = (
        "Composition of the CT-ICH cohort: slices and hemorrhagic slices per patient, "
        "slice-level class balance, radiologist label prevalence, lesion-area "
        "distribution and patient age by sex.")
    viz.fig_fold_composition(folds, "F02_cv_folds")
    FIG_CAPTIONS["F02_cv_folds"] = (
        "Patient-wise stratified 5-fold cross-validation: patients and hemorrhagic "
        "slices per fold, and hemorrhage prevalence per fold.")

    _fig_preprocessing(cfg, idx)
    _fig_augmentation(cfg, idx)

    stages_class.stage_class_examples(cfg, idx, FIG_CAPTIONS, n_per_class=3)
    stages_class.stage_augmentation_gallery(cfg, idx, FIG_CAPTIONS)

    strip = analyse_stripping()
    if not strip["summary"].empty:
        report.register("T00_skull_strip_cost", strip["summary"],
                        "Cost of morphological skull stripping on this cohort: lesion "
                        "pixels destroyed and slices where the skull ring leaks. Basis "
                        "for disabling stripping in the default pipeline.")
        report.register("T00b_skull_strip_per_slice", strip["per_slice"],
                        "Per-slice lesion retention under intracranial masking.")
        s = strip["summary"].set_index("statistic")["value"]
        log(f"skull-strip audit: ring leaks on {s.get('% slices where the ring leaked')}% "
            f"of annotated slices; forcing the strip would retain only "
            f"{s.get('lesion pixels retained if stripping is forced (%)')}% of lesion "
            f"pixels -> stripping stays off by default")

    log(f"registered {len(report.registered())} tables, wrote figures to {DIR_FIGURES}")
    return {"index": idx, "folds": folds, "demo": demo}


def _fig_preprocessing(cfg: Config, idx: pd.DataFrame) -> None:
    """Show the windowing / skull-strip / CLAHE chain on real hemorrhagic slices."""
    import nibabel as nib
    from src.config import CT_DIR
    from src.preprocess import apply_window, enhance_clahe, skull_strip

    pos = idx[idx.mask_px > 0].sort_values("mask_px", ascending=False)
    picks = pos.drop_duplicates("patient").head(3)[["patient", "slice"]].values
    stages = []
    for pid, sl in picks:
        hu = nib.load(os.path.join(CT_DIR, f"{int(pid):03d}.nii")).get_fdata()[:, :, int(sl)]
        brain = (apply_window(hu, 40, 80) * 255).astype(np.uint8)
        sub = (apply_window(hu, 80, 200) * 255).astype(np.uint8)
        bone = (apply_window(hu, 600, 2800) * 255).astype(np.uint8)

        ch0 = skull_strip(brain, hu) if cfg.skull_strip else brain
        if cfg.clahe:
            ch0 = enhance_clahe(ch0)
        stages.append({
            "raw HU": np.clip((hu + 1000) / 4000, 0, 1),
            "brain window\n(40/80)": brain / 255,
            "channel 1\n(brain + CLAHE)": ch0 / 255,
            "channel 2\nsubdural (80/200)": sub / 255,
            "channel 3\nbone (600/2800)": bone / 255,
            "3-window input": np.stack([ch0, sub, bone], -1) / 255,
            "skull strip\n(ablation only)": skull_strip(brain, hu) / 255,
        })
    viz.fig_preprocessing_stages(stages, "F03_preprocessing")
    FIG_CAPTIONS["F03_preprocessing"] = (
        "Preprocessing chain on three hemorrhagic slices: raw Hounsfield data, the "
        "brain window with CLAHE, the subdural and bone windows, and the composite "
        "three-window input. The rightmost column shows morphological skull stripping, "
        "which is disabled by default because it erases peripheral hemorrhage "
        "(quantified in Table T00) and is evaluated only as an ablation.")


def _fig_augmentation(cfg: Config, idx: pd.DataFrame) -> None:
    """Eight sampled augmentations of one slice, image and mask together."""
    pos = idx[idx.mask_px > 0].sort_values("mask_px", ascending=False).iloc[0]
    store = VolumeStore(cfg, [int(pos.patient)])
    img = store.slice_channels(int(pos.patient), int(pos.slice), cfg.context_slices)
    msk = store.mask(int(pos.patient), int(pos.slice))
    tfm = train_transform(cfg)
    from src.dataset import _norm_stats
    mu, sd = _norm_stats(cfg.in_chans)
    ci = cfg.context_slices

    samples = [{"image": img[:, :, ci] / 255.0, "mask": msk, "label": "original"}]
    for k in range(7):
        np.random.seed(k)
        out = tfm(image=img, mask=msk)
        ch = out["image"][ci].numpy() * sd[ci] + mu[ci]
        samples.append({"image": np.clip(ch, 0, 1), "mask": out["mask"].numpy(),
                        "label": f"augmented {k + 1}"})
    viz.fig_augmentation(samples, "F04_augmentation")
    FIG_CAPTIONS["F04_augmentation"] = (
        "Online augmentation applied during training (horizontal flip, affine "
        "scale/translate/rotate, brightness-contrast jitter, Gaussian noise and "
        "coarse dropout). The lesion mask undergoes the identical geometric "
        "transform and is outlined in green.")


def stage_train(cfg: Config, data: Dict, device: str) -> Dict:
    banner("STAGE 2/7 — training the proposed model and the baselines")
    idx, folds = data["index"], data["folds"]
    models = list(cfg.baselines) + [PROPOSED]

    results, histories, ssl_log = [], [], []
    per_slice: Dict[str, List[pd.DataFrame]] = {}
    cls_rows: Dict[str, List[pd.DataFrame]] = {}

    for fold in cfg.folds:
        tr, va, te = split_patients(folds, fold, cfg)
        log(f"fold {fold}: {len(tr)} train / {len(va)} val / {len(te)} test patients")
        store = VolumeStore(cfg, sorted(set(tr) | set(va) | set(te)))
        ssl_ckpt = prepare_ssl(cfg, store, idx, tr, fold, device, ssl_log)

        for name in models:
            r = run_experiment(name, cfg, store, idx, tr, va, te, fold, device,
                               ssl_ckpt=ssl_ckpt, reuse_existing=REUSE_TRAINED)
            results.append(r.summary)
            histories.extend(r.history)
            r.seg_rows.assign(model=name, fold=fold).to_csv(
                os.path.join(DIR_PREDS, f"{cfg.run_name}_seg_{name}_f{fold}.csv"), index=False)
            r.cls_rows.assign(model=name, fold=fold).to_csv(
                os.path.join(DIR_PREDS, f"{cfg.run_name}_cls_{name}_f{fold}.csv"), index=False)
            per_slice.setdefault(name, []).append(r.seg_rows.assign(fold=fold))
            cls_rows.setdefault(name, []).append(r.cls_rows.assign(fold=fold))
        del store

    summary = pd.DataFrame(results)
    summary["display"] = summary.tag.map(display_name)
    history = pd.DataFrame(histories)
    hist_path = os.path.join(DIR_LOGS, f"{cfg.run_name}_history.csv")
    if os.path.exists(hist_path):
        prev = pd.read_csv(hist_path)
        fresh = set(history.tag.unique()) if not history.empty else set()
        keep = prev[~prev.tag.isin(fresh)] if "tag" in prev and fresh else prev
        history = pd.concat([keep, history], ignore_index=True)
    history.to_csv(hist_path, index=False)
    if ssl_log:
        pd.DataFrame(ssl_log).to_csv(
            os.path.join(DIR_LOGS, f"{cfg.run_name}_ssl_history.csv"), index=False)

    ps = {k: pd.concat(v, ignore_index=True) for k, v in per_slice.items()}
    cr = {k: pd.concat(v, ignore_index=True) for k, v in cls_rows.items()}
    return {"summary": summary, "history": history, "per_slice": ps, "cls_rows": cr,
            "ssl_log": pd.DataFrame(ssl_log)}


def load_train_artifacts(cfg: Config, data: Dict) -> Dict:
    """Rebuild the training stage's outputs from what it wrote to disk.

    Lets the reporting, statistics and figure stages be re-run — after a plotting
    tweak, say — without spending another two hours on the GPU.
    """
    import glob

    per_slice: Dict[str, List[pd.DataFrame]] = {}
    cls_rows: Dict[str, List[pd.DataFrame]] = {}
    for path in sorted(glob.glob(os.path.join(DIR_PREDS, f"{cfg.run_name}_seg_*_f*.csv"))):
        name = os.path.basename(path)[len(f"{cfg.run_name}_seg_"):].rsplit("_f", 1)[0]
        per_slice.setdefault(name, []).append(pd.read_csv(path))
    for path in sorted(glob.glob(os.path.join(DIR_PREDS, f"{cfg.run_name}_cls_*_f*.csv"))):
        name = os.path.basename(path)[len(f"{cfg.run_name}_cls_"):].rsplit("_f", 1)[0]
        cls_rows.setdefault(name, []).append(pd.read_csv(path))
    if not per_slice:
        raise FileNotFoundError(
            f"no saved predictions for run '{cfg.run_name}' in {DIR_PREDS} — "
            "run the train stage first")

    summary_csv = os.path.join(DIR_TABLES, "T04_results_per_fold.csv")
    hist_csv = os.path.join(DIR_LOGS, f"{cfg.run_name}_history.csv")
    summary = pd.read_csv(summary_csv) if os.path.exists(summary_csv) else pd.DataFrame()
    history = pd.read_csv(hist_csv) if os.path.exists(hist_csv) else pd.DataFrame()
    if not summary.empty and "display" not in summary:
        summary["display"] = summary.tag.map(display_name)

    log(f"reloaded predictions for {len(per_slice)} models from {DIR_PREDS}")
    return {
        "summary": summary,
        "history": history,
        "per_slice": {k: pd.concat(v, ignore_index=True) for k, v in per_slice.items()},
        "cls_rows": {k: pd.concat(v, ignore_index=True) for k, v in cls_rows.items()},
        "index": data["index"],
        "spacing": data["spacing"],
    }


def stage_results(cfg: Config, train: Dict) -> Dict:
    banner("STAGE 3/7 — performance tables and figures")
    summary, history = train["summary"], train["history"]
    per_slice, cls_rows = train["per_slice"], train["cls_rows"]

    num = [c for c in summary.select_dtypes("number").columns]
    agg = summary.groupby("tag")[num].agg(["mean", "std"])
    agg.columns = [a if b == "mean" else f"{a}__cv_std" for a, b in agg.columns]
    agg = agg.reset_index()
    agg["display"] = agg.tag.map(display_name)
    assert not agg.columns.duplicated().any(), "duplicate columns in fold-averaged summary"

    report.register("T04_results_per_fold", summary,
                    "Every metric for every model on every evaluated fold.")
    report.register("T05_results_mean_over_folds", agg,
                    "Fold-averaged test metrics (mean and standard deviation).")

    ci_rows = []
    for name, df in per_slice.items():
        pos = df[df.gt_empty == 0]
        for met in ("Dice", "IoU", "Precision", "Recall", "HD95", "ASSD", "NSD"):
            if met not in pos:
                continue
            b = stats.bootstrap_ci(pos[met].values, cfg.n_bootstrap, cfg.alpha, cfg.seed)
            b.update(tag=name, display=display_name(name), metric=met,
                     n=int(pos[met].notna().sum()))
            ci_rows.append(b)
        pdice = df.groupby("patient").patient_dice.first().values
        b = stats.bootstrap_ci(pdice, cfg.n_bootstrap, cfg.alpha, cfg.seed)
        b.update(tag=name, display=display_name(name), metric="Dice_patient", n=len(pdice))
        ci_rows.append(b)
    ci = pd.DataFrame(ci_rows)
    report.register("T06_bootstrap_confidence_intervals", ci,
                    f"Percentile bootstrap 95% CIs ({cfg.n_bootstrap} resamples) of the "
                    "per-slice and per-patient metrics.")

    order = [m for m in viz.MODEL_ORDER if m in set(agg.tag)] + \
            [m for m in agg.tag if m not in viz.MODEL_ORDER]
    main = report.paper_table(agg.set_index("tag").loc[order].reset_index(), ci)
    report.register("T07_main_comparison_table", main,
                    "Camera-ready main comparison: metric with 95% bootstrap CI.")

    cls_summary, sub_rows = [], []
    for name, df in cls_rows.items():
        if df.prob.isna().all():
            continue
        th = float(summary.loc[summary.tag == name, "cls_threshold"].iloc[0]) \
            if "cls_threshold" in summary else 0.5
        m = binary_cls_metrics(df.target_ich.values, df.prob.values, th)
        m.update(tag=name, display=display_name(name))
        cls_summary.append(m)
        yt = df[[f"true_{c}" for c in MULTILABEL]].values
        yp = df[[f"prob_{c}" for c in MULTILABEL]].values
        if not np.isnan(yp).all():
            for r in multilabel_metrics(yt, yp, MULTILABEL):
                r.update(tag=name, display=display_name(name))
                sub_rows.append(r)
    cls_df = pd.DataFrame(cls_summary)
    sub_df = pd.DataFrame(sub_rows)
    report.register("T08_classification_slice_level", cls_df,
                    "Slice-level hemorrhage detection at the operating point selected "
                    "on the inner validation split.")
    report.register("T09_classification_per_subtype", sub_df,
                    "Per-class metrics of the multi-label ICH-subtype and fracture head.")

    allslices = pd.concat([d.assign(model=k) for k, d in per_slice.items()], ignore_index=True)
    report.register("T10_per_slice_metrics", allslices,
                    "Per-slice segmentation metrics for every model — the input to every "
                    "statistical test in this study.")

    viz.fig_metric_bars(agg, "F05_model_comparison", ci=ci)
    FIG_CAPTIONS["F05_model_comparison"] = (
        "Test-set Dice, IoU, HD95 and per-patient Dice for every model, with 95% "
        "bootstrap confidence intervals.")

    viz.fig_dice_distribution(per_slice, DISPLAY_NAMES, "F06_dice_distribution")
    FIG_CAPTIONS["F06_dice_distribution"] = (
        "Distribution of per-slice Dice on hemorrhagic test slices. Boxes show the "
        "quartiles, dots individual slices, and the mean is printed above each box.")

    viz.fig_patient_dice(per_slice, DISPLAY_NAMES, "F07_paired_patient_dice")
    FIG_CAPTIONS["F07_paired_patient_dice"] = (
        "Paired per-patient Dice: each line links the same patient under a baseline and "
        "under the proposed model.")

    if PROPOSED in history.tag.unique():
        viz.fig_training_curves(history[history.tag == PROPOSED].sort_values("epoch"),
                                "F08_training_curves_proposed",
                                f"{display_name(PROPOSED)} — training dynamics")
        FIG_CAPTIONS["F08_training_curves_proposed"] = (
            "Training dynamics of the proposed model: total loss, the individual loss "
            "components, and validation Dice/AUROC per epoch.")
    viz.fig_all_training_curves(history, DISPLAY_NAMES, "F09_training_curves_all")
    FIG_CAPTIONS["F09_training_curves_all"] = (
        "Validation Dice and loss per epoch for every model under an identical schedule.")

    if len(cls_rows):
        viz.fig_roc_pr(cls_rows, DISPLAY_NAMES, "F10_roc_pr")
        FIG_CAPTIONS["F10_roc_pr"] = (
            "ROC and precision–recall curves for slice-level hemorrhage detection.")
    if PROPOSED in cls_rows and not cls_rows[PROPOSED].prob.isna().all():
        d = cls_rows[PROPOSED]
        th = float(cls_df.loc[cls_df.tag == PROPOSED, "threshold"].iloc[0]) \
            if len(cls_df[cls_df.tag == PROPOSED]) else 0.5
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(d.target_ich.values, (d.prob.values >= th).astype(int),
                              labels=[0, 1])
        viz.fig_confusion(cm, "F11_confusion_matrix",
                          title=f"{display_name(PROPOSED)} — slice-level detection "
                                f"(threshold {th:.2f})")
        FIG_CAPTIONS["F11_confusion_matrix"] = (
            "Confusion matrix of the proposed model's slice-level hemorrhage detection.")
        viz.fig_calibration(d.target_ich.values, d.prob.values, "F12_calibration")
        FIG_CAPTIONS["F12_calibration"] = (
            "Reliability diagram and predicted-probability histograms by true class.")
    if len(sub_df):
        viz.fig_subtype_auc(sub_df[sub_df.tag == PROPOSED], "F13_subtype_auc")
        FIG_CAPTIONS["F13_subtype_auc"] = (
            "Per-class AUROC of the multi-label subtype/fracture head of the proposed model.")
    if len(cfg.folds) > 1:
        viz.fig_fold_variability(summary, DISPLAY_NAMES, "F14_fold_variability")
        FIG_CAPTIONS["F14_fold_variability"] = (
            "Per-fold test Dice for every model; the bar marks the across-fold mean.")

    cw_frames = []
    for met in ("Dice", "IoU"):
        t = viz_class.classwise_segmentation(per_slice, train["index"], met)
        t["display"] = t.model.map(display_name)
        cw_frames.append(t.assign(metric=met))
        viz_class.fig_classwise_metric(t, DISPLAY_NAMES, f"F18_classwise_{met.lower()}", met)
        FIG_CAPTIONS[f"F18_classwise_{met.lower()}"] = (
            f"Mean per-slice {met} on hemorrhagic test slices, broken down by ICH "
            f"subtype. Slice counts per subtype are printed on the axis; the rarer "
            f"subtypes carry few test slices, so their bars are correspondingly "
            f"noisy and should be read alongside Table T07b.")
    classwise = pd.concat(cw_frames, ignore_index=True)
    report.register("T07b_classwise_segmentation", classwise,
                    "Segmentation performance per ICH subtype for every model. "
                    "'n_slices' is the number of hemorrhagic test slices carrying "
                    "that subtype label.")

    return {"agg": agg, "ci": ci, "cls_df": cls_df, "sub_df": sub_df, "main": main,
            "classwise": classwise}


def stage_stats(cfg: Config, train: Dict, results: Dict) -> Dict:
    banner("STAGE 4/7 — statistical analysis")
    per_slice, cls_rows = train["per_slice"], train["cls_rows"]
    summary = train["summary"]

    norm_rows = []
    for name, df in per_slice.items():
        pos = df[df.gt_empty == 0]
        for met in ("Dice", "IoU", "HD95"):
            if met in pos:
                r = stats.normality(pos[met].values, met)
                r.update(tag=name, display=display_name(name))
                norm_rows.append(r)
    report.register("T11_normality_tests", pd.DataFrame(norm_rows),
                    "Shapiro-Wilk normality screen on the per-slice metric "
                    "distributions; non-normality motivates the non-parametric tests.")

    tests = stats.compare_against_reference(
        per_slice, PROPOSED, metrics=("Dice", "IoU", "HD95", "ASSD", "NSD", "Precision", "Recall"),
        alpha=cfg.alpha, n_boot=cfg.n_bootstrap)
    tests["display_b"] = tests.model_b.map(display_name)
    report.register("T12_paired_tests_per_slice", tests,
                    "Wilcoxon signed-rank tests of the proposed model against every "
                    "baseline on paired per-slice metrics, Holm-corrected within each "
                    "metric family, with bootstrap CI of the difference and effect sizes.")

    pat = {k: (df.groupby("patient", as_index=False)
                 .agg(patient_dice=("patient_dice", "first"))
                 .assign(gt_empty=0, slice=0, Dice=lambda d: d.patient_dice))
           for k, df in per_slice.items()}
    tests_pat = stats.compare_against_reference(pat, PROPOSED, metrics=("Dice",),
                                               key=("patient",), alpha=cfg.alpha,
                                               n_boot=cfg.n_bootstrap)
    if not tests_pat.empty:
        tests_pat["display_b"] = tests_pat.model_b.map(display_name)
        report.register("T13_paired_tests_per_patient", tests_pat,
                        "The same paired analysis at patient level on volumetric Dice.")

    fried = None
    aligned = _align_metric(per_slice, "Dice")
    if aligned is not None and aligned.shape[1] > 2:
        omnibus, ranks, nem = stats.friedman_nemenyi(aligned)
        ranks["display"] = ranks.model.map(display_name)
        report.register("T14_friedman_ranks",
                        ranks.assign(**{k: v for k, v in omnibus.items()}),
                        "Friedman omnibus test over all models with mean ranks of "
                        "per-slice Dice.")
        nem["display_a"] = nem.model_a.map(display_name)
        nem["display_b"] = nem.model_b.map(display_name)
        report.register("T15_nemenyi_posthoc", nem,
                        "Nemenyi post-hoc pairwise rank differences against the critical "
                        "difference.")
        cd = float(nem.critical_difference.iloc[0]) if len(nem) else 0.0
        viz.fig_critical_difference(ranks, cd, DISPLAY_NAMES, "F15_critical_difference",
                                    omnibus)
        FIG_CAPTIONS["F15_critical_difference"] = (
            "Critical-difference diagram (Friedman test with Nemenyi post-hoc) on "
            "per-slice Dice; models joined by a bar are not significantly different.")
        fried = omnibus

    delong_rows, mcnemar_rows = [], []
    if PROPOSED in cls_rows and not cls_rows[PROPOSED].prob.isna().all():
        ref = cls_rows[PROPOSED].set_index(["patient", "slice"]).sort_index()
        th_ref = _cls_threshold(summary, PROPOSED)
        for name, df in cls_rows.items():
            if name == PROPOSED or df.prob.isna().all():
                continue
            oth = df.set_index(["patient", "slice"]).sort_index()
            common = ref.index.intersection(oth.index)
            y = ref.loc[common, "target_ich"].values
            d = stats.delong_test(y, ref.loc[common, "prob"].values,
                                  oth.loc[common, "prob"].values)
            d.update(model_a=PROPOSED, model_b=name, display_b=display_name(name),
                     n=len(common))
            delong_rows.append(d)

            m = stats.mcnemar_test(
                y,
                (ref.loc[common, "prob"].values >= th_ref).astype(int),
                (oth.loc[common, "prob"].values >= _cls_threshold(summary, name)).astype(int))
            m.update(model_a=PROPOSED, model_b=name, display_b=display_name(name),
                     n=len(common))
            mcnemar_rows.append(m)

    if delong_rows:
        dl = pd.DataFrame(delong_rows)
        adj, _ = stats.holm_correction(dl.p_delong.fillna(1).values, cfg.alpha)
        dl["p_holm"] = adj
        dl["signif_label"] = dl.p_holm.map(stats.stars)
        report.register("T16_delong_auc_tests", dl,
                        "DeLong tests for correlated ROC curves: proposed vs each "
                        "baseline on slice-level detection AUC, Holm-corrected.")
    if mcnemar_rows:
        mc = pd.DataFrame(mcnemar_rows)
        adj, _ = stats.holm_correction(mc.p_mcnemar.fillna(1).values, cfg.alpha)
        mc["p_holm"] = adj
        mc["signif_label"] = mc.p_holm.map(stats.stars)
        report.register("T17_mcnemar_tests", mc,
                        "McNemar tests on paired slice-level detection decisions.")

    spacing = (train["index"].groupby("patient").spacing_mm.first()
               if "index" in train else None)
    vol_rows = []
    for name, df in per_slice.items():
        g = df.groupby("patient").agg(gt_vol=("gt_px", "sum"), pred_vol=("pred_px", "sum"))
        if spacing is not None:
            sp = g.index.map(spacing).to_numpy(dtype=float)
        else:
            sp = np.ones(len(g))
        ml = (sp ** 2) * 5.0 / 1000.0
        true_v = g["gt_vol"].to_numpy(dtype=float) * ml
        pred_v = g["pred_vol"].to_numpy(dtype=float) * ml
        ba = stats.bland_altman(pred_v, true_v)
        ba.update(stats.icc21(true_v, pred_v))
        ba.update(tag=name, display=display_name(name))
        vol_rows.append(ba)
        if name == PROPOSED:
            viz.fig_bland_altman(true_v, pred_v, ba, "F16_bland_altman")
            FIG_CAPTIONS["F16_bland_altman"] = (
                "Agreement between predicted and reference hemorrhage volume per patient: "
                "scatter with the identity line and ICC, and the Bland–Altman plot with "
                "bias and 95% limits of agreement.")
    report.register("T18_volume_agreement", pd.DataFrame(vol_rows),
                    "Per-patient hemorrhage-volume agreement: ICC(2,1), Pearson r, "
                    "Bland–Altman bias and limits of agreement.")

    if not tests.empty:
        viz.fig_pvalue_heatmap(tests, DISPLAY_NAMES, "F17_significance", "Dice")
        FIG_CAPTIONS["F17_significance"] = (
            "Holm-adjusted significance of the proposed model over each baseline on "
            "per-slice Dice; the dashed line marks alpha = 0.05.")

    log(f"statistics complete — {len(tests)} paired comparisons")
    return {"tests": tests, "tests_patient": tests_pat, "friedman": fried}


def _align_metric(per_slice: Dict[str, pd.DataFrame], metric: str) -> Optional[pd.DataFrame]:
    """(slices x models) table aligned on (patient, slice), positive slices only."""
    frames = []
    for name, df in per_slice.items():
        d = df[df.gt_empty == 0][["patient", "slice", metric]].copy()
        d = d.rename(columns={metric: name}).set_index(["patient", "slice"])
        frames.append(d)
    if not frames:
        return None
    out = frames[0]
    for f in frames[1:]:
        out = out.join(f, how="inner")
    return out.dropna()


def _cls_threshold(summary: pd.DataFrame, tag: str) -> float:
    if "cls_threshold" in summary.columns:
        s = summary.loc[summary.tag == tag, "cls_threshold"]
        if len(s) and np.isfinite(s.iloc[0]):
            return float(s.iloc[0])
    return 0.5


def stage_ablation(cfg: Config, data: Dict, device: str,
                   ablation_epochs: Optional[int] = None) -> Dict:
    banner("STAGE 5/7 — ablation study")
    idx, folds = data["index"], data["folds"]
    rows, hist, ssl_hist = [], [], []
    per_slice: Dict[str, pd.DataFrame] = {}
    base_store: Optional[VolumeStore] = None

    for fold in cfg.ablation_folds:
        tr, va, te = split_patients(folds, fold, cfg)
        patients = sorted(set(tr) | set(va) | set(te))
        base_store = VolumeStore(cfg, patients)

        for spec in ABLATIONS:
            acfg = cfg.clone(**dict(spec["kw"]))
            if ablation_epochs:
                acfg = acfg.clone(epochs=ablation_epochs)

            if acfg.cache_dir != cfg.cache_dir:
                build_cache(acfg)
                store = VolumeStore(acfg, patients)
            else:
                store = base_store
            ssl_tag = (f"{fold}_{spec['tag']}"
                       if acfg.in_chans != cfg.in_chans or acfg.cache_dir != cfg.cache_dir
                       else fold)
            ssl_ckpt = prepare_ssl(acfg, store, idx, tr, ssl_tag, device, ssl_hist)
            log(f"ablation {spec['tag']}: {spec['label']}")
            r = run_experiment(PROPOSED, acfg, store, idx, tr, va, te, fold, device,
                               tag=spec["tag"], ssl_ckpt=ssl_ckpt)
            s = dict(r.summary)
            s.update(tag=spec["tag"], label=spec["label"], fold=fold)
            rows.append(s)
            hist.extend(r.history)
            prev = per_slice.get(spec["tag"])
            per_slice[spec["tag"]] = (pd.concat([prev, r.seg_rows], ignore_index=True)
                                      if prev is not None else r.seg_rows)
            r.seg_rows.assign(variant=spec["tag"], fold=fold).to_csv(
                os.path.join(DIR_PREDS, f"{cfg.run_name}_abl_{spec['tag']}_f{fold}.csv"),
                index=False)
            if store is not base_store:
                del store
        del base_store
        base_store = None

    abl = pd.DataFrame(rows)
    if abl.empty:
        return {"ablation": abl}
    full = abl[abl.tag == "A0_full"]
    base = float(full.Dice.mean()) if len(full) else float(abl.Dice.max())
    abl["delta_Dice"] = abl.Dice - base
    abl["delta_pct"] = abl.delta_Dice / max(base, 1e-9) * 100

    if "A0_full" in per_slice:
        t = stats.compare_against_reference(per_slice, "A0_full", metrics=("Dice", "IoU", "HD95"),
                                           alpha=cfg.alpha, n_boot=cfg.n_bootstrap)
        lab = {s["tag"]: s["label"] for s in ABLATIONS}
        t["variant_label"] = t.model_b.map(lab)
        report.register("T20_ablation_significance", t,
                        "Paired Wilcoxon tests of each ablated variant against the full "
                        "proposed model on the same test slices.")

    report.register("T19_ablation_study",
                    abl[[c for c in ("tag", "label", "Dice", "Dice_std", "IoU", "Precision",
                                     "Recall", "HD95", "ASSD", "NSD", "Dice_patient",
                                     "cls_AUROC", "cls_F1", "delta_Dice", "delta_pct",
                                     "params_M", "threshold", "best_epoch", "train_minutes")
                         if c in abl.columns]],
                    "Component ablation: one architectural or methodological element "
                    "removed per row, everything else held fixed. All variants share "
                    "one schedule (see epochs_run), so rows are comparable to each "
                    "other and to A0_full — but not to the main comparison table, "
                    "which trains longer.")
    pd.DataFrame(hist).to_csv(
        os.path.join(DIR_LOGS, f"{cfg.run_name}_ablation_history.csv"), index=False)
    if ssl_hist:
        pd.DataFrame(ssl_hist).to_csv(
            os.path.join(DIR_LOGS, f"{cfg.run_name}_ablation_ssl_history.csv"), index=False)

    viz.fig_ablation(abl, "F18_ablation", "Dice")
    FIG_CAPTIONS["F18_ablation"] = (
        "Component ablation. Left: test Dice per variant (the full model in accent "
        "colour, dashed line at its score). Right: change in Dice caused by removing "
        "each component.")
    return {"ablation": abl, "ablation_per_slice": per_slice}


def stage_qualitative(cfg: Config, data: Dict, train: Dict, device: str) -> Dict:
    banner("STAGE 6/7 — qualitative results and threshold sensitivity")
    idx, folds = data["index"], data["folds"]
    fold = cfg.folds[0]
    tr, va, te = split_patients(folds, fold, cfg)
    store = VolumeStore(cfg, te)
    te_df = idx[idx.patient.isin(te)].reset_index(drop=True)
    from torch.utils.data import DataLoader
    loader = DataLoader(ICHDataset(store, te_df, cfg, eval_transform(cfg)),
                        batch_size=cfg.eval_batch_size, shuffle=False,
                        num_workers=cfg.num_workers)

    models = list(cfg.baselines) + [PROPOSED]
    probs: Dict[str, np.ndarray] = {}
    curves: Dict[str, pd.DataFrame] = {}
    sens_rows = []

    for name in models:
        ck = os.path.join(DIR_MODELS, f"{cfg.run_name}_{name}_f{fold}.pth")
        if not os.path.exists(ck):
            continue
        m = build_model(name, cfg).to(device)
        m.load_state_dict(torch.load(ck, map_location=device))
        p = collect_probabilities(m, loader, device, cfg, cfg.tta)
        probs[name] = p["seg_prob"]
        masks = p["mask"]
        grid = np.linspace(0.05, 0.95, 19)
        cv = [{"threshold": float(t),
               "dice": dice_at_threshold(p["seg_prob"], masks, float(t))} for t in grid]
        curves[name] = pd.DataFrame(cv)
        for row in cv:
            sens_rows.append({"tag": name, "display": display_name(name), **row})
        del m
        torch.cuda.empty_cache()

    if sens_rows:
        report.register("T21_threshold_sensitivity", pd.DataFrame(sens_rows),
                        "Aggregated test Dice as a function of the binarisation "
                        "threshold, per model.")
        viz.fig_threshold_sensitivity(curves, DISPLAY_NAMES, "F19_threshold_sensitivity")
        FIG_CAPTIONS["F19_threshold_sensitivity"] = (
            "Sensitivity of aggregated test Dice to the probability threshold; the marker "
            "shows each model's best value.")

    if PROPOSED in probs:
        summ = train["summary"]
        model_th = {r.tag: float(r.threshold) for _, r in
                    summ[summ.fold == fold].iterrows() if pd.notna(r.threshold)}
        th = model_th.get(PROPOSED, 0.5)
        rows = train["per_slice"][PROPOSED]
        rows = rows[(rows.fold == fold) & (rows.gt_empty == 0)]
        picks = _diverse_picks(rows, n_best=3, n_worst=2)
        samples, gallery = [], []
        pos_idx = {(int(r.patient), int(r.slice)): i for i, r in te_df.iterrows()}
        mean, sd = _norm(cfg)
        ds = ICHDataset(store, te_df, cfg, eval_transform(cfg))

        for _, r in picks.iterrows():
            key = (int(r.patient), int(r.slice))
            if key not in pos_idx:
                continue
            i = pos_idx[key]
            x, mk, _, _ = ds[i]
            ct = np.clip(x[cfg.context_slices].numpy() * sd + mean, 0, 1)
            gt = mk.squeeze().numpy()
            pr = remove_small_components(probs[PROPOSED][i].astype(np.float32) >= th, 0)
            samples.append({"ct": ct, "gt": gt, "pred": pr.astype(float),
                            "patient": key[0], "slice": key[1], "dice": float(r.Dice)})
            gp, gd = {}, {}
            for name in probs:
                b = probs[name][i].astype(np.float32) >= model_th.get(name, th)
                gp[name] = b.astype(float)
                inter = float(np.sum(b & (gt > 0.5)))
                gd[name] = 2 * inter / (float(b.sum()) + float((gt > 0.5).sum()) + 1e-7)
            gallery.append({"ct": ct, "gt": gt, "preds": gp, "dice": gd,
                            "patient": key[0], "slice": key[1]})

        if samples:
            viz.fig_qualitative(samples, "F20_qualitative_proposed",
                                f"{display_name(PROPOSED)} — best (top) and worst "
                                "(bottom) test slices")
            FIG_CAPTIONS["F20_qualitative_proposed"] = (
                "Qualitative segmentation by the proposed model on its three best and "
                "two worst hemorrhagic test slices, with true positives, false "
                "positives and false negatives colour-coded.")
        if gallery:
            viz.fig_model_gallery(gallery, DISPLAY_NAMES, "F21_qualitative_all_models")
            FIG_CAPTIONS["F21_qualitative_all_models"] = (
                "The same test slices segmented by every model, each binarised at its "
                "own validation-tuned threshold, with per-slice Dice printed under "
                "each panel.")

        cw = []
        lab = te_df[["patient", "slice", "mask_px"]].merge(
            idx[["patient", "slice"] + [c for c in MULTILABEL + ["target_ich"]
                                        if c in idx.columns]],
            on=["patient", "slice"], how="left")
        used = set()
        for cls in viz_class.CLASSES:
            sub = lab[viz_class.class_mask(lab, cls)]
            if sub.empty:
                continue
            sub = sub.sort_values("mask_px", ascending=(cls == "No_Hemorrhage"))
            pick = next((r for _, r in sub.iterrows()
                         if (int(r.patient), int(r["slice"])) in pos_idx
                         and (int(r.patient), int(r["slice"])) not in used), None)
            if pick is None:
                continue
            key = (int(pick.patient), int(pick["slice"]))
            used.add(key)
            i = pos_idx[key]
            x, mk, _, _ = ds[i]
            gt = mk.squeeze().numpy()
            pr = (probs[PROPOSED][i].astype(np.float32) >= th)
            inter = float(np.sum(pr & (gt > 0.5)))
            cw.append({"ct": np.clip(x[cfg.context_slices].numpy() * sd + mean, 0, 1),
                       "gt": gt, "pred": pr.astype(float),
                       "patient": key[0], "slice": key[1],
                       "dice": 2 * inter / (float(pr.sum()) + float((gt > 0.5).sum()) + 1e-7),
                       "label": viz_class.PRETTY[cls]})
        if cw:
            viz.fig_qualitative(cw, "F25_qualitative_by_class",
                                f"{display_name(PROPOSED)} — one test slice per class")
            FIG_CAPTIONS["F25_qualitative_by_class"] = (
                "Segmentation by the proposed model on one representative test slice "
                "from each class, with true positives, false positives and false "
                "negatives colour-coded. The hemorrhage-free row shows the model's "
                "false-positive behaviour on a normal slice, where any coloured pixel "
                "is an error.")
    return {"probs_available": list(probs), "curves": curves, "store": store,
            "te_df": te_df, "fold": fold}


def _norm(cfg: Config):
    from src.dataset import _norm_stats
    mu, sd = _norm_stats(cfg.in_chans)
    return mu[cfg.context_slices], sd[cfg.context_slices]


def _diverse_picks(rows: pd.DataFrame, n_best: int, n_worst: int = 0,
                   col: str = "Dice") -> pd.DataFrame:
    """Best/worst slices drawn from *distinct patients*.

    Without the de-duplication the panels fill up with consecutive slices of one
    patient, which looks like four examples but is really one.
    """
    if rows.empty:
        return rows
    ranked = rows.sort_values(col, ascending=False)
    best = ranked.drop_duplicates("patient").head(n_best)
    if n_worst <= 0:
        return best
    worst = (ranked[~ranked.index.isin(best.index)]
             .sort_values(col)
             .drop_duplicates("patient")
             .head(n_worst))
    return pd.concat([best, worst])


def stage_xai(cfg: Config, data: Dict, train: Dict, qual: Dict, device: str) -> Dict:
    banner("STAGE 7/7 — explainability (Grad-CAM / Grad-CAM++ / LayerCAM)")
    fold = qual["fold"]
    store, te_df = qual["store"], qual["te_df"]
    ds = ICHDataset(store, te_df, cfg, eval_transform(cfg))
    mean, sd = _norm(cfg)

    rows_all = train["per_slice"][PROPOSED]
    rows_all = rows_all[(rows_all.fold == fold) & (rows_all.gt_empty == 0)]
    picks = _diverse_picks(rows_all, n_best=4)
    pos_idx = {(int(r.patient), int(r.slice)): i for i, r in te_df.iterrows()}

    samples = []
    for _, r in picks.iterrows():
        key = (int(r.patient), int(r.slice))
        if key not in pos_idx:
            continue
        x, mk, _, _ = ds[pos_idx[key]]
        samples.append({"x": x, "gt": mk.squeeze().numpy(),
                        "ct": np.clip(x[cfg.context_slices].numpy() * sd + mean, 0, 1),
                        "patient": key[0], "slice": key[1]})
    if not samples:
        log("no positive test slices available for XAI")
        return {}

    cam_tables, cam_records = [], []
    for name in (PROPOSED, "hemoclr_net"):
        ck = os.path.join(DIR_MODELS, f"{cfg.run_name}_{name}_f{fold}.pth")
        if not os.path.exists(ck):
            continue
        m = build_model(name, cfg).to(device)
        m.load_state_dict(torch.load(ck, map_location=device))
        recs, rows = explain_samples(m, name, samples, device)
        cam_records.append((name, recs))
        cam_tables.extend(rows)
        del m
        torch.cuda.empty_cache()

    xai_df = pd.DataFrame(cam_tables)
    if not xai_df.empty:
        report.register("T22_xai_localisation", xai_df,
                        "Quantitative explainability: pointing-game hit rate, share of "
                        "CAM energy inside the lesion, CAM-lesion IoU/Dice, "
                        "saliency ratio, and deletion AUC.")
        num = xai_df.select_dtypes("number").columns
        summ = (xai_df.groupby(["model", "method", "layer"])[list(num)]
                .mean().reset_index())
        report.register("T23_xai_summary", summ,
                        "Explainability metrics averaged over the analysed slices, by "
                        "CAM method and target layer.")

    for name, recs in cam_records:
        if recs:
            _fig_cams(recs, name, cfg)

    _fig_cams_by_class(cfg, data, te_df, ds, pos_idx, fold, device)
    return {"xai": xai_df}


def _fig_cams_by_class(cfg: Config, data: Dict, te_df: pd.DataFrame, ds, pos_idx: Dict,
                       fold: int, device: str) -> None:
    """Grad-CAM and Grad-CAM++ for one representative test slice of each class.

    The class vocabulary is the same seven radiologist labels used everywhere
    else, so a reader can trace one subtype from the example gallery through the
    result tables to its explanation.
    """
    from src.xai import CAM, pick_layers, seg_score

    ck = os.path.join(DIR_MODELS, f"{cfg.run_name}_{PROPOSED}_f{fold}.pth")
    if not os.path.exists(ck):
        log("proposed checkpoint missing — skipping class-wise CAMs")
        return

    mean, sd = _norm(cfg)
    label_cols = [c for c in MULTILABEL + ["target_ich"] if c in data["index"].columns]
    cand_all = te_df[["patient", "slice", "mask_px"]].merge(
        data["index"][["patient", "slice"] + label_cols], on=["patient", "slice"], how="left")

    chosen: List[Tuple[str, int, int]] = []
    used: set = set()
    for cls in viz_class.CLASSES:
        sub = cand_all[viz_class.class_mask(cand_all, cls)].copy()
        if sub.empty:
            continue
        if cls == "No_Hemorrhage":
            n = sub.patient.map(data["index"].groupby("patient").n_slices.first())
            sub = sub.assign(_mid=(sub["slice"] - n / 2.0).abs()).sort_values("_mid")
        else:
            sub = sub.sort_values("mask_px", ascending=False)
        pick = next((r for _, r in sub.iterrows()
                     if (int(r.patient), int(r["slice"])) not in used), None)
        if pick is None:
            continue
        key = (int(pick.patient), int(pick["slice"]))
        used.add(key)
        chosen.append((cls, key[0], key[1]))

    if not chosen:
        log("no class representatives in the test fold — skipping class-wise CAMs")
        return

    model = build_model(PROPOSED, cfg).to(device)
    model.load_state_dict(torch.load(ck, map_location=device))
    model.eval()
    cands = pick_layers(model, PROPOSED, 3)
    layer_path, layer_mod, layer_lbl = cands[1] if len(cands) > 1 else cands[0]

    entries = []
    for cls, pid, sl in chosen:
        key = (pid, sl)
        if key not in pos_idx:
            continue
        x, mk, _, _ = ds[pos_idx[key]]
        xb = x.unsqueeze(0).to(device)
        cams = {}
        for meth, fn in (("Grad-CAM", "gradcam"), ("Grad-CAM++", "gradcam_pp")):
            try:
                with CAM(model, layer_mod) as eng:
                    cams[meth] = getattr(eng, fn)(xb.clone().requires_grad_(True),
                                                  seg_score(use_pred=True))[0]
            except Exception as exc:
                log(f"CAM failed for {cls}: {str(exc)[:80]}")
        if len(cams) == 2:
            entries.append({"cls": cls, "patient": pid, "slice": sl,
                            "ct": np.clip(x[cfg.context_slices].numpy() * sd + mean, 0, 1),
                            "gt": mk.squeeze().numpy(), "cams": cams})

    if entries:
        viz_class.fig_cam_by_class(
            entries, "F24_gradcam_by_class",
            f"Grad-CAM and Grad-CAM++ by class — {display_name(PROPOSED)} "
            f"(target layer: {layer_lbl})")
        FIG_CAPTIONS["F24_gradcam_by_class"] = (
            f"Class-wise explainability for the proposed model. Each row is one "
            f"representative test slice of a class: the CT slice, the reference "
            f"lesion (green), then Grad-CAM and Grad-CAM++ as a raw heatmap and "
            f"overlaid on the anatomy, both computed at the {layer_lbl} layer "
            f"against the mean predicted-region logit. The white contour on the "
            f"overlays marks the reference lesion, so agreement between heat and "
            f"contour can be judged directly. Quantitative versions of that "
            f"agreement are in Tables T22-T23.")
    del model
    torch.cuda.empty_cache()


def _fig_cams(records: List[Dict], model_name: str, cfg: Config) -> None:
    """One figure per CAM method plus a method-comparison figure."""
    import matplotlib.pyplot as plt

    viz.apply_style()
    layers = sorted({r["layer"] for r in records},
                    key=lambda l: [r["layer"] for r in records].index(l))
    methods = sorted({r["method"] for r in records},
                     key=lambda m: [r["method"] for r in records].index(m))
    slices = sorted({(r["patient"], r["slice"]) for r in records})

    pretty = {"gradcam": "Grad-CAM", "gradcam_pp": "Grad-CAM++",
              "layercam": "LayerCAM", "eigencam": "Eigen-CAM"}

    for meth in methods:
        sel = [r for r in records if r["method"] == meth]
        if not sel:
            continue
        ncol = 2 + len(layers)
        fig, ax = plt.subplots(len(slices), ncol,
                              figsize=(2.4 * ncol, 2.6 * len(slices)), squeeze=False)
        for ri, key in enumerate(slices):
            base = next((r for r in sel if (r["patient"], r["slice"]) == key), None)
            if base is None:
                continue
            ct, gt = base["ct"], base["gt"] > 0.5
            ax[ri][0].imshow(ct, cmap="gray", vmin=0, vmax=1)
            ax[ri][1].imshow(ct, cmap="gray", vmin=0, vmax=1)
            ov = np.zeros((*gt.shape, 4)); ov[gt] = [0.0, 0.51, 0.0, 0.55]
            ax[ri][1].imshow(ov)
            for ci, lay in enumerate(layers):
                r = next((q for q in sel if (q["patient"], q["slice"]) == key
                          and q["layer"] == lay), None)
                a = ax[ri][2 + ci]
                a.imshow(ct, cmap="gray", vmin=0, vmax=1)
                if r is not None:
                    a.imshow(r["cam"], cmap="jet", alpha=0.5, vmin=0, vmax=1)
                    a.contour(gt.astype(float), levels=[0.5], colors="#00ff9d",
                              linewidths=1.2)
            titles = ["CT", "Ground truth"] + [f"{pretty.get(meth,meth)}\n{l}" for l in layers]
            for ci in range(ncol):
                if ri == 0:
                    ax[ri][ci].set_title(titles[ci], fontsize=9, fontweight="bold")
                ax[ri][ci].set_xticks([]); ax[ri][ci].set_yticks([])
                for s in ax[ri][ci].spines.values():
                    s.set_visible(False)
            ax[ri][0].set_ylabel(f"pt {key[0]}\nsl {key[1]}", fontsize=8.5)
        stem = f"F22_xai_{meth}_{model_name}"
        fig.suptitle(f"{pretty.get(meth, meth)} — {display_name(model_name)} "
                     "(green contour = reference lesion)",
                     fontsize=12, fontweight="bold")
        viz.save(fig, stem)
        FIG_CAPTIONS[stem] = (
            f"{pretty.get(meth, meth)} saliency for {display_name(model_name)} at "
            "three network depths; the green contour marks the reference lesion.")

    lay = layers[-1]
    fig, ax = plt.subplots(len(slices), 2 + len(methods),
                           figsize=(2.4 * (2 + len(methods)), 2.6 * len(slices)),
                           squeeze=False)
    for ri, key in enumerate(slices):
        base = next((r for r in records if (r["patient"], r["slice"]) == key), None)
        if base is None:
            continue
        ct, gt = base["ct"], base["gt"] > 0.5
        ax[ri][0].imshow(ct, cmap="gray", vmin=0, vmax=1)
        ax[ri][1].imshow(ct, cmap="gray", vmin=0, vmax=1)
        ov = np.zeros((*gt.shape, 4)); ov[gt] = [0.0, 0.51, 0.0, 0.55]
        ax[ri][1].imshow(ov)
        for ci, meth in enumerate(methods):
            r = next((q for q in records if (q["patient"], q["slice"]) == key
                      and q["method"] == meth and q["layer"] == lay), None)
            a = ax[ri][2 + ci]
            a.imshow(ct, cmap="gray", vmin=0, vmax=1)
            if r is not None:
                a.imshow(r["cam"], cmap="jet", alpha=0.5, vmin=0, vmax=1)
                a.contour(gt.astype(float), levels=[0.5], colors="#00ff9d", linewidths=1.2)
        titles = ["CT", "Ground truth"] + [pretty.get(m, m) for m in methods]
        for ci in range(2 + len(methods)):
            if ri == 0:
                ax[ri][ci].set_title(titles[ci], fontsize=9, fontweight="bold")
            ax[ri][ci].set_xticks([]); ax[ri][ci].set_yticks([])
            for s in ax[ri][ci].spines.values():
                s.set_visible(False)
        ax[ri][0].set_ylabel(f"pt {key[0]}\nsl {key[1]}", fontsize=8.5)
    stem = f"F23_xai_method_comparison_{model_name}"
    fig.suptitle(f"CAM method comparison at {lay} — {display_name(model_name)}",
                 fontsize=12, fontweight="bold")
    viz.save(fig, stem)
    FIG_CAPTIONS[stem] = (
        f"Grad-CAM, Grad-CAM++ and LayerCAM compared at the {lay} layer of "
        f"{display_name(model_name)}.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="paper", choices=sorted(PRESETS))
    ap.add_argument("--stages", default="data,train,results,stats,qualitative,xai",
                    help="comma-separated subset of stages to run "
                         "(add 'ablation' explicitly if you want it)")
    ap.add_argument("--folds", default=None, help="override folds, e.g. 0,1,2")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--ablation-epochs", type=int, default=None)
    ap.add_argument("--img-size", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--no-ssl", action="store_true")
    ap.add_argument("--no-ablation", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--models", default=None,
                    help="override the baseline list, e.g. unet,unetpp "
                         "(the proposed model is always included)")
    ap.add_argument("--reuse-trained", action="store_true",
                    help="skip training for models that already have a checkpoint "
                         "for this run/fold and only re-evaluate them")
    args = ap.parse_args()

    global REUSE_TRAINED
    REUSE_TRAINED = args.reuse_trained

    cfg = PRESETS[args.preset]()
    if args.folds:
        cfg = cfg.clone(folds=[int(x) for x in args.folds.split(",")])
    if args.epochs:
        cfg = cfg.clone(epochs=args.epochs)
    if args.img_size:
        cfg = cfg.clone(img_size=args.img_size)
    if args.batch_size:
        cfg = cfg.clone(batch_size=args.batch_size)
    if args.workers is not None:
        cfg = cfg.clone(num_workers=args.workers)
    if args.no_ssl:
        cfg = cfg.clone(ssl_enabled=False)
    if args.no_ablation:
        cfg = cfg.clone(run_ablation=False)
    if args.models:
        cfg = cfg.clone(baselines=[m.strip() for m in args.models.split(",") if m.strip()])

    stages = {s.strip() for s in args.stages.split(",")}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    banner(f"ICH hybrid study — preset '{cfg.run_name}' on {device}")
    log(f"{cfg.img_size}px · {cfg.in_chans} input channels · {cfg.epochs} epochs · "
        f"folds {cfg.folds} · batch {cfg.batch_size}")
    cfg.save(os.path.join(OUT_ROOT, f"config_{cfg.run_name}.json"))
    t0 = time.time()

    data = stage_data(cfg)
    data["spacing"] = float(data["index"].spacing_mm.mean())

    train = results = st = abl = qual = {}
    if "train" in stages:
        train = stage_train(cfg, data, device)
        train["index"] = data["index"]
        train["spacing"] = data["spacing"]
    elif stages & {"results", "stats", "qualitative", "xai"}:
        train = load_train_artifacts(cfg, data)
    if "results" in stages and train:
        results = stage_results(cfg, train)
    if "stats" in stages and train:
        st = stage_stats(cfg, train, results)
    if "ablation" in stages and cfg.run_ablation:
        abl = stage_ablation(cfg, data, device, args.ablation_epochs)
    if "qualitative" in stages and train:
        qual = stage_qualitative(cfg, data, train, device)
    if "xai" in stages and train and qual:
        stage_xai(cfg, data, train, qual, device)

    banner("EXPORT — tables, workbook and summary")
    report.register("T24_figure_manifest", report.figure_manifest(FIG_CAPTIONS),
                    "Every figure produced, with its caption and file paths.")
    report.register("T25_run_configuration",
                    pd.DataFrame([{"key": k, "value": str(v)}
                                  for k, v in json.loads(
                                      open(os.path.join(OUT_ROOT,
                                           f"config_{cfg.run_name}.json")).read()).items()]),
                    "Exact configuration of this run, for reproducibility.")

    adopted = report.adopt_existing_tables()
    if adopted:
        log(f"adopted {adopted} table(s) from earlier stage runs")
    xlsx = report.write_workbook(os.path.join(OUT_ROOT,
                                              f"ICH_Hybrid_Results_{cfg.run_name}.xlsx"))
    md = report.write_markdown_summary(
        os.path.join(OUT_ROOT, f"RESULTS_{cfg.run_name}.md"), cfg,
        results.get("main", pd.DataFrame()), st.get("tests"),
        abl.get("ablation"), FIG_CAPTIONS)

    log(f"workbook  -> {xlsx}")
    log(f"summary   -> {md}")
    log(f"tables    -> {DIR_TABLES} ({len(report.registered())} CSVs)")
    log(f"figures   -> {DIR_FIGURES}")
    log(f"total elapsed {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
