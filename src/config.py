"""Central configuration for the ICH hybrid segmentation + classification study.

Every experiment (proposed model, baselines, ablations) is driven from a single
`Config` object so that a full-scale rerun is a one-line change.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(
    os.path.dirname(PROJECT_ROOT),
    "computed-tomography-images-for-intracranial-hemorrhage-detection-and-segmentation-1.3.1",
)
CT_DIR = os.path.join(DATA_ROOT, "ct_scans")
MASK_DIR = os.path.join(DATA_ROOT, "masks")
SLICE_LABEL_CSV = os.path.join(DATA_ROOT, "hemorrhage_diagnosis_raw_ct.csv")
DEMOGRAPHICS_CSV = os.path.join(DATA_ROOT, "Patient_demographics.csv")

OUT_ROOT = os.path.join(PROJECT_ROOT, "outputs")
DIR_TABLES = os.path.join(OUT_ROOT, "tables")
DIR_FIGURES = os.path.join(OUT_ROOT, "figures")
DIR_LOGS = os.path.join(OUT_ROOT, "logs")
DIR_MODELS = os.path.join(OUT_ROOT, "models")
DIR_PREDS = os.path.join(OUT_ROOT, "predictions")
DIR_CACHE = os.path.join(OUT_ROOT, "cache")

for _d in (OUT_ROOT, DIR_TABLES, DIR_FIGURES, DIR_LOGS, DIR_MODELS, DIR_PREDS, DIR_CACHE):
    os.makedirs(_d, exist_ok=True)

SUBTYPES: List[str] = [
    "Intraventricular",
    "Intraparenchymal",
    "Subarachnoid",
    "Epidural",
    "Subdural",
]
MULTILABEL: List[str] = SUBTYPES + ["Fracture"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass
class Config:
    run_name: str = "quick"
    seed: int = 42

    img_size: int = 256
    windows: tuple = ((40, 80), (80, 200), (600, 2800))
    skull_strip: bool = False
    clahe: bool = True
    context_slices: int = 1

    n_folds: int = 5
    folds: List[int] = field(default_factory=lambda: [0])
    inner_val_frac: float = 0.18

    epochs: int = 12
    batch_size: int = 8
    eval_batch_size: int = 16
    base_lr: float = 2e-4
    encoder_lr_mult: float = 0.05
    weight_decay: float = 1e-4
    freeze_epochs: int = 3
    warmup_epochs: int = 3
    grad_clip: float = 1.0
    amp: bool = True
    num_workers: int = 6
    early_stop_patience: int = 25
    ema_decay: float = 0.0

    positive_ratio: float = 0.5

    w_dice: float = 0.4
    w_tversky: float = 0.4
    w_focal: float = 0.2
    tversky_alpha: float = 0.3
    tversky_beta: float = 0.7
    w_boundary: float = 0.3
    w_cls_binary: float = 0.4
    w_cls_multilabel: float = 0.2
    w_deep_sup: float = 0.4

    ssl_enabled: bool = True
    ssl_epochs: int = 15
    ssl_batch_size: int = 32
    ssl_lr: float = 5e-4
    ssl_temperature: float = 0.2

    cnn_encoder: str = "resnet50"
    trans_encoder: str = "pvt_v2_b2"
    baseline_trans_encoder: str = "pvt_v2_b2"
    baseline_cnn_encoder: str = "resnet50"
    decoder_channels: tuple = (256, 128, 64, 32)
    fuse_dims: tuple = (96, 192, 384, 512)
    dropout: float = 0.10
    use_cnn_stream: bool = True
    use_trans_stream: bool = True
    use_cross_fusion: bool = True
    use_mrhdc: bool = True
    use_coord_attn: bool = True
    use_attn_gate: bool = True
    use_deep_sup: bool = True
    use_cls_head: bool = True

    tta: bool = True
    threshold_grid: tuple = (0.20, 0.80, 25)
    min_component: int = 12

    n_bootstrap: int = 2000
    alpha: float = 0.05

    baselines: List[str] = field(
        default_factory=lambda: [
            "unet",
            "attention_unet",
            "unetpp",
            "resunet50",
            "deeplabv3p",
            "pvt_unet",
            "hemoclr_net",
        ]
    )
    run_ablation: bool = True
    ablation_folds: List[int] = field(default_factory=lambda: [0])

    @property
    def in_chans(self) -> int:
        """Input channels: multi-window stack (+ brain window of neighbours)."""
        return len(self.windows) + 2 * self.context_slices

    @property
    def cache_dir(self) -> str:
        tag = f"sz{self.img_size}_ss{int(self.skull_strip)}_cl{int(self.clahe)}"
        return os.path.join(DIR_CACHE, tag)

    def save(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(asdict(self), fh, indent=2, default=str)

    def clone(self, **kw) -> "Config":
        import copy

        c = copy.deepcopy(self)
        for k, v in kw.items():
            if not hasattr(c, k):
                raise AttributeError(f"Config has no field '{k}'")
            setattr(c, k, v)
        return c


def paper_config() -> Config:
    """The study as specified: one patient-wise split, 30 epochs, 4 baselines.

    Two deliberate departures from the first exploratory run, both applied to
    *every* model so the comparison stays fair:

    * ``encoder_lr_mult`` 0.05 -> 0.25.  At 1e-5 the pretrained encoders barely
      moved in 30 epochs, which is why the ImageNet-initialised models lost to a
      from-scratch U-Net.
    * ``ema_decay`` 0 -> 0.999.  With ~200 hemorrhage-positive training slices the
      epoch-to-epoch validation Dice swung by more than 0.2, so checkpoint
      selection was picking lucky epochs rather than better models.

    The proposed model additionally drops to a lighter encoder pair
    (ResNet-34 + PVTv2-B1, ~35 M against the earlier 66.7 M) because 318 positive
    slices in the whole dataset cannot support the larger version.  The
    transformer *baseline* stays at PVTv2-B2 via ``baseline_trans_encoder``, so
    shrinking the proposal never weakens what it is measured against.

    Resolution is 384 px, not 256.  The median lesion covers 273 px at 256x256 —
    under 1 % of the image, roughly a 16x16 patch — and at 256 px the proposed
    model only tied U-Net++ on validation.  384 px enlarges the same lesion about
    2.25x.  Every model is trained at 384, so the comparison is unaffected.

    Fusion width, dropout and weight decay come from the fold-0 validation sweep
    in ``ab_tune.py`` (``outputs/logs/ab_tuning_results.csv``): narrowing the
    randomly-initialised fusion stages from (96,192,384,512) to (64,128,256,384)
    and regularising harder raised best validation Dice from 0.427 to 0.500.
    Selection used the inner validation split only.
    """
    return Config(
        run_name="paper",
        img_size=384,
        epochs=30,
        ssl_epochs=20,
        ssl_batch_size=16,
        folds=[0],
        n_folds=5,
        batch_size=8,
        encoder_lr_mult=0.25,
        warmup_epochs=4,
        ema_decay=0.999,
        early_stop_patience=30,
        cnn_encoder="resnet34",
        trans_encoder="pvt_v2_b1",
        fuse_dims=(64, 128, 256, 384),
        dropout=0.20,
        weight_decay=3e-4,
        baselines=["unet", "attention_unet", "unetpp", "pvt_unet"],
        run_ablation=False,
        n_bootstrap=5000,
    )


def quick_config() -> Config:
    """Single-fold run at 256 px with schedules long enough to mean something.

    Budget on one GB10: ~35 s/epoch for the light baselines and ~42 s/epoch for
    the two-stream models, so 30 epochs x 8 models lands near two hours.  The
    ablation is run separately (``--stages ablation``) because 15 variants at the
    same depth would triple that.
    """
    return Config(
        run_name="quick",
        img_size=256,
        epochs=30,
        ssl_epochs=20,
        folds=[0],
        ablation_folds=[0],
        early_stop_patience=12,
        n_bootstrap=5000,
    )


def full_config() -> Config:
    """Paper-scale run: 5-fold patient-wise CV at 384 px."""
    return Config(
        run_name="full",
        img_size=384,
        epochs=70,
        ssl_epochs=60,
        batch_size=6,
        folds=[0, 1, 2, 3, 4],
        ablation_folds=[0],
        early_stop_patience=25,
        n_bootstrap=10000,
    )


PRESETS = {"paper": paper_config, "quick": quick_config, "full": full_config}
