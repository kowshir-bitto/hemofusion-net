"""Model registry — one name -> one architecture, so experiments are declarative."""
from __future__ import annotations

from typing import Callable, Dict

import torch.nn as nn

from ..config import Config
from .baselines import AttentionUNet, DeepLabV3Plus, EncoderUNet, UNet, UNetPP
from .hemoclr import HemoCLRNet
from .hemofusion import HemoFusionNet

DISPLAY_NAMES: Dict[str, str] = {
    "unet": "U-Net",
    "attention_unet": "Attention U-Net",
    "unetpp": "U-Net++",
    "resunet50": "ResUNet-50 (CNN only)",
    "deeplabv3p": "DeepLabV3+",
    "pvt_unet": "PVTv2-UNet (Transformer only)",
    "hemoclr_net": "HemoCLR-Net (prior work)",
    "hemofusion": "HemoFusion-Net (proposed)",
}

_BUILDERS: Dict[str, Callable[[Config], nn.Module]] = {
    "unet": lambda cfg: UNet(cfg),
    "attention_unet": lambda cfg: AttentionUNet(cfg),
    "unetpp": lambda cfg: UNetPP(cfg),
    "resunet50": lambda cfg: EncoderUNet(cfg, cfg.baseline_cnn_encoder),
    "deeplabv3p": lambda cfg: DeepLabV3Plus(cfg, cfg.baseline_cnn_encoder),
    "pvt_unet": lambda cfg: EncoderUNet(cfg, cfg.baseline_trans_encoder),
    "hemoclr_net": lambda cfg: HemoCLRNet(cfg),
    "hemofusion": lambda cfg: HemoFusionNet(cfg),
}

SSL_CAPABLE = {"hemofusion", "hemoclr_net"}

PRETRAINED_ENCODER = {"resunet50", "deeplabv3p", "pvt_unet", "hemoclr_net", "hemofusion"}


def has_pretrained_encoder(name: str) -> bool:
    return name in PRETRAINED_ENCODER


def build_model(name: str, cfg: Config) -> nn.Module:
    if name not in _BUILDERS:
        raise KeyError(f"unknown model '{name}'. available: {sorted(_BUILDERS)}")
    return _BUILDERS[name](cfg)


def display_name(name: str) -> str:
    return DISPLAY_NAMES.get(name, name)


def count_params(model: nn.Module) -> Dict[str, float]:
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"params_M": total / 1e6, "trainable_M": train / 1e6}


__all__ = [
    "build_model", "display_name", "count_params", "DISPLAY_NAMES", "SSL_CAPABLE",
    "PRETRAINED_ENCODER", "has_pretrained_encoder",
    "UNet", "AttentionUNet", "UNetPP", "EncoderUNet", "DeepLabV3Plus",
    "HemoCLRNet", "HemoFusionNet",
]
