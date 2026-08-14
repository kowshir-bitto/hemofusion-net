"""HemoCLR-Net — reference implementation of the preceding single-stream model.

This reproduces the architecture the study builds on (SimCLR-pretrained
ResNet-50 encoder, MSPool + CoordAttention refined skips, MRHDC bottleneck,
attention-gated decoder, deep supervision) so the comparison table can quantify
what the hybrid dual-stream design actually adds.  It is given the identical
input channels, loss, schedule and augmentation as every other model.
"""
from __future__ import annotations

from typing import Dict

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import Config
from .baselines import ClsHead
from .blocks import AttentionGate, ConvBlock, CoordAttention, MRHDC, MSPool


class HemoCLRNet(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.enc = timm.create_model(
            "resnet50", pretrained=True, features_only=True,
            out_indices=(0, 1, 2, 3, 4), in_chans=cfg.in_chans,
        )
        c0, c1, c2, c3, c4 = self.enc.feature_info.channels()
        d = cfg.dropout

        self.ms1, self.ms2, self.ms3 = MSPool(c1), MSPool(c2), MSPool(c3)
        self.ca1, self.ca2, self.ca3 = CoordAttention(c1), CoordAttention(c2), CoordAttention(c3)
        self.bottleneck = MRHDC(c4)

        self.up4 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.ag4 = AttentionGate(c3, c3, c3 // 2)
        self.dc4 = ConvBlock(c3 * 2, 512, d)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.ag3 = AttentionGate(256, c2, 256)
        self.dc3 = ConvBlock(256 + c2, 256, d)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.ag2 = AttentionGate(128, c1, 128)
        self.dc2 = ConvBlock(128 + c1, 128, d)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dc1 = ConvBlock(64 + c0, 64, d)
        self.up0 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dc0 = ConvBlock(32, 32, d * 0.5)
        self.head = nn.Conv2d(32, 1, 1)

        self.ds3 = nn.Conv2d(256, 1, 1) if cfg.use_deep_sup else None
        self.ds2 = nn.Conv2d(128, 1, 1) if cfg.use_deep_sup else None
        self.cls = ClsHead(c4) if cfg.use_cls_head else None

    def load_ssl_encoder(self, path: str) -> int:
        sd = torch.load(path, map_location="cpu")
        sd = sd.get("model", sd)
        enc = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
        if not enc:
            return 0
        own = self.enc.state_dict()
        keep = {k: v for k, v in enc.items() if k in own and own[k].shape == v.shape}
        own.update(keep)
        self.enc.load_state_dict(own, strict=False)
        return len(keep)

    def encoder_parameters(self):
        return list(self.enc.parameters())

    @staticmethod
    def _up(layer, src, ref):
        y = layer(src)
        if y.shape[-2:] != ref.shape[-2:]:
            y = F.interpolate(y, size=ref.shape[-2:], mode="bilinear", align_corners=False)
        return y

    def forward(self, x) -> Dict[str, torch.Tensor]:
        H, W = x.shape[-2:]
        e0, e1, e2, e3, e4 = self.enc(x)
        s1 = self.ca1(self.ms1(e1))
        s2 = self.ca2(self.ms2(e2))
        s3 = self.ca3(self.ms3(e3))
        b = self.bottleneck(e4)

        u4 = self._up(self.up4, b, s3)
        d4 = self.dc4(torch.cat([self.ag4(u4, s3), u4], 1))
        u3 = self._up(self.up3, d4, s2)
        d3 = self.dc3(torch.cat([self.ag3(u3, s2), u3], 1))
        u2 = self._up(self.up2, d3, s1)
        d2 = self.dc2(torch.cat([self.ag2(u2, s1), u2], 1))
        u1 = self._up(self.up1, d2, e0)
        d1 = self.dc1(torch.cat([e0, u1], 1))
        d0 = self.dc0(F.interpolate(self.up0(d1), size=(H, W), mode="bilinear", align_corners=False))

        out: Dict[str, torch.Tensor] = {"seg": self.head(d0)}
        if self.ds3 is not None:
            out["ds"] = [
                F.interpolate(self.ds3(d3), size=(H, W), mode="bilinear", align_corners=False),
                F.interpolate(self.ds2(d2), size=(H, W), mode="bilinear", align_corners=False),
            ]
        if self.cls is not None:
            out["cls"], out["multi"] = self.cls(b)
        return out
