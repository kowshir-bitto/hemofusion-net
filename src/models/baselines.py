"""Baseline segmentation networks for the comparison table.

Every baseline is wrapped so it returns the same ``dict`` interface as the
proposed model (``seg`` + optional ``cls``/``multi``), takes the identical
number of input channels, and — where it has a pretrained backbone — uses the
same ImageNet initialisation.  Only the architecture differs, which is what
makes the comparison fair.
"""
from __future__ import annotations

from typing import Dict, List

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import MULTILABEL, Config
from .blocks import AttentionGate, ConvBlock, ConvBNAct, MRHDC


class ClsHead(nn.Module):
    def __init__(self, cin: int, hidden: int = 256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(2 * cin, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.bin = nn.Linear(hidden, 1)
        self.multi = nn.Linear(hidden, len(MULTILABEL))

    def forward(self, feat):
        pooled = torch.cat([F.adaptive_avg_pool2d(feat, 1).flatten(1),
                            F.adaptive_max_pool2d(feat, 1).flatten(1)], 1)
        z = self.trunk(pooled)
        return self.bin(z), self.multi(z)


class UNet(nn.Module):
    def __init__(self, cfg: Config, base: int = 32):
        super().__init__()
        ch = [base, base * 2, base * 4, base * 8, base * 16]
        self.enc = nn.ModuleList()
        cin = cfg.in_chans
        for c in ch:
            self.enc.append(ConvBlock(cin, c, cfg.dropout))
            cin = c
        self.pool = nn.MaxPool2d(2)
        self.ups = nn.ModuleList(
            [nn.ConvTranspose2d(ch[i], ch[i - 1], 2, stride=2) for i in range(4, 0, -1)]
        )
        self.dec = nn.ModuleList(
            [ConvBlock(ch[i - 1] * 2, ch[i - 1], cfg.dropout) for i in range(4, 0, -1)]
        )
        self.head = nn.Conv2d(ch[0], 1, 1)
        self.cls = ClsHead(ch[-1]) if cfg.use_cls_head else None

    def forward(self, x) -> Dict[str, torch.Tensor]:
        skips: List[torch.Tensor] = []
        for i, e in enumerate(self.enc):
            x = e(x if i == 0 else self.pool(x))
            skips.append(x)
        bott = skips[-1]
        y = bott
        for up, dec, sk in zip(self.ups, self.dec, reversed(skips[:-1])):
            y = up(y)
            if y.shape[-2:] != sk.shape[-2:]:
                y = F.interpolate(y, size=sk.shape[-2:], mode="bilinear", align_corners=False)
            y = dec(torch.cat([sk, y], 1))
        out = {"seg": self.head(y)}
        if self.cls is not None:
            out["cls"], out["multi"] = self.cls(bott)
        return out


class AttentionUNet(UNet):
    def __init__(self, cfg: Config, base: int = 32):
        super().__init__(cfg, base)
        ch = [base, base * 2, base * 4, base * 8, base * 16]
        self.gates = nn.ModuleList(
            [AttentionGate(ch[i - 1], ch[i - 1], max(8, ch[i - 1] // 2)) for i in range(4, 0, -1)]
        )

    def forward(self, x) -> Dict[str, torch.Tensor]:
        skips: List[torch.Tensor] = []
        for i, e in enumerate(self.enc):
            x = e(x if i == 0 else self.pool(x))
            skips.append(x)
        bott = skips[-1]
        y = bott
        for up, dec, gate, sk in zip(self.ups, self.dec, self.gates, reversed(skips[:-1])):
            y = up(y)
            if y.shape[-2:] != sk.shape[-2:]:
                y = F.interpolate(y, size=sk.shape[-2:], mode="bilinear", align_corners=False)
            y = dec(torch.cat([gate(y, sk), y], 1))
        out = {"seg": self.head(y)}
        if self.cls is not None:
            out["cls"], out["multi"] = self.cls(bott)
        return out


class UNetPP(nn.Module):
    def __init__(self, cfg: Config, base: int = 32):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8, base * 16]
        self.pool = nn.MaxPool2d(2)
        d = cfg.dropout
        self.x00 = ConvBlock(cfg.in_chans, c[0], d)
        self.x10 = ConvBlock(c[0], c[1], d)
        self.x20 = ConvBlock(c[1], c[2], d)
        self.x30 = ConvBlock(c[2], c[3], d)
        self.x40 = ConvBlock(c[3], c[4], d)

        self.x01 = ConvBlock(c[0] + c[1], c[0], d)
        self.x11 = ConvBlock(c[1] + c[2], c[1], d)
        self.x21 = ConvBlock(c[2] + c[3], c[2], d)
        self.x31 = ConvBlock(c[3] + c[4], c[3], d)

        self.x02 = ConvBlock(c[0] * 2 + c[1], c[0], d)
        self.x12 = ConvBlock(c[1] * 2 + c[2], c[1], d)
        self.x22 = ConvBlock(c[2] * 2 + c[3], c[2], d)

        self.x03 = ConvBlock(c[0] * 3 + c[1], c[0], d)
        self.x13 = ConvBlock(c[1] * 3 + c[2], c[1], d)
        self.x04 = ConvBlock(c[0] * 4 + c[1], c[0], d)
        self.head = nn.Conv2d(c[0], 1, 1)
        self.cls = ClsHead(c[4]) if cfg.use_cls_head else None

    @staticmethod
    def _up(x, ref):
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x) -> Dict[str, torch.Tensor]:
        x00 = self.x00(x)
        x10 = self.x10(self.pool(x00))
        x01 = self.x01(torch.cat([x00, self._up(x10, x00)], 1))
        x20 = self.x20(self.pool(x10))
        x11 = self.x11(torch.cat([x10, self._up(x20, x10)], 1))
        x02 = self.x02(torch.cat([x00, x01, self._up(x11, x00)], 1))
        x30 = self.x30(self.pool(x20))
        x21 = self.x21(torch.cat([x20, self._up(x30, x20)], 1))
        x12 = self.x12(torch.cat([x10, x11, self._up(x21, x10)], 1))
        x03 = self.x03(torch.cat([x00, x01, x02, self._up(x12, x00)], 1))
        x40 = self.x40(self.pool(x30))
        x31 = self.x31(torch.cat([x30, self._up(x40, x30)], 1))
        x22 = self.x22(torch.cat([x20, x21, self._up(x31, x20)], 1))
        x13 = self.x13(torch.cat([x10, x11, x12, self._up(x22, x10)], 1))
        x04 = self.x04(torch.cat([x00, x01, x02, x03, self._up(x13, x00)], 1))
        out = {"seg": self.head(x04)}
        if self.cls is not None:
            out["cls"], out["multi"] = self.cls(x40)
        return out


class EncoderUNet(nn.Module):
    """Generic timm-encoder U-Net.

    With ``encoder='resnet50'`` this is the CNN-only control (ResUNet-50); with
    ``encoder='pvt_v2_b2'`` it is the transformer-only control.  Both isolate a
    single stream of the proposed hybrid.
    """

    def __init__(self, cfg: Config, encoder: str, dec: tuple = (256, 128, 64, 32)):
        super().__init__()
        self.enc = timm.create_model(encoder, pretrained=True, features_only=True,
                                     in_chans=cfg.in_chans)
        ch = list(self.enc.feature_info.channels())
        self.n = len(ch)
        self.ch = ch
        d = cfg.dropout

        self.ups, self.decs = nn.ModuleList(), nn.ModuleList()
        cur = ch[-1]
        for i, oc in enumerate(dec[: self.n - 1]):
            self.ups.append(nn.ConvTranspose2d(cur, oc, 2, stride=2))
            self.decs.append(ConvBlock(oc + ch[-2 - i], oc, d))
            cur = oc
        self.final = nn.Sequential(ConvBlock(cur, 32, d * 0.5), nn.Conv2d(32, 1, 1))
        self.cls = ClsHead(ch[-1]) if cfg.use_cls_head else None

    def forward(self, x) -> Dict[str, torch.Tensor]:
        H, W = x.shape[-2:]
        feats = self.enc(x)
        y = feats[-1]
        for i, (up, dec) in enumerate(zip(self.ups, self.decs)):
            sk = feats[-2 - i]
            y = up(y)
            if y.shape[-2:] != sk.shape[-2:]:
                y = F.interpolate(y, size=sk.shape[-2:], mode="bilinear", align_corners=False)
            y = dec(torch.cat([y, sk], 1))
        y = F.interpolate(y, size=(H, W), mode="bilinear", align_corners=False)
        out = {"seg": self.final(y)}
        if self.cls is not None:
            out["cls"], out["multi"] = self.cls(feats[-1])
        return out


class DeepLabV3Plus(nn.Module):
    def __init__(self, cfg: Config, encoder: str = "resnet50"):
        super().__init__()
        self.enc = timm.create_model(encoder, pretrained=True, features_only=True,
                                     out_indices=(1, 4), in_chans=cfg.in_chans)
        low_ch, high_ch = self.enc.feature_info.channels()
        self.aspp = nn.Sequential(ConvBNAct(high_ch, 256, k=1), MRHDC(256, rates=(1, 6, 12, 18)))
        self.low = ConvBNAct(low_ch, 48, k=1)
        self.fuse = nn.Sequential(ConvBlock(256 + 48, 256, cfg.dropout), ConvBNAct(256, 256, k=3))
        self.head = nn.Conv2d(256, 1, 1)
        self.cls = ClsHead(high_ch) if cfg.use_cls_head else None

    def forward(self, x) -> Dict[str, torch.Tensor]:
        H, W = x.shape[-2:]
        low, high = self.enc(x)
        y = self.aspp(high)
        y = F.interpolate(y, size=low.shape[-2:], mode="bilinear", align_corners=False)
        y = self.fuse(torch.cat([y, self.low(low)], 1))
        y = F.interpolate(y, size=(H, W), mode="bilinear", align_corners=False)
        out = {"seg": self.head(y)}
        if self.cls is not None:
            out["cls"], out["multi"] = self.cls(high)
        return out
