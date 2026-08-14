"""HemoFusion-Net — the proposed hybrid CNN + Transformer model.

Two ImageNet-pretrained encoders run in parallel over the same 2.5D
multi-window CT input:

* a **CNN stream** (ResNet-50) that resolves the sharp, high-contrast texture of
  fresh blood, and
* a **transformer stream** (PVTv2) whose global receptive field supplies the
  anatomical context needed to separate hemorrhage from calcification, partial
  volume artefacts and the falx.

Their pyramids are merged scale-by-scale with *bidirectional* cross-modal
fusion — gated fusion at the three high-resolution stages, full multi-head
cross-attention at the bottleneck — and decoded by an attention-gated U-Net
decoder with deep supervision.  A second head performs slice-level hemorrhage
and subtype classification from the shared bottleneck, so detection and
delineation are learned jointly.

Every architectural component is switchable from ``Config`` for the ablation
study.
"""
from __future__ import annotations

from typing import Dict, List

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import MULTILABEL, Config
from .blocks import (
    AttentionGate,
    ConcatFusion,
    ConvBlock,
    ConvBNAct,
    CoordAttention,
    CrossAttentionFusion,
    GatedCrossFusion,
    MRHDC,
    MSPool,
    SingleStream,
)

FUSE_DIMS = (96, 192, 384, 512)


def _to_nchw(t: torch.Tensor, ch: int) -> torch.Tensor:
    """timm feature maps are NCHW, but a few transformer families emit NHWC."""
    if t.ndim == 4 and t.shape[1] != ch and t.shape[-1] == ch:
        return t.permute(0, 3, 1, 2).contiguous()
    return t


class HemoFusionNet(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.use_cnn = cfg.use_cnn_stream
        self.use_trans = cfg.use_trans_stream
        if not (self.use_cnn or self.use_trans):
            raise ValueError("at least one encoder stream must be enabled")

        self.cnn = self.cnn_ch = None
        if self.use_cnn:
            self.cnn = timm.create_model(
                cfg.cnn_encoder, pretrained=True, features_only=True,
                out_indices=(0, 1, 2, 3, 4), in_chans=cfg.in_chans,
            )
            self.cnn_ch: List[int] = list(self.cnn.feature_info.channels())

        self.trans = self.trans_ch = None
        if self.use_trans:
            self.trans = timm.create_model(
                cfg.trans_encoder, pretrained=True, features_only=True,
                in_chans=cfg.in_chans,
            )
            self.trans_ch: List[int] = list(self.trans.feature_info.channels())
            self.trans_ch = self.trans_ch[-4:]
            self._trans_keep = 4

        self.stem_ch = self.cnn_ch[0] if self.use_cnn else 0

        f1, f2, f3, f4 = getattr(cfg, "fuse_dims", FUSE_DIMS)
        cnn_stage = self.cnn_ch[1:] if self.use_cnn else [0, 0, 0, 0]
        trs_stage = self.trans_ch if self.use_trans else [0, 0, 0, 0]

        def make_fusion(i: int, cout: int, deepest: bool) -> nn.Module:
            cc, ct = cnn_stage[i], trs_stage[i]
            if not (self.use_cnn and self.use_trans):
                return SingleStream(cc if self.use_cnn else ct, cout)
            if not cfg.use_cross_fusion:
                return ConcatFusion(cc, ct, cout)
            return (CrossAttentionFusion(cc, ct, cout) if deepest
                    else GatedCrossFusion(cc, ct, cout))

        self.fuse1 = make_fusion(0, f1, False)
        self.fuse2 = make_fusion(1, f2, False)
        self.fuse3 = make_fusion(2, f3, False)
        self.fuse4 = make_fusion(3, f4, True)

        self.ms1, self.ms2, self.ms3 = MSPool(f1), MSPool(f2), MSPool(f3)
        if cfg.use_coord_attn:
            self.ca1, self.ca2, self.ca3 = CoordAttention(f1), CoordAttention(f2), CoordAttention(f3)
        else:
            self.ca1 = self.ca2 = self.ca3 = nn.Identity()

        self.bottleneck = MRHDC(f4) if cfg.use_mrhdc else ConvBNAct(f4, f4, k=3)

        d = cfg.dropout
        self.up3 = nn.ConvTranspose2d(f4, f3, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(f3, f2, 2, stride=2)
        self.up1 = nn.ConvTranspose2d(f2, f1, 2, stride=2)

        if cfg.use_attn_gate:
            self.ag3 = AttentionGate(f3, f3, f3 // 2)
            self.ag2 = AttentionGate(f2, f2, f2 // 2)
            self.ag1 = AttentionGate(f1, f1, f1 // 2)
        else:
            self.ag3 = self.ag2 = self.ag1 = None

        self.dc3 = ConvBlock(f3 * 2, f3, d)
        self.dc2 = ConvBlock(f2 * 2, f2, d)
        self.dc1 = ConvBlock(f1 * 2, f1, d)

        self.up_stem = nn.ConvTranspose2d(f1, 64, 2, stride=2)
        self.dc_stem = ConvBlock(64 + self.stem_ch, 64, d)
        self.up_full = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dc_full = ConvBlock(32, 32, d * 0.5)
        self.head = nn.Conv2d(32, 1, 1)

        self.ds3 = nn.Conv2d(f3, 1, 1) if cfg.use_deep_sup else None
        self.ds2 = nn.Conv2d(f2, 1, 1) if cfg.use_deep_sup else None

        if cfg.use_cls_head:
            self.cls_trunk = nn.Sequential(
                nn.Linear(2 * f4, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
                nn.Dropout(0.3),
            )
            self.cls_bin = nn.Linear(256, 1)
            self.cls_multi = nn.Linear(256, len(MULTILABEL))
        else:
            self.cls_trunk = self.cls_bin = self.cls_multi = None

    def load_ssl_encoder(self, path: str) -> int:
        """Load SimCLR-pretrained weights into the CNN stream."""
        sd = torch.load(path, map_location="cpu")
        sd = sd.get("model", sd)
        enc = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
        if not enc or self.cnn is None:
            return 0
        own = self.cnn.state_dict()
        keep = {k: v for k, v in enc.items() if k in own and own[k].shape == v.shape}
        own.update(keep)
        self.cnn.load_state_dict(own, strict=False)
        return len(keep)

    def encoder_parameters(self):
        mods = [m for m in (self.cnn, self.trans) if m is not None]
        return [p for m in mods for p in m.parameters()]

    def _encode(self, x):
        cnn_f = self.cnn(x) if self.use_cnn else None
        if cnn_f is not None:
            cnn_f = [_to_nchw(f, c) for f, c in zip(cnn_f, self.cnn_ch)]

        trs_f = None
        if self.use_trans:
            trs_f = self.trans(x)[-self._trans_keep:]
            trs_f = [_to_nchw(f, c) for f, c in zip(trs_f, self.trans_ch)]
        return cnn_f, trs_f

    def forward(self, x) -> Dict[str, torch.Tensor]:
        H, W = x.shape[-2:]
        cnn_f, trs_f = self._encode(x)

        c = cnn_f[1:] if cnn_f is not None else [None] * 4
        t = trs_f if trs_f is not None else [None] * 4
        pick = (lambda a, b: (a, b)) if (self.use_cnn and self.use_trans) else \
               (lambda a, b: (a if self.use_cnn else b, None))

        s1 = self.ca1(self.ms1(self.fuse1(*pick(c[0], t[0]))))
        s2 = self.ca2(self.ms2(self.fuse2(*pick(c[1], t[1]))))
        s3 = self.ca3(self.ms3(self.fuse3(*pick(c[2], t[2]))))
        b = self.bottleneck(self.fuse4(*pick(c[3], t[3])))

        def up(layer, src, ref):
            y = layer(src)
            if y.shape[-2:] != ref.shape[-2:]:
                y = F.interpolate(y, size=ref.shape[-2:], mode="bilinear", align_corners=False)
            return y

        u3 = up(self.up3, b, s3)
        k3 = self.ag3(u3, s3) if self.ag3 is not None else s3
        d3 = self.dc3(torch.cat([k3, u3], 1))

        u2 = up(self.up2, d3, s2)
        k2 = self.ag2(u2, s2) if self.ag2 is not None else s2
        d2 = self.dc2(torch.cat([k2, u2], 1))

        u1 = up(self.up1, d2, s1)
        k1 = self.ag1(u1, s1) if self.ag1 is not None else s1
        d1 = self.dc1(torch.cat([k1, u1], 1))

        if self.use_cnn:
            stem = cnn_f[0]
            us = up(self.up_stem, d1, stem)
            ds = self.dc_stem(torch.cat([us, stem], 1))
        else:
            us = F.interpolate(self.up_stem(d1), size=(H // 2, W // 2),
                               mode="bilinear", align_corners=False)
            ds = self.dc_stem(us)

        full = F.interpolate(self.up_full(ds), size=(H, W), mode="bilinear", align_corners=False)
        out: Dict[str, torch.Tensor] = {"seg": self.head(self.dc_full(full))}

        if self.ds3 is not None:
            out["ds"] = [
                F.interpolate(self.ds3(d3), size=(H, W), mode="bilinear", align_corners=False),
                F.interpolate(self.ds2(d2), size=(H, W), mode="bilinear", align_corners=False),
            ]

        if self.cls_trunk is not None:
            pooled = torch.cat([F.adaptive_avg_pool2d(b, 1).flatten(1),
                                F.adaptive_max_pool2d(b, 1).flatten(1)], 1)
            z = self.cls_trunk(pooled)
            out["cls"] = self.cls_bin(z)
            out["multi"] = self.cls_multi(z)
        return out
