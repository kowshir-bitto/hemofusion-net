"""Building blocks: attention, multi-scale context and cross-modal fusion."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Sequential):
    def __init__(self, cin: int, cout: int, k: int = 3, s: int = 1, d: int = 1, g: int = 1):
        super().__init__(
            nn.Conv2d(cin, cout, k, s, padding=d * (k // 2), dilation=d, groups=g, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )


class ConvBlock(nn.Module):
    """Double 3x3 conv with the dropout *after* both convolutions."""

    def __init__(self, cin: int, cout: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(cin, cout), ConvBNAct(cout, cout), nn.Dropout2d(dropout)
        )

    def forward(self, x):
        return self.block(x)


class MSPool(nn.Module):
    """Multi-scale pooling: depthwise 3x3 branch fused with a 3x3 max branch.

    Cheap way to sharpen small, high-contrast lesions before they enter a skip
    connection.
    """

    def __init__(self, ch: int):
        super().__init__()
        self.dw = ConvBNAct(ch, ch, k=3, g=ch)
        self.mx = nn.MaxPool2d(3, stride=1, padding=1)
        self.fuse = ConvBNAct(2 * ch, ch, k=1)

    def forward(self, x):
        return self.fuse(torch.cat([self.dw(x), self.mx(x)], 1))


class MRHDC(nn.Module):
    """Multi-Rate Hybrid Dilated Convolution bottleneck.

    Four parallel 3x3 branches at dilation 1/2/4/8 plus a global-average branch,
    concatenated and projected back — an ASPP variant sized for the bottleneck.
    """

    def __init__(self, ch: int, rates=(1, 2, 4, 8)):
        super().__init__()
        mid = max(32, ch // 4)
        self.branches = nn.ModuleList([ConvBNAct(ch, mid, k=3, d=r) for r in rates])
        self.gpool = nn.Sequential(nn.AdaptiveAvgPool2d(1), ConvBNAct(ch, mid, k=1))
        self.fuse = ConvBNAct(mid * (len(rates) + 1), ch, k=1)

    def forward(self, x):
        feats = [b(x) for b in self.branches]
        g = F.interpolate(self.gpool(x), size=x.shape[-2:], mode="nearest")
        return self.fuse(torch.cat(feats + [g], 1)) + x


class CoordAttention(nn.Module):
    """Coordinate attention — factorises spatial attention into H and W profiles,
    which suits the elongated shape of subdural / subarachnoid bleeds."""

    def __init__(self, ch: int, reduction: int = 32):
        super().__init__()
        mid = max(8, ch // reduction)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.proj = ConvBNAct(ch, mid, k=1)
        self.conv_h = nn.Conv2d(mid, ch, 1, bias=False)
        self.conv_w = nn.Conv2d(mid, ch, 1, bias=False)

    def forward(self, x):
        _, _, H, W = x.shape
        h = self.pool_h(x)
        w = self.pool_w(x).permute(0, 1, 3, 2)
        hw = self.proj(torch.cat([h, w], dim=2))
        fh, fw = hw.split([H, W], dim=2)
        fw = fw.permute(0, 1, 3, 2)
        return x * torch.sigmoid(self.conv_h(fh)) * torch.sigmoid(self.conv_w(fw))


class AttentionGate(nn.Module):
    """Additive attention gate (Oktay et al.) filtering a skip connection with
    the coarser decoder signal."""

    def __init__(self, g_ch: int, x_ch: int, mid: int):
        super().__init__()
        self.wg = nn.Conv2d(g_ch, mid, 1, bias=False)
        self.wx = nn.Conv2d(x_ch, mid, 1, bias=False)
        self.psi = nn.Sequential(nn.Conv2d(mid, 1, 1, bias=False), nn.Sigmoid())

    def forward(self, g, x):
        if g.shape[-2:] != x.shape[-2:]:
            g = F.interpolate(g, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return x * self.psi(F.relu(self.wg(g) + self.wx(x), inplace=True))


class GatedCrossFusion(nn.Module):
    """Bidirectional gated fusion for the high-resolution stages.

    Each stream produces the channel gate *and* the spatial gate applied to the
    other stream, so local (CNN) and global (transformer) evidence modulate one
    another before concatenation.  A projected residual keeps the un-gated
    signal reachable.
    """

    def __init__(self, c_cnn: int, c_trans: int, cout: int, reduction: int = 8):
        super().__init__()
        self.pc = ConvBNAct(c_cnn, cout, k=1)
        self.pt = ConvBNAct(c_trans, cout, k=1)
        mid = max(8, cout // reduction)
        self.mlp_c = nn.Sequential(nn.Conv2d(cout, mid, 1), nn.ReLU(inplace=True),
                                   nn.Conv2d(mid, cout, 1))
        self.mlp_t = nn.Sequential(nn.Conv2d(cout, mid, 1), nn.ReLU(inplace=True),
                                   nn.Conv2d(mid, cout, 1))
        self.sp_c = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sp_t = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.out = ConvBNAct(2 * cout, cout, k=1)
        self.res = ConvBNAct(2 * cout, cout, k=1)

    @staticmethod
    def _spatial_desc(x):
        return torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)

    def forward(self, fc, ft):
        c = self.pc(fc)
        t = self.pt(ft)
        if t.shape[-2:] != c.shape[-2:]:
            t = F.interpolate(t, size=c.shape[-2:], mode="bilinear", align_corners=False)
        raw = torch.cat([c, t], 1)

        c_g = c * torch.sigmoid(self.mlp_t(F.adaptive_avg_pool2d(t, 1)))
        t_g = t * torch.sigmoid(self.mlp_c(F.adaptive_avg_pool2d(c, 1)))

        c_g = c_g * torch.sigmoid(self.sp_t(self._spatial_desc(t_g)))
        t_g = t_g * torch.sigmoid(self.sp_c(self._spatial_desc(c_g)))

        return self.out(torch.cat([c_g, t_g], 1)) + self.res(raw)


class CrossAttentionFusion(nn.Module):
    """True bidirectional multi-head cross-attention for the deepest stage.

    At stride 32 the token count is small (64 tokens at 256 px input), so full
    attention is affordable and gives the two streams a global exchange that the
    gated variant only approximates.
    """

    def __init__(self, c_cnn: int, c_trans: int, cout: int, heads: int = 8):
        super().__init__()
        self.pc = ConvBNAct(c_cnn, cout, k=1)
        self.pt = ConvBNAct(c_trans, cout, k=1)
        self.n_c = nn.LayerNorm(cout)
        self.n_t = nn.LayerNorm(cout)
        self.a_c2t = nn.MultiheadAttention(cout, heads, batch_first=True)
        self.a_t2c = nn.MultiheadAttention(cout, heads, batch_first=True)
        self.ff = nn.Sequential(nn.LayerNorm(2 * cout), nn.Linear(2 * cout, cout),
                                nn.GELU(), nn.Linear(cout, cout))
        self.out = ConvBNAct(cout, cout, k=3)

    def forward(self, fc, ft):
        c = self.pc(fc)
        t = self.pt(ft)
        if t.shape[-2:] != c.shape[-2:]:
            t = F.interpolate(t, size=c.shape[-2:], mode="bilinear", align_corners=False)
        B, C, H, W = c.shape

        cs = self.n_c(c.flatten(2).transpose(1, 2))
        ts = self.n_t(t.flatten(2).transpose(1, 2))
        c_att, _ = self.a_t2c(cs, ts, ts)
        t_att, _ = self.a_c2t(ts, cs, cs)
        merged = self.ff(torch.cat([cs + c_att, ts + t_att], dim=-1))
        merged = merged.transpose(1, 2).reshape(B, C, H, W)
        return self.out(merged) + c + t


class ConcatFusion(nn.Module):
    """Ablation control: plain concatenation + 1x1 projection, no cross-gating."""

    def __init__(self, c_cnn: int, c_trans: int, cout: int):
        super().__init__()
        self.proj = ConvBNAct(c_cnn + c_trans, cout, k=1)

    def forward(self, fc, ft):
        if ft.shape[-2:] != fc.shape[-2:]:
            ft = F.interpolate(ft, size=fc.shape[-2:], mode="bilinear", align_corners=False)
        return self.proj(torch.cat([fc, ft], 1))


class SingleStream(nn.Module):
    """Ablation control: pass one stream through when the other is disabled."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.proj = ConvBNAct(cin, cout, k=1)

    def forward(self, f, _unused=None):
        return self.proj(f)
