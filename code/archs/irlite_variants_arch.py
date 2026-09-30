# FDRNet: ONE lightweight backbone with a configurable block_type -> many RK3588-real-time variants.
# Ops restricted to RKNN-friendly ones (Conv2d/DepthwiseConv/PReLU/GRN-Mul/Add/Concat/Reshape/PixelShuffle),
# no GELU / heavy attention. Same loss+recipe as comparison; only architecture varies.
import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.archs.guided_baselines import build_baseline, BASELINES
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.degcodec2 import DegCodec
from basicsr.archs.degcodec3 import DegCodecCoarse
from basicsr.archs.crossmodal_blocks import CrossModalContext, BlockVEG, BlockVSG, BlockCRM, CrossModalFusion, CrossModalFusionSimple, HFLS, HFLS2, HFLS3, HFLS4, StructureGatedHFFusion


def _act(ch): return nn.PReLU(num_parameters=ch)


# ---------------------------------------------------------------- priority-2 switches
import os as _os

_DC_MODE = _os.environ.get('IR_DC_MODE', 'auto')        # auto | fixed16
_DESC = _os.environ.get('IR_DESC', 'rich')              # rich | basic
_REP_INIT = _os.environ.get('IR_REP_INIT', 'rand')      # rand | zero


def _dc_for(c, dc):
    """#8: adaptive bottleneck width so wide backbones stop being throttled by dc=16."""
    if dc is not None:
        return dc
    return max(16, c // 4) if _DC_MODE == 'auto' else 16


def _rep_1x1(conv):
    """#10: zero init makes the extra branch stay near zero for a long time; small random is better."""
    if _REP_INIT == 'zero':
        import torch.nn as _nn
        _nn.init.zeros_(conv.weight)
    else:
        import torch.nn as _nn
        _nn.init.normal_(conv.weight, mean=0.0, std=0.01)


def _gate_in(c, basic=False):
    """gate input width for the rich/basic descriptor (#9).

    basic -> [mean,std]                        : 2c
    rich  -> + grad mean/std (2) + 2 levels of per-channel mean/std (4c) : 6c + 2
    """
    return 2 * c if (basic or _DESC == 'basic') else 6 * c + 2


def deg_descriptor(x, basic=False):
    """#9: degradation descriptor.

    basic -> [mean, std] over the whole map                       (2c)
    rich  -> + gradient-energy mean/std (blur sensitive) +        (2)
             mean/std after 2x and 4x average pooling             (4)
    The probe study showed noise is highly identifiable (0.97) but blur is not (0.47); the extra
    gradient/multi-scale terms are exactly the blur-sensitive statistics.
    """
    import torch
    import torch.nn.functional as _F
    flat = torch.cat([x.mean(dim=(2, 3)), x.std(dim=(2, 3))], 1)
    if basic or _DESC == 'basic':
        return flat
    gx = x[:, :, 1:, :] - x[:, :, :-1, :]
    gy = x[:, :, :, 1:] - x[:, :, :, :-1]
    g = (gx[:, :, :, :-1] ** 2 + gy[:, :, :-1, :] ** 2).sqrt()
    gstat = torch.stack([g.mean(dim=(1, 2, 3)), g.std(dim=(1, 2, 3))], 1)          # [B,2]
    stats = []
    for s in (2, 4):
        p = _F.avg_pool2d(x, s) if min(x.shape[-2:]) >= s else x
        stats += [p.mean(dim=(2, 3)), p.std(dim=(2, 3))]
    ms = torch.cat(stats, 1)                                                      # [B,4]
    return torch.cat([flat, gstat, ms], 1)


class BlockCS(nn.Module):
    """(cs) channel-split distillation: split -> 3x3 branch + 1x1 branch -> fuse."""
    def __init__(self, c):
        super().__init__(); h = c // 2
        self.a = nn.Conv2d(h, h, 3, padding=1)
        self.b = nn.Conv2d(h, h, 1)
        self.f = nn.Sequential(nn.Conv2d(c, c, 1), _act(c))
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        y = torch.cat([self.a(x1), self.b(x2)], dim=1)
        return x + self.f(y)


class BlockDW(nn.Module):
    """(dw) depthwise-separable + channel attention (SE)."""
    def __init__(self, c):
        super().__init__()
        self.dw = nn.Sequential(nn.Conv2d(c, c, 3, padding=1, groups=c), _act(c), nn.Conv2d(c, c, 1))
        self.se = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, c // 4, 1), _act(c // 4), nn.Conv2d(c // 4, c, 1), nn.Sigmoid())
    def forward(self, x):
        return x + self.dw(x) * self.se(x)


class BlockGS(nn.Module):
    """(gs) grouped conv + channel shuffle."""
    def __init__(self, c):
        super().__init__()
        g = 2
        self.gc = nn.Sequential(nn.Conv2d(c, c, 3, padding=1, groups=g), _act(c), nn.Conv2d(c, c, 1))
        self.f = nn.Conv2d(c, c, 1)
    def forward(self, x):
        y = self.gc(x)
        c = y.size(1)
        y = y.view(y.size(0), 2, c // 2, y.size(2), y.size(3)).transpose(1, 2).reshape(y.size(0), c, y.size(2), y.size(3))
        return x + self.f(y)


class BlockMS(nn.Module):
    """(ms) multi-scale dilated conv pyramid + concat + fuse."""
    def __init__(self, c):
        super().__init__()
        self.d1 = nn.Conv2d(c, c, 3, padding=1, dilation=1)
        self.d2 = nn.Conv2d(c, c, 3, padding=2, dilation=2)
        self.d4 = nn.Conv2d(c, c, 3, padding=4, dilation=4)
        self.f = nn.Sequential(nn.Conv2d(3 * c, c, 1), _act(c))
    def forward(self, x):
        y = torch.cat([self.d1(x), self.d2(x), self.d4(x)], dim=1)
        return x + self.f(y)


class BlockRP(nn.Module):
    """(rp) re-parameterizable multi-branch (1x1-3x3 + 3x3 + sobel-like), train-time branches; fusable at inference."""
    def __init__(self, c):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv2d(c, c, 1), nn.Conv2d(c, c, 3, padding=1))
        self.b2 = nn.Conv2d(c, c, 3, padding=1)
        # sobel-like depthwise edge branch
        k = torch.zeros(c, 1, 3, 3); k[:, 0, 0, 0] = 1; k[:, 0, 1, 0] = 2; k[:, 0, 2, 0] = 1
        k[:, 0, 0, 2] = -1; k[:, 0, 1, 2] = -2; k[:, 0, 2, 2] = -1
        self.b3 = nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False)
        self.b3.weight = nn.Parameter(k)
        self.f = nn.Sequential(nn.Conv2d(3 * c, c, 1), _act(c))
    def forward(self, x):
        y = torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1)
        return x + self.f(y)


class BlockSL(nn.Module):
    """(sl) ultra-light spectral depthwise + 1x1 (low-FLOPs baseline)."""
    def __init__(self, c):
        super().__init__()
        self.op = nn.Sequential(nn.Conv2d(c, c, 3, padding=1, groups=c), _act(c), nn.Conv2d(c, c, 1))
    def forward(self, x):
        return x + self.op(x)


class BlockGF(nn.Module):
    """(gf) graph spectral filter: Chebyshev polynomial of the local graph adjacency (3x3 conv = adjacency op)."""
    def __init__(self, c, order=2):
        super().__init__(); self.o = order
        self.op = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), _act(c))
        self.th = nn.ModuleList([nn.Conv2d(c, c, 1) for _ in range(order + 1)])
        self.f = nn.Conv2d(c, c, 1)
    def forward(self, x):
        T = [x]
        T.append(self.op(T[0]))
        for k in range(1, self.o):
            T.append(2 * self.op(T[k]) - T[k - 1])
        y = 0
        for k in range(self.o + 1):
            y = y + self.th[k](T[k])
        return x + self.f(y)


class BlockKR(nn.Module):
    """(kr) large-kernel reparameterization: dwconv5x5 + dwconv3x3 + 1x1 branches, fusable at inference."""
    def __init__(self, c):
        super().__init__()
        self.k5 = nn.Conv2d(c, c, 5, padding=2, groups=c)
        self.k3 = nn.Conv2d(c, c, 3, padding=1, groups=c)
        self.k1 = nn.Conv2d(c, c, 1)
        self.f = nn.Sequential(nn.Conv2d(c, c, 1), _act(c))
    def forward(self, x):
        return x + self.f(self.k5(x) + self.k3(x) + self.k1(x))


class BlockWP(nn.Module):
    """(wp) wavelet-like: learnable low/high frequency split via grouped convs."""
    def __init__(self, c):
        super().__init__(); g = 4; h = c // 2
        self.lo = nn.Sequential(nn.Conv2d(c, h, 1), _act(h))
        self.hi = nn.Sequential(nn.Conv2d(c, h, 3, padding=1, groups=g), _act(h))
        self.f = nn.Sequential(nn.Conv2d(c, c, 1), _act(c))
    def forward(self, x):
        lo = self.lo(x); hi = self.hi(x)
        return x + self.f(torch.cat([lo, hi], dim=1))


class BlockMO(nn.Module):
    """(mo) mixture-of-experts: 2 expert convs + sigmoid channel gate weighted sum."""
    def __init__(self, c):
        super().__init__()
        self.e1 = nn.Conv2d(c, c, 3, padding=1)
        self.e2 = nn.Conv2d(c, c, 5, padding=2)
        self.g = nn.Sequential(nn.Conv2d(c, c, 1), nn.Sigmoid())
        self.f = nn.Conv2d(c, c, 1)
    def forward(self, x):
        g = self.g(x)
        y = g * self.e1(x) + (1 - g) * self.e2(x)
        return x + self.f(y)


class BlockDY(nn.Module):
    """(dy) dynamic content-conditioned modulation: pooled latent -> per-channel scale on a depthwise conv."""
    def __init__(self, c):
        super().__init__(); self.c = c
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.op = nn.Sequential(nn.Conv2d(c, c, 3, padding=1, groups=c), _act(c), nn.Conv2d(c, c, 1))
        self.gamma = nn.Sequential(nn.Conv2d(c, c, 1), nn.Sigmoid())
    def forward(self, x):
        g = self.gamma(self.pool(x))
        return x + self.op(x) * g


class BlockODE(nn.Module):
    """(ode) neural ODE / unrolled diffusion step: x + h*(denoise(x) - x)."""
    def __init__(self, c):
        super().__init__(); self.h = nn.Parameter(torch.tensor(0.2))
        self.n = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), _act(c), nn.Conv2d(c, c, 1))
    def forward(self, x):
        return x + self.h * (self.n(x) - x)


class BlockGC(nn.Module):
    """(gc) coordinate-aware conv (GoConv): append normalized x,y coords then conv."""
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(c + 2, c, 3, padding=1), _act(c))
    def forward(self, x):
        B, C, H, W = x.shape
        yy = torch.linspace(-1, 1, H, device=x.device).view(1, 1, H, 1).expand(B, 1, H, W)
        xx = torch.linspace(-1, 1, W, device=x.device).view(1, 1, 1, W).expand(B, 1, H, W)
        xc = torch.cat([x, yy, xx], dim=1)
        return x + self.conv(xc)


class BlockHY(nn.Module):
    """(hy) hypernetwork/FiLM modulation: latent -> per-channel scale+shift on a conv output."""
    def __init__(self, c):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.gamma = nn.Sequential(nn.Conv2d(c, c, 1), nn.Sigmoid())
        self.beta = nn.Conv2d(c, c, 1)
        self.op = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), _act(c), nn.Conv2d(c, c, 1))
    def forward(self, x):
        lat = self.pool(x)
        y = self.op(x)
        return x + self.gamma(lat) * y + self.beta(lat)


class BlockSSM(nn.Module):
    """(ssm) lightweight state-space-like: causal depthwise conv (finite-horizon recurrence) + 1x1."""
    def __init__(self, c):
        super().__init__()
        self.causal = nn.Conv2d(c, c, 3, padding=1, groups=c)  # depthwise context
        self.mix = nn.Sequential(nn.Conv2d(c, c, 1), _act(c))
        self.gate = nn.Sequential(nn.Conv2d(c, c, 1), nn.Sigmoid())
    def forward(self, x):
        h = self.causal(x)
        return x + self.mix(h) * self.gate(x)


class BlockLP(nn.Module):
    """(lp) Laplacian pyramid / 2-scale residual."""
    def __init__(self, c):
        super().__init__()
        self.d = nn.Sequential(nn.Conv2d(c, c, 1), _act(c))
        self.b = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), _act(c), nn.Conv2d(c, c, 1))
    def forward(self, x):
        low = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        low = self.b(F.interpolate(low, scale_factor=2.0, mode='bilinear', align_corners=False))
        return x + self.d(low)


class BlockD2(nn.Module):
    """(d2) dual-branch: 3x3 conv + dilated conv -> concat -> fuse."""
    def __init__(self, c):
        super().__init__()
        self.a = nn.Conv2d(c, c // 2, 3, padding=1)
        self.b = nn.Conv2d(c, c // 2, 3, padding=2, dilation=2)
        self.f = nn.Sequential(nn.Conv2d(c, c, 1), _act(c))
    def forward(self, x):
        y = torch.cat([self.a(x), self.b(x)], dim=1)
        return x + self.f(y)


class BlockRG(nn.Module):
    """(rg) MIX: reparameterization (multi-branch) + graph spectral filter."""
    def __init__(self, c):
        super().__init__(); self.rp = BlockRP(c); self.gf = BlockGF(c)
        self.f = nn.Conv2d(c, c, 1)
    def forward(self, x):
        return x + self.f(self.rp(x) + self.gf(x) - x)


class BlockCD(nn.Module):
    """(cd) MIX: channel-split distillation + channel gate (SE-like)."""
    def __init__(self, c):
        super().__init__(); self.cs = BlockCS(c)
        self.se = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, c // 4, 1), _act(c // 4), nn.Conv2d(c // 4, c, 1), nn.Sigmoid())
    def forward(self, x):
        return x + self.cs(x) * self.se(x)


class BlockOD(nn.Module):
    """(od) MIX: neural-ODE diffusion step + causal depthwise (state-space-like)."""
    def __init__(self, c):
        super().__init__(); self.ode = BlockODE(c)
        self.causal = nn.Conv2d(c, c, 3, padding=1, groups=c)
        self.f = nn.Conv2d(c, c, 1)
    def forward(self, x):
        h = self.ode(x)
        return x + self.f(self.causal(h))


class BlockMA(nn.Module):
    """(ma) MIX: multi-scale dilated pyramid + channel attention."""
    def __init__(self, c):
        super().__init__(); self.ms = BlockMS(c)
        self.se = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, c // 4, 1), _act(c // 4), nn.Conv2d(c // 4, c, 1), nn.Sigmoid())
    def forward(self, x):
        return x + self.ms(x) * self.se(x)


class BlockMP(nn.Module):
    """(mp) MIX: mixture-of-experts + Laplacian pyramid."""
    def __init__(self, c):
        super().__init__(); self.mo = BlockMO(c); self.lp = BlockLP(c)
        self.f = nn.Conv2d(c, c, 1)
    def forward(self, x):
        return x + self.f(self.mo(x) + self.lp(x) - x)


class BlockFR(nn.Module):
    """(fr) NEW mechanism: content-adaptive fractional-order graph spectral operator ~ L^{alpha(x)}.
    Theory: for a graph Laplacian L (sym. PSD, spectrum in [0, lammax]) and alpha in [0,1],
    L^alpha = U diag(lambda^alpha) U^T is sym. PSD; alpha->0 is near-identity (sharp), alpha->1 is diffusion (smooth).
    We approximate lambda^alpha on [0,lammax] with a Chebyshev polynomial whose coefficients are
    generated from a content-adaptive alpha(x) (global pooled value), giving a spectrally-adaptive filter.
    """
    def __init__(self, c, K=3):
        super().__init__(); self.c = c; self.K = K
        self.op = nn.Conv2d(c, c, 3, padding=1)  # normalized local adjacency op (<= lammax)
        self.alpha = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, c, 1), nn.Sigmoid())
        self.gamma = nn.ModuleList([nn.Conv2d(c, c, 1) for _ in range(K + 1)])
        self.f = nn.Conv2d(c, c, 1)
    def forward(self, x):
        a = self.alpha(x)  # (B,C,1,1) content-adaptive order in (0,1)
        # Chebyshev terms of the (rescaled) graph operator
        xr = x * 0.5
        T = [xr]
        T.append(self.op(T[0]) - 0.5 * T[0])
        for k in range(1, self.K):
            T.append(2 * (self.op(T[k]) - 0.5 * T[k]) - T[k - 1])
        y = 0
        for k in range(self.K + 1):
            y = y + self.gamma[k](T[k]) * a
        return x + self.f(y)


class BlockGA(nn.Module):
    """(ga) graph-attention: 3x3 graph conv + per-channel attention gate."""
    def __init__(self, c):
        super().__init__()
        self.g = nn.Conv2d(c, c, 3, padding=1)
        self.att = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, c // 4, 1), _act(c // 4), nn.Conv2d(c // 4, c, 1), nn.Sigmoid())
        self.f = nn.Conv2d(c, c, 1)
    def forward(self, x):
        return x + self.f(self.g(x) * self.att(x))


class BlockMC(nn.Module):
    """(mc) multi-context kernel: parallel depthwise 3x3/5x5/7x7 -> concat -> 1x1."""
    def __init__(self, c):
        super().__init__()
        self.k3 = nn.Conv2d(c, c, 3, padding=1, groups=c)
        self.k5 = nn.Conv2d(c, c, 5, padding=2, groups=c)
        self.k7 = nn.Conv2d(c, c, 7, padding=3, groups=c)
        self.f = nn.Sequential(nn.Conv2d(3 * c, c, 1), _act(c))
    def forward(self, x):
        return x + self.f(torch.cat([self.k3(x), self.k5(x), self.k7(x)], dim=1))


class BlockLR(nn.Module):
    """(lr) low-rank separable conv: 1x5 then 5x1 (rank-1 factorized depthwise)."""
    def __init__(self, c):
        super().__init__()
        self.h = nn.Conv2d(c, c, (1, 5), padding=(0, 2), groups=c)
        self.v = nn.Conv2d(c, c, (5, 1), padding=(2, 0), groups=c)
        self.f = nn.Sequential(nn.Conv2d(c, c, 1), _act(c))
    def forward(self, x):
        return x + self.f(self.v(self.h(x)))


class BlockSA(nn.Module):
    """(sa) lightweight channel+spatial attention (1x1 channel gate + depthwise spatial gate)."""
    def __init__(self, c):
        super().__init__()
        self.cg = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, c // 4, 1), _act(c // 4), nn.Conv2d(c // 4, c, 1), nn.Sigmoid())
        self.sg = nn.Sequential(nn.Conv2d(c, 1, 1), nn.Sigmoid())
        self.f = nn.Conv2d(c, c, 1)
    def forward(self, x):
        return x + self.f(x * self.cg(x) * self.sg(x))


class BlockEQ(nn.Module):
    """(eq) Reflection-equivariant residual operator.
    The depthwise kernel is group-symmetrized over the reflection group D1={id, hv, vv, hvv}
    so the operator is HARD-reflection-equivariant (a hard constraint, not learned)."""
    def __init__(self, c):
        super().__init__()
        self.k = nn.Parameter(torch.randn(c, 1, 3, 3) * 0.02)
        self.gate = nn.Sequential(nn.Conv2d(c, c, 1), nn.Sigmoid())
        self.mix = nn.Conv2d(c, c, 1)
    def _sym_kernel(self):
        k = self.k
        return (k + torch.flip(k, dims=[3]) + torch.flip(k, dims=[2]) + torch.flip(torch.flip(k, dims=[2]), dims=[3])) / 4
    def forward(self, x):
        k = self._sym_kernel()
        y = F.conv2d(x, k, padding=1, groups=x.size(1))
        return x + self.mix(y * self.gate(x))


class BlockPR(nn.Module):
    """(pr) perfect-reconstruction (PR) orthonormal filter-bank operator (signal theory).
    Haar analysis [L,H] is orthonormal (energy-preserving) and synthesis (L+H)/sqrt2 == x (PR identity).
    Learnable band gains weight the subbands while keeping the PR-consistent structure."""
    def __init__(self, c):
        super().__init__()
        self.low = nn.Sequential(nn.Conv2d(c, c, 1), nn.Sigmoid())
        self.high = nn.Sequential(nn.Conv2d(c, c, 1), nn.Sigmoid())
        self.f = nn.Conv2d(c, c, 1)
    def forward(self, x):
        s2 = 2.0 ** 0.5
        lo = (x + torch.flip(x, dims=[3])) / s2
        hi = (x - torch.flip(x, dims=[3])) / s2
        rec = (lo * self.low(x) + hi * self.high(x)) / s2
        return x + self.f(rec)


class BlockPG(nn.Module):
    """(pg) learnable proximal-gradient lossy-reconstruction operator (MAP/PnP-inspired).
    Represents one denoising-as-proximal step: x + eta*(den(x) - x) (RED).
    Lossy: recovers a smoothed/regularized estimate, not the true HR."""
    def __init__(self, c):
        super().__init__()
        self.eta = nn.Parameter(torch.tensor(0.1))
        self.den = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.PReLU(c), nn.Conv2d(c, c, 3, padding=1), nn.PReLU(c), nn.Conv2d(c, c, 1))
        self.f = nn.Conv2d(c, c, 1)
    def forward(self, x):
        d = self.den(x)
        return x + self.f(self.eta * d + (1 - self.eta) * x)


class BlockBD(nn.Module):
    """(bd) low/high-FREQUENCY decoupled distillation (real coupling decoupling): a low-pass (smooth) path
    and a high-pass (detail) residual path, using a learnable low-pass filter to separate the two."""
    def __init__(self, c):
        super().__init__()
        self.low = nn.Sequential(nn.Conv2d(c, c, 3, padding=1, groups=c), nn.PReLU(c), nn.Conv2d(c, c, 1))
        self.high = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.PReLU(c), nn.Conv2d(c, c, 3, padding=1))
        self.f = nn.Conv2d(c, c, 1)
    def forward(self, x):
        low = self.low(x)             # low-frequency (smooth) estimate
        high = self.high(x - low)     # high-frequency residual (detail)
        rec = low + high              # decoupled low/high reconstruction
        return x + self.f(rec)


class BlockMSD(nn.Module):
    """(msd) multi-scale + dense-coupling distillation: a 1/2-scale coarse branch (structure/low-freq)
    + a fine branch (detail) + a cheap depthwise cross-channel coupling (coupling HELPS, not decoupling)."""
    def __init__(self, c):
        super().__init__()
        self.down = nn.Sequential(nn.Conv2d(c, c, 1), _act(c))
        self.coarse = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), _act(c), nn.Conv2d(c, c, 1))
        self.fine = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), _act(c), nn.Conv2d(c, c, 1))
        self.dw = nn.Conv2d(c, c, 3, padding=1, groups=c)
        self.f = nn.Conv2d(c, c, 1)
    def forward(self, x):
        low = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        low = self.coarse(low)
        low = F.interpolate(low, scale_factor=2.0, mode='bilinear', align_corners=False)
        fine = self.fine(x)
        coupled = self.dw(x + low)
        return x + self.f(fine + low + coupled)


class BlockDR(nn.Module):
    """(dr) channel-split distillation block with a configurable COMPRESSION RATIO (Q2).
    ratio = fraction of channels processed by the heavy 3x3; the rest pass through a cheap 1x1."""
    def __init__(self, c, ratio=0.5):
        super().__init__(); self.ratio = ratio
        d = max(1, int(round(c * ratio)))
        self.conv3 = nn.Conv2d(d, d, 3, padding=1)
        self.conv1 = nn.Conv2d(c - d, c - d, 1)
        self.f = nn.Conv2d(c, c, 1)
    def forward(self, x):
        d = self.conv3.weight.size(0)
        x1, x2 = x[:, :d], x[:, d:]
        return x + self.f(torch.cat([self.conv3(x1), self.conv1(x2)], dim=1))


class BlockAF(nn.Module):
    """(af) attention-weighted feature-reweighting block (Q1: use complementary info better)."""
    def __init__(self, c):
        super().__init__()
        self.att = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, c // 4, 1), _act(c // 4), nn.Conv2d(c // 4, c, 1), nn.Sigmoid())
        self.f = nn.Conv2d(c, c, 1)
    def forward(self, x):
        return x + self.f(x * self.att(x))


class BlockIRD(nn.Module):
    """(ird) Importance-Recalibrated & Rank-Preserving Distillation block.
    Theory: RFDB distillation drops effective rank (0.77->0.41); cross-block features are de-correlated/complementary.
    Fix: (1) SE importance recalibration to USE the compressed info; (2) a group-conv rank-preserving pathway to LOSE less.
    mode: 'full' (SE+rank), 'nose' (rank only), 'norank' (SE only)."""
    def __init__(self, c, mode='full'):
        super().__init__(); self.mode = mode
        assert c % 2 == 0
        self.conv3 = nn.Conv2d(c // 2, c // 2, 3, padding=1)
        self.conv1 = nn.Conv2d(c - c // 2, c - c // 2, 1)
        self.se = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, c // 4, 1), _act(c // 4), nn.Conv2d(c // 4, c, 1), nn.Sigmoid())
        self.g = nn.Conv2d(c, c, 3, padding=1, groups=4)
        self.f = nn.Conv2d(c, c, 1)
    def forward(self, x):
        c = x.size(1)
        x1, x2 = x[:, :c // 2], x[:, c // 2:]
        y = torch.cat([self.conv3(x1), self.conv1(x2)], dim=1)
        if self.mode == 'nose':
            y = y + self.g(x)
        elif self.mode == 'norank':
            y = y * self.se(y)
        else:
            y = y * self.se(y) + self.g(x)
        return x + self.f(y)


class BlockRD(nn.Module):
    """(rd) recurrent/recursive distillation: a weight-shared block applied twice -> deeper at same params."""
    def __init__(self, c):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), _act(c), nn.Conv2d(c, c, 1))
        self.b2 = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), _act(c), nn.Conv2d(c, c, 1))
        self.f = nn.Conv2d(c, c, 1)
    def forward(self, x):
        h = self.b1(x) + x
        h = self.b2(h) + h
        return x + self.f(h)


class BlockSD(nn.Module):
    """(sd) spectral-selective distillation: distilled chunk frozen (1x1); remaining channels filtered by a
    content-adaptive fractional spectral operator (fr) -> distillation selects by frequency, not arbitrary half."""
    def __init__(self, c):
        super().__init__()
        self.distil = nn.Conv2d(c, c // 2, 1)
        self.fr = BlockFR(c // 2)
        self.f = nn.Conv2d(c, c, 1)
    def forward(self, x):
        c = x.size(1); dc = c // 2
        d = self.distil(x)          # frozen distilled chunk
        r = self.fr(x[:, dc:])      # spectral-selective on the remaining half
        return x + self.f(torch.cat([d, r], dim=1))


class BlockIB(nn.Module):
    """(ib) VIB bottleneck (information-bottleneck distillation, loss side).
    Maps features to (mu, logvar), reparameterized sample Z=mu+std*eps, caches pooled log-var so
    EdgeKLLoss (with kl_weight=beta) adds the KL(q(zi|xi)||N(0,I)) term -> rate-distortion IB."""
    def __init__(self, c, bottleneck=None):
        super().__init__(); b = bottleneck or c
        self.mu = nn.Conv2d(c, b, 1)
        self.logvar = nn.Conv2d(c, b, 1)
        self.cached_pooled_log_var = None
        self.dec = nn.Conv2d(b, c, 1)
    def forward(self, x):
        mu = self.mu(x)
        logvar = torch.clamp(self.logvar(x), -10.0, 10.0)
        self.cached_pooled_log_var = logvar.mean(dim=(2, 3), keepdim=True)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        z = mu + std * eps if self.training else mu
        return x + self.dec(z)


class BlockIBD(nn.Module):
    """(ibd) rate-distortion-aware distillation (R-D Distill module): fr spectral-selective + info-aware gate
    + anti-bottleneck residual + VIB bottleneck (KL via cached log-var -> EdgeKLLoss adds beta*KL)."""
    def __init__(self, c):
        super().__init__()
        self.d = nn.Conv2d(c, c // 2, 1)
        self.fr = BlockFR(c // 2)
        self.gate = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, c, 1), nn.Sigmoid())
        self.res = nn.Conv2d(c, c, 3, padding=1)
        self.ib = BlockIB(c)
        self.f = nn.Conv2d(c, c, 1)
    def forward(self, x):
        c = x.size(1); dc = c // 2
        d = self.d(x)
        r = self.fr(x[:, dc:])
        y = torch.cat([d, r], dim=1)
        y = y * self.gate(y)
        z = self.ib(y)
        out = self.res(x) + z
        return x + self.f(out)


class BlockLK(nn.Module):
    """(lk) large-kernel frequency-enhanced: large depthwise conv (big RF) + fr spectral band enhancement."""
    def __init__(self, c):
        super().__init__()
        self.large = nn.Sequential(nn.Conv2d(c, c, 7, padding=3, groups=c), _act(c), nn.Conv2d(c, c, 1))
        self.fr = BlockFR(c)
        self.f = nn.Conv2d(c, c, 1)
    def forward(self, x):
        return x + self.f(self.large(x) + self.fr(x))


class BlockSP(nn.Module):
    """(sp) spectral-phase prompt distillation (IRLite-SP): a per-channel spectral/phase prompt gate
    (GPSMamba/PIRN-style) applied to a fr-based spectral-selective distillation path."""
    def __init__(self, c):
        super().__init__()
        self.distil = nn.Conv2d(c, c // 2, 1)
        self.fr = BlockFR(c // 2)
        self.prompt = nn.Sequential(nn.Conv2d(c, c, 3, padding=1, groups=c), _act(c), nn.Conv2d(c, c, 1), nn.Sigmoid())
        self.f = nn.Conv2d(c, c, 1)
    def forward(self, x):
        c = x.size(1); dc = c // 2
        d = self.distil(x)
        r = self.fr(x[:, dc:])
        p = self.prompt(x)
        y = torch.cat([d, r], dim=1) * p
        return x + self.f(y)


class BlockRFDS(nn.Module):
    """(rfds) RFDB distillation BACKBONE (progressive channel-split info distillation) + sp (spectral-phase prompt)
    + dag (degradation-aware gate). mode for component ablations (full/nosp/nodag)."""
    def __init__(self, c, mode='full'):
        super().__init__(); self.mode = mode
        dc = c // 2
        self.c1_d = nn.Conv2d(c, dc, 1); self.c1_r = nn.Conv2d(c, c, 3, padding=1)
        self.c2_d = nn.Conv2d(c, dc, 1); self.c2_r = nn.Conv2d(c, c, 3, padding=1)
        self.c3_d = nn.Conv2d(c, dc, 1); self.c3_r = nn.Conv2d(c, c, 3, padding=1)
        self.c4 = nn.Conv2d(c, dc, 3, padding=1)
        self.c5 = nn.Conv2d(dc * 4, c, 1)
        # channel attention (contrast + avg, CCA-style)
        self.cca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, c // 4, 1), _act(c // 4), nn.Conv2d(c // 4, c, 1), nn.Sigmoid())
        self.sp = nn.Sequential(nn.Conv2d(c, c, 3, padding=1, groups=c), _act(c), nn.Conv2d(c, c, 1), nn.Sigmoid())
        self.dag = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, c // 4, 1), _act(c // 4), nn.Conv2d(c // 4, c, 1), nn.Sigmoid())
        self.act = nn.PReLU()
    def forward(self, x):
        dc = x.size(1) // 2
        d1 = self.act(self.c1_d(x)); r1 = self.act(self.c1_r(x) + x)
        d2 = self.act(self.c2_d(r1)); r2 = self.act(self.c2_r(r1) + r1)
        d3 = self.act(self.c3_d(r2)); r3 = self.act(self.c3_r(r2) + r2)
        r4 = self.act(self.c4(r3))
        y = self.c5(torch.cat([d1, d2, d3, r4], dim=1))
        y = y * self.cca(y)
        if self.mode != 'nosp' and self.mode != 'none':
            y = y * self.sp(y)
        if self.mode != 'nodag' and self.mode != 'none':
            y = y * self.dag(y)
        return x + y



class BlockNAF(nn.Module):
    """NAFNet-style block (Chen et al., ECCV 2022) — the literature block used as a module-level control.

    Faithful to the original design: LayerNorm2d -> 1x1 expand -> 3x3 depth-wise -> SimpleGate (channel-split
    multiply, no activation) -> SCA (global-pool channel attention) -> 1x1 project, then a SimpleGate FFN.
    Residual branches are scaled by learnable per-channel factors initialised to zero (as in the paper).
    Ops used: Conv2d / DWConv / GroupNorm / mul / add  (all on the RKNN whitelist, no softmax).
    """

    def __init__(self, c, dw_expand=3, ffn_expand=2):
        super().__init__()
        dw = int(c * dw_expand)
        hid = max(dw // 2, 8)
        ff = int(c * ffn_expand)
        # --- spatial mixing (SimpleGate + SCA) ---
        self.norm1 = nn.GroupNorm(1, c)
        self.conv1 = nn.Conv2d(c, dw, 1)
        self.conv2 = nn.Conv2d(dw, dw, 3, padding=1, groups=dw)
        self.conv3 = nn.Conv2d(hid, c, 1)
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(hid, hid, 1))
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        # --- channel mixing (SimpleGate FFN) ---
        self.norm2 = nn.GroupNorm(1, c)
        self.conv4 = nn.Conv2d(c, ff, 1)
        self.conv5 = nn.Conv2d(ff // 2, c, 1)
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))
        nn.init.zeros_(self.conv3.weight); nn.init.zeros_(self.conv3.bias)
        nn.init.zeros_(self.conv5.weight); nn.init.zeros_(self.conv5.bias)

    @staticmethod
    def _gate(t):
        a, b = t.chunk(2, dim=1)
        return a * b

    def forward(self, x):
        y = self.conv1(self.norm1(x))
        y = self.conv2(y)
        y = self._gate(y)
        y = y * self.sca(y)
        y = self.conv3(y)
        x = x + y * self.beta
        z = self._gate(self.conv4(self.norm2(x)))
        z = self.conv5(z)
        return x + z * self.gamma


class BlockCSAB(nn.Module):
    """M2 (PIR-SR): Contrast-Split Attention Block  (inspired by LISN: split + shift + DW + CCA).

    * channel split (c -> c/2 + c/2)
    * branch A: ZERO-PARAM directional shift (implemented as FIXED 3x3 kernels -> RKNN-friendly)
    * branch B: 1x1 + 3x3 depth-wise + Sigmoid  (LISN's RDB)
    * merge: concat -> 1x1
    * CCA (contrast-aware channel attention): per-channel contrast statistic (std/mean) -> 1x1-PReLU-1x1-Sigmoid
    * residual with ZERO-INIT output (starts as identity)
    All ops: Conv2d / DWConv / PReLU / Sigmoid / Add / Concat  (RKNN whitelist).
    """

    def __init__(self, c):
        super().__init__()
        h = max(c // 2, 8)
        self.h = h
        self.act = _act(c)
        # fixed directional-shift kernels (1 in different positions => shift by one pixel)
        k = torch.zeros(4, h, 1, 3, 3)
        k[0, :, 0, 1, 0] = 1.0   # shift right
        k[1, :, 0, 1, 2] = 1.0   # shift left
        k[2, :, 0, 0, 1] = 1.0   # shift down
        k[3, :, 0, 2, 1] = 1.0   # shift up
        self.register_buffer('shift_k', k)
        self.rdb1 = nn.Conv2d(h, h, 1)
        self.rdb2 = nn.Conv2d(h, h, 3, padding=1, groups=h)
        self.merge = nn.Conv2d(2 * h, c, 1)
        # CCA: contrast-aware channel attention
        self.cca = nn.Sequential(nn.Conv2d(2 * c, max(c // 4, 8), 1), _act(max(c // 4, 8)),
                                 nn.Conv2d(max(c // 4, 8), c, 1), nn.Sigmoid())
        self.out = nn.Conv2d(c, c, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=1)                       # channel split
        # branch A: zero-param directional shift (fixed kernels), summed over 4 directions
        sh = sum(torch.nn.functional.conv2d(x1, self.shift_k[i], padding=1, groups=self.h) for i in range(4)) / 4.0
        # branch B: RDB (1x1 -> 3x3 DW -> Sigmoid)
        rd = torch.sigmoid(self.rdb2(self.rdb1(x2)))
        y = self.merge(torch.cat([sh, rd], dim=1))
        # CCA: contrast statistics (std and mean) -> channel weights
        stat = torch.cat([y.std(dim=(2, 3), keepdim=True), y.mean(dim=(2, 3), keepdim=True)], dim=1)
        y = y * self.cca(stat)
        return x + self.out(self.act(y))


class DegAware(nn.Module):
    """M-D (PIR-SR): degradation-aware conditioning.

    Learns a LOCAL degradation descriptor d (no explicit kernel estimation, cf. DDR arXiv:2108.00406)
    and turns it into a bounded FiLM modulation (gamma, beta) of the backbone features:
        f <- f * (1 + tanh(gamma)) + beta
    FiLM is ZERO-INITIALISED => the model starts EXACTLY at the plain baseline.
    Ops: Conv2d / DWConv / PReLU / tanh / mul-add  (all RKNN whitelist).
    """

    def __init__(self, c, in_nc=3, nd=8):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_nc, nd, 3, padding=1), _act(nd),
            nn.Conv2d(nd, nd, 3, padding=1, groups=nd), _act(nd),
            nn.Conv2d(nd, nd, 1), _act(nd))
        self.film = nn.Conv2d(nd, 2 * c, 1)
        nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)   # identity at init

    def forward(self, lr):
        d = self.enc(lr)
        gb = self.film(d)
        gamma, beta = gb.chunk(2, dim=1)
        return torch.tanh(gamma), beta


class DegradationAwareMoEBlock(nn.Module):
    """Degradation-gated Mixture-of-Experts block (ours, cross-domain robustness).

    Motivation: different datasets have different degradations (FLIR = optical/motion blur + sensor noise,
    LLVIP = bicubic). A single fixed transformation must compromise; instead we instantiate a small pool of
    lightweight EXPERTS and let a degradation descriptor gate them per-sample:

        y = Base(x) + sum_k g_k(x) * Expert_k(x),      g = sigmoid(MLP([mean(x), std(x)]))

    * Base   : our proven foldable RFDX block (rfdxrep) -> inference = single 3x3, zero extra cost.
    * Experts: 1x1 -> DW 3x3 -> 1x1 (cheap), LAST LAYER ZERO-INITIALISED => model starts exactly at the base.
    * Gate   : global mean/std -> Linear -> PReLU -> Linear -> Sigmoid (no softmax => RKNN-friendly).
    """

    def __init__(self, c, K=2, dc=16):
        super().__init__()
        self.base = FoldableRepBlock(c)
        self.experts = nn.ModuleList()
        for _ in range(K):
            self.experts.append(nn.Sequential(
                nn.Conv2d(c, c, 1), _act(c),
                nn.Conv2d(c, c, 3, padding=1, groups=c), _act(c),
                nn.Conv2d(c, c, 1)))
        for e in self.experts:
            nn.init.zeros_(e[-1].weight); nn.init.zeros_(e[-1].bias)
        h = max(c // 4, 8)
        self.gate = nn.Sequential(nn.Linear(_gate_in(c), h), nn.PReLU(h), nn.Linear(h, K), nn.Sigmoid())
        nn.init.zeros_(self.gate[-2].weight); nn.init.zeros_(self.gate[-2].bias)

    def forward(self, x):
        exp_res = _os.environ.get('IR_EXP_RES', '0') == '1'      # #12 optional expert residual
        y = self.base(x)
        s = deg_descriptor(x)                              # #9 rich degradation descriptor
        g = self.gate(s)                                   # (B, K) per-sample expert weights
        for k, e in enumerate(self.experts):
            y = y + g[:, k].view(-1, 1, 1, 1) * ((x + e(x)) if exp_res else e(x))   # #12 expert residual
        self.last_gate = g.detach()
        return y


class MultiScaleDenseBlock(nn.Module):
    """M1 (PIR-SR): Multi-Scale Reparameterizable Distillation Block.

    Train-time branches — **every one of them folds EXACTLY into a single 3x3 conv**:
        1x1 (learnable) | identity | 1x3 + 3x1 (asymmetric, learnable) |
        3x3 average pooling (per-channel, learnable weight) |
        Sobel-x / Sobel-y / Laplacian (per-channel, learnable weight)
    Deploy: ONE 3x3 conv  => zero extra inference cost/params, multi-scale + directional priors.
    RFDX's progressive distillation + recirculation + dual-stat gate are kept unchanged.
    (5x5 / dilated branches are deliberately NOT used: they cannot fold into a 3x3.)
    """

    def __init__(self, c, gc=32, dc=16):
        super().__init__()
        self.c = c
        self.act = _act(c); self.act_dc = _act(dc)
        self.c1_d = nn.Conv2d(c, dc, 1); self.c2_d = nn.Conv2d(c, dc, 1); self.c3_d = nn.Conv2d(c, dc, 1)
        self.c1_r = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.c2_r = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.c3_r = nn.Conv2d(c, c, 3, padding=1, bias=False)
        # ---- foldable train-time branches ----
        self.b1x1 = nn.Conv2d(c, c, 1, bias=False); nn.init.zeros_(self.b1x1.weight)
        self.b13 = nn.Conv2d(c, c, (1, 3), padding=(0, 1), bias=False); nn.init.zeros_(self.b13.weight)
        self.b31 = nn.Conv2d(c, c, (3, 1), padding=(1, 0), bias=False); nn.init.zeros_(self.b31.weight)
        self.w_pool = nn.Parameter(torch.zeros(1, c, 1, 1))      # per-channel avg-pool weight
        self.w_edge = nn.Parameter(torch.zeros(3, c, 1, 1, 1))   # sobel-x, sobel-y, laplacian
        sx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 4.0
        sy = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]) / 4.0
        lp = torch.tensor([[0., -1., 0.], [-1., 4., -1.], [0., -1., 0.]]) / 4.0
        self.register_buffer('edge_k', torch.stack([sx, sy, lp]).unsqueeze(1).unsqueeze(1).repeat(1, c, 1, 1, 1))
        self.register_buffer('pool_k', torch.full((c, 1, 3, 3), 1.0 / 9.0))
        self.mix = nn.Conv2d(dc, c, 1); nn.init.zeros_(self.mix.weight); nn.init.zeros_(self.mix.bias)
        self.c4 = nn.Conv2d(c, c, 3, padding=1)
        self.c5 = nn.Conv2d(c + dc * 3, c, 1)
        self.gate = nn.Sequential(nn.Conv2d(c * 2, c // 4, 1), _act(c // 4), nn.Conv2d(c // 4, c, 1), nn.Sigmoid())
        k = torch.tensor([[[[0., -1., 0.], [-1., 4., -1.], [0., -1., 0.]]]])
        self.register_buffer('hf', k.repeat(c, 1, 1, 1))

    def _fold(self, w3, b1x1, b13, b31):
        c = w3.size(0)
        out = w3.clone()
        eye = torch.eye(c, device=w3.device, dtype=w3.dtype).view(c, c, 1, 1)
        out[:, :, 1:2, 1:2] = out[:, :, 1:2, 1:2] + eye                       # identity
        out = out + torch.nn.functional.pad(b1x1, (1, 1, 1, 1))               # 1x1
        out = out + torch.nn.functional.pad(b13, (0, 0, 1, 1))                # 1x3 -> middle row
        out = out + torch.nn.functional.pad(b31, (1, 1, 0, 0))                # 3x1 -> middle col
        idx = torch.arange(c, device=w3.device)                               # per-channel (grouped) branches
        per = (self.w_edge.view(3, c, 1) * self.edge_k.view(3, c, 9)).sum(0).view(c, 3, 3)             + self.w_pool.view(c, 1, 1) * self.pool_k.view(c, 3, 3)
        out = out.clone()
        out[idx, idx] = out[idx, idx] + per
        return out

    def _conv(self, x, conv3, b1x1, b13, b31):
        if self.training:
            e = self.w_pool.view(1, -1, 1, 1) * torch.nn.functional.avg_pool2d(x, 3, 1, 1)                 + sum(self.w_edge[i].view(1, -1, 1, 1) *
                      torch.nn.functional.conv2d(x, self.edge_k[i], padding=1, groups=self.c)
                      for i in range(3))
            return self.act(conv3(x) + b1x1(x) + b13(x) + b31(x) + x + e)
        return self.act(torch.nn.functional.conv2d(
            x, self._fold(conv3.weight, b1x1.weight, b13.weight, b31.weight), padding=1))

    def forward(self, x):
        d1 = self.act_dc(self.c1_d(x)); r1 = self._conv(x, self.c1_r, self.b1x1, self.b13, self.b31)
        d2 = self.act_dc(self.c2_d(r1)); r2 = self._conv(r1 + self.mix(d1), self.c2_r, self.b1x1, self.b13, self.b31)
        d3 = self.act_dc(self.c3_d(r2)); r3 = self._conv(r2 + self.mix(d2), self.c3_r, self.b1x1, self.b13, self.b31)
        r4 = self.act(self.c4(r3))
        y = self.c5(torch.cat([d1, d2, d3, r4], dim=1))
        stat = y.mean(dim=(2, 3), keepdim=True) + y.std(dim=(2, 3), keepdim=True)
        hfe = torch.nn.functional.conv2d(y, self.hf, padding=1, groups=y.size(1)).pow(2).mean(dim=(2, 3), keepdim=True)
        g = self.gate(torch.cat([stat, hfe], dim=1))
        return x + g * y


class FoldableRepEdgeBlock(nn.Module):
    """RFDX + FLIR-paper-style recipe: edge-oriented re-parameterisation (ECBSR/RepFADN) + SE attention.

    Train-time branches per main 3x3 conv: 1x1 + identity + FIXED edge kernels (Sobel-x, Sobel-y, Laplacian)
    with learnable per-channel scalars. Every branch is a 3x3 (or padable 1x1) kernel, so deploy folds EXACTLY:
        W_deploy = W3 + pad(W1) + I_center + sum_e w_e * E_e
    An SE (squeeze-excitation) gate on the fused output adds cheap channel attention (RepFADN-style).
    Recirculation + dual-stat gate of RFDX are kept.
    """

    def __init__(self, c, gc=32, dc=None):
        dc = _dc_for(c, dc)
        super().__init__()
        self.c = c
        self.act = _act(c); self.act_dc = _act(dc)
        self.c1_d = nn.Conv2d(c, dc, 1); self.c2_d = nn.Conv2d(c, dc, 1); self.c3_d = nn.Conv2d(c, dc, 1)
        self.c1_r = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.c2_r = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.c3_r = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.r1_1x1 = nn.Conv2d(c, c, 1, bias=False); _rep_1x1(self.r1_1x1)
        self.r2_1x1 = nn.Conv2d(c, c, 1, bias=False); _rep_1x1(self.r2_1x1)
        self.r3_1x1 = nn.Conv2d(c, c, 1, bias=False); _rep_1x1(self.r3_1x1)
        # learnable per-channel weights for the three fixed edge kernels (zero-init -> identity start)
        self.w_edge = nn.Parameter(torch.zeros(3, c, 1, 1, 1))
        sx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 4.0
        sy = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]) / 4.0
        lp = torch.tensor([[0., -1., 0.], [-1., 4., -1.], [0., -1., 0.]]) / 4.0
        self.register_buffer('edge_k', torch.stack([sx, sy, lp]).unsqueeze(1).unsqueeze(1).repeat(1, c, 1, 1, 1))  # (3,c,1,3,3)
        self.mix = nn.Conv2d(dc, c, 1); nn.init.zeros_(self.mix.weight); nn.init.zeros_(self.mix.bias)
        self.c4 = nn.Conv2d(c, c, 3, padding=1)
        self.c5 = nn.Conv2d(c + dc * 3, c, 1)
        self.gate = nn.Sequential(nn.Conv2d(c * 2, c // 4, 1), _act(c // 4), nn.Conv2d(c // 4, c, 1), nn.Sigmoid())
        self.se = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, max(c // 4, 8), 1), nn.PReLU(max(c // 4, 8)),
                                nn.Conv2d(max(c // 4, 8), c, 1), nn.Sigmoid())
        k = torch.tensor([[[[0., -1., 0.], [-1., 4., -1.], [0., -1., 0.]]]])
        self.register_buffer('hf', k.repeat(c, 1, 1, 1))

    def _fold(self, w3, w1):
        out = w3.clone()
        eye = torch.eye(w3.size(0), device=w3.device, dtype=w3.dtype).view(w3.size(0), w3.size(1), 1, 1)
        out[:, :, 1:2, 1:2] = out[:, :, 1:2, 1:2] + eye
        out = out + torch.nn.functional.pad(w1, (1, 1, 1, 1))
        # edge branches are per-channel (grouped) kernels -> place them on the DIAGONAL of the full kernel
        c = out.size(0)
        kern = (self.w_edge.view(3, c, 1) * self.edge_k.view(3, c, 9)).sum(0).view(c, 3, 3)  # (c,3,3)
        idx = torch.arange(c, device=out.device)
        out = out.clone()
        out[idx, idx] = out[idx, idx] + kern
        return out

    def _conv(self, x, conv3, conv1):
        if self.training:
            e = sum(self.w_edge[i].view(1, -1, 1, 1) *
                    torch.nn.functional.conv2d(x, self.edge_k[i], padding=1, groups=self.c) for i in range(3))
            return self.act(conv3(x) + conv1(x) + x + e)
        return self.act(torch.nn.functional.conv2d(x, self._fold(conv3.weight, conv1.weight), padding=1))

    def forward(self, x):
        d1 = self.act_dc(self.c1_d(x)); r1 = self._conv(x, self.c1_r, self.r1_1x1)
        d2 = self.act_dc(self.c2_d(r1)); r2 = self._conv(r1 + self.mix(d1), self.c2_r, self.r2_1x1)
        d3 = self.act_dc(self.c3_d(r2)); r3 = self._conv(r2 + self.mix(d2), self.c3_r, self.r3_1x1)
        r4 = self.act(self.c4(r3))
        y = self.c5(torch.cat([d1, d2, d3, r4], dim=1))
        stat = y.mean(dim=(2, 3), keepdim=True) + y.std(dim=(2, 3), keepdim=True)
        hfe = torch.nn.functional.conv2d(y, self.hf, padding=1, groups=y.size(1)).pow(2).mean(dim=(2, 3), keepdim=True)
        g = self.gate(torch.cat([stat, hfe], dim=1))
        y = y * self.se(y)                                   # SE channel attention (RepFADN-style)
        return x + g * y


class FoldableRepBlock(nn.Module):
    """RFDX with train-time re-parameterizable branches (EXACT deploy-time fusion).

    Each main 3x3 residual conv is augmented in training with parallel 1x1 and identity branches.
    All branches are linear & channel-preserving, so at inference they fold EXACTLY into the single
    original 3x3 kernel:  W = W3 + pad(W1) + I_center ,  b = b3 (bias-free here).
    => same inference cost/params as the plain RFDX block, richer training signal (RepVGG/ECBSR recipe).
    Recirculation + dual-stat gate of RFDX are kept unchanged.
    """

    def __init__(self, c, gc=32, dc=None):
        dc = _dc_for(c, dc)
        super().__init__()
        self.c = c
        self.act = _act(c)
        self.act_dc = _act(dc)
        self.c1_d = nn.Conv2d(c, dc, 1)
        self.c2_d = nn.Conv2d(c, dc, 1)
        self.c3_d = nn.Conv2d(c, dc, 1)
        self.c1_r = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.c2_r = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.c3_r = nn.Conv2d(c, c, 3, padding=1, bias=False)
        # train-time parallel branches (folded away at deploy)
        self.r1_1x1 = nn.Conv2d(c, c, 1, bias=False); _rep_1x1(self.r1_1x1)
        self.r2_1x1 = nn.Conv2d(c, c, 1, bias=False); _rep_1x1(self.r2_1x1)
        self.r3_1x1 = nn.Conv2d(c, c, 1, bias=False); _rep_1x1(self.r3_1x1)
        self.mix = nn.Conv2d(dc, c, 1)
        nn.init.zeros_(self.mix.weight); nn.init.zeros_(self.mix.bias)
        self.c4 = nn.Conv2d(c, c, 3, padding=1)
        self.c5 = nn.Conv2d(c + dc * 3, c, 1)
        self.gate = nn.Sequential(nn.Conv2d(c * 2, c // 4, 1), _act(c // 4), nn.Conv2d(c // 4, c, 1), nn.Sigmoid())
        k = torch.tensor([[[[0., -1., 0.], [-1., 4., -1.], [0., -1., 0.]]]])
        self.register_buffer('hf', k.repeat(c, 1, 1, 1))

    @staticmethod
    def _fold(w3, w1):
        out = w3.clone()
        out[:, :, 1:2, 1:2] = out[:, :, 1:2, 1:2] + torch.eye(w3.size(0), device=w3.device, dtype=w3.dtype).view(w3.size(0), w3.size(1), 1, 1)
        return out + torch.nn.functional.pad(w1, (1, 1, 1, 1))

    def _w(self):
        if self.training:
            return self.c1_r.weight, self.r1_1x1.weight
        return self._fold(self.c1_r.weight, self.r1_1x1.weight), None

    def forward(self, x):
        def block(x, conv3, conv1):
            if self.training:
                main = conv3(x) + conv1(x) + x
            else:
                folded = self._fold(conv3.weight, conv1.weight)
                main = torch.nn.functional.conv2d(x, folded, padding=1)
            return self.act(main)

        d1 = self.act_dc(self.c1_d(x)); r1 = block(x, self.c1_r, self.r1_1x1)
        d2 = self.act_dc(self.c2_d(r1)); r2 = block(r1 + self.mix(d1), self.c2_r, self.r2_1x1)
        d3 = self.act_dc(self.c3_d(r2)); r3 = block(r2 + self.mix(d2), self.c3_r, self.r3_1x1)
        r4 = self.act(self.c4(r3))
        y = self.c5(torch.cat([d1, d2, d3, r4], dim=1))
        stat = y.mean(dim=(2, 3), keepdim=True) + y.std(dim=(2, 3), keepdim=True)
        hfe = torch.nn.functional.conv2d(y, self.hf, padding=1, groups=y.size(1)).pow(2).mean(dim=(2, 3), keepdim=True)
        g = self.gate(torch.cat([stat, hfe], dim=1))
        return x + g * y




class FoldableDepthwiseBlock(FoldableRepBlock):
    """#6 deploy-oriented variant: same RFDX topology but the three main 3x3 convs become
    depthwise-3x3 + pointwise-1x1, cutting FLOPs roughly 5-8x. Not foldable into one kernel,
    so it trades the re-parameterisation trick for real deployment budget on RK3588.
    """

    def __init__(self, c, gc=32, dc=None):
        super().__init__(c, gc=gc, dc=dc)
        self.c1_r = nn.Sequential(nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False),
                                  nn.Conv2d(c, c, 1, bias=False))
        self.c2_r = nn.Sequential(nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False),
                                  nn.Conv2d(c, c, 1, bias=False))
        self.c3_r = nn.Sequential(nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False),
                                  nn.Conv2d(c, c, 1, bias=False))
        self.r1_1x1 = nn.Conv2d(c, c, 1, bias=False)
        self.r2_1x1 = nn.Conv2d(c, c, 1, bias=False)
        self.r3_1x1 = nn.Conv2d(c, c, 1, bias=False)

    def forward(self, x):
        def blk(x, conv, conv1):
            main = conv(x) + conv1(x) + x          # DW version is not foldable -> always multi-branch
            return self.act(main)

        d1 = self.act_dc(self.c1_d(x)); r1 = blk(x, self.c1_r, self.r1_1x1)
        d2 = self.act_dc(self.c2_d(r1)); r2 = blk(r1 + self.mix(d1), self.c2_r, self.r2_1x1)
        d3 = self.act_dc(self.c3_d(r2)); r3 = blk(r2 + self.mix(d2), self.c3_r, self.r3_1x1)
        r4 = self.act(self.c4(r3))
        y = self.c5(torch.cat([d1, d2, d3, r4], dim=1))
        return self.act(y + x)
class RecirculatedDistillBlock(nn.Module):
    """(rfdx) creative RFDN-derived backbone block: FDC progressive distillation + DISTILLATE RECIRCULATION —
    each 1x1 distillate is softly fed back into the following 3x3 residual stage (feedback-FDC; RFDN does not do
    this). Final fusion gate is dual-stat: CCA contrast + HF-energy statistic (frequency-aware)."""
    def __init__(self, c):
        super().__init__()
        dc = c // 2
        self.c1_d = nn.Conv2d(c, dc, 1); self.c2_d = nn.Conv2d(c, dc, 1); self.c3_d = nn.Conv2d(c, dc, 1)
        self.c1_r = nn.Conv2d(c, c, 3, padding=1); self.c2_r = nn.Conv2d(c, c, 3, padding=1); self.c3_r = nn.Conv2d(c, c, 3, padding=1)
        self.c4 = nn.Conv2d(c, dc, 3, padding=1)
        self.mix = nn.Conv2d(dc, c, 1)                              # recirculation up-mix (soft feedback)
        nn.init.zeros_(self.mix.weight); nn.init.zeros_(self.mix.bias)
        self.c5 = nn.Conv2d(dc * 4, c, 1)
        self.gate = nn.Sequential(nn.Conv2d(c * 2, c // 4, 1), _act(c // 4), nn.Conv2d(c // 4, c, 1), nn.Sigmoid())
        self.act = nn.PReLU()
        hf = torch.tensor([[[[0., -1., 0.], [-1., 4., -1.], [0., -1., 0.]]]])
        self.register_buffer('hf', hf.repeat(c, 1, 1, 1))
    def forward(self, x):
        c = x.size(1); dc = c // 2
        d1 = self.act(self.c1_d(x))
        r1 = self.act(self.c1_r(x + self.mix(d1)) + x)
        d2 = self.act(self.c2_d(r1))
        r2 = self.act(self.c2_r(r1 + self.mix(d2)) + r1)
        d3 = self.act(self.c3_d(r2))
        r3 = self.act(self.c3_r(r2 + self.mix(d3)) + r2)
        r4 = self.act(self.c4(r3))
        y = self.c5(torch.cat([d1, d2, d3, r4], dim=1))
        stat = y.mean(dim=(2, 3), keepdim=True) + y.std(dim=(2, 3), keepdim=True)
        hfe = F.conv2d(y, self.hf, padding=1, groups=c).pow(2).mean(dim=(2, 3), keepdim=True)
        g = self.gate(torch.cat([stat, hfe], dim=1))
        return x + y * g


class BlockRadCons(nn.Module):
    """(radcons) radiometric-consistency block, embedded into the mac_b backbone: an invertible radiometric
    domain transform (monotone radiance R, learnable gamma/bias) with a residual radiance-preserving branch.
    Numerically stabilized: gamma is bounded, the rad-domain residual is bounded, and the inverse is clamped,
    so the branch energy stays comparable to the other branches (no exp blow-up)."""
    def __init__(self, c):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(0.0))
        self.bias = nn.Parameter(torch.tensor(0.0))
        self.conv = nn.Conv2d(c, c, 3, padding=1, groups=c)
        self.f = nn.Conv2d(c, c, 1)
    def forward(self, x):
        g = torch.sigmoid(self.gamma) * 0.9 + 0.1          # gamma in (0.1, 1.0): bounded
        b = torch.tanh(self.bias) * 0.1                     # bias in (-0.1, 0.1)
        r = torch.log1p(x.clamp(0, 1)) * g + b              # radiance domain (bounded)
        r = r + torch.tanh(self.conv(r)) * 0.1              # bounded rad-domain residual
        x_out = torch.expm1(torch.clamp((r - b) / g, 0, 1.2))  # inverse, clamped (max ~ e^1.2-1 ~ 2.3)
        x_out = torch.clamp(x_out, 0, 3)
        return x + self.f(x_out)


class BlockDAG(nn.Module):
    """(dag) degradation-aware block: estimates a degradation code (pooled) and applies it as a channel gate/FiLM (C1)."""
    def __init__(self, c):
        super().__init__()
        self.distil = nn.Conv2d(c, c // 2, 1)
        self.res = nn.Conv2d(c, c // 2, 3, padding=1)
        self.deg = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, c // 4, 1), _act(c // 4), nn.Conv2d(c // 4, c, 1), nn.Sigmoid())
        self.f = nn.Conv2d(c, c, 1)
    def forward(self, x):
        d = self.distil(x)
        r = self.res(x)
        y = torch.cat([d, r], dim=1) * self.deg(x)
        return x + self.f(y)


class BlockPHY(nn.Module):
    """(phy) physics-prior-guided module: enforces thermal radiometric prior via a monotonic temperature-intensity
    consistency recalibration (removes non-uniformity/null-pattern bias) + spectral-phase enhancement (C1+C2)."""
    def __init__(self, c):
        super().__init__()
        self.distil = nn.Conv2d(c, c // 2, 1)
        self.res = nn.Conv2d(c, c // 2, 3, padding=1)
        # radiometric calibration: per-pixel gain/bias (nonuniformity correction) + thermal monotonic prior
        self.gain = nn.Conv2d(c // 2, c // 2, 1)
        self.bias = nn.Conv2d(c // 2, c // 2, 1)
        # spectral-phase enhancement (C2)
        self.fr = BlockFR(c // 2)
        self.prompt = nn.Sequential(nn.Conv2d(c, c, 3, padding=1, groups=c), _act(c), nn.Conv2d(c, c, 1), nn.Sigmoid())
        self.f = nn.Conv2d(c, c, 1)
    def forward(self, x):
        c = x.size(1); dc = c // 2
        d = self.distil(x)
        r = self.res(x)
        # physical radiometric recalibration: y = gain * d + bias (gain>0 via exp -> monotonic temp->intensity prior)
        g = torch.exp(self.gain(d))
        b = self.bias(d)
        r_cal = self.fr(d * g + b)              # calibrated high-freq branch (spectral enhancement)
        y = torch.cat([r_cal, self.fr(r)], dim=1)
        y = y * self.prompt(y)
        return x + self.f(y)


class BlockGSP(nn.Module):
    """(gsp) gradient-spectral dual-prior block (self-made innovation): a zero-parameter Sobel gradient prior
    (high-frequency structure, MODE-GAN/Scharr-inspired) is fused with the fractional-spectral (fr) prior
    (band-preserving, L^alpha theory) into the feature stream. Stable: additive, cheap, mostly-backbone."""
    def __init__(self, c):
        super().__init__()
        self.fr = BlockFR(c)
        self.fuse = nn.Conv2d(c, c, 1)
        self.w_g = nn.Parameter(torch.tensor(0.1))          # gradient-prior strength (learnable, small init)
        # zero-parameter Sobel kernels (depthwise per-channel: [c,1,3,3])
        kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]])
        self.register_buffer('kx', (kx / 4.0).view(1, 1, 3, 3).repeat(c, 1, 1, 1))
        self.register_buffer('ky', (ky / 4.0).view(1, 1, 3, 3).repeat(c, 1, 1, 1))
    def forward(self, x):
        gx = F.conv2d(x, self.kx, padding=1, groups=x.size(1))
        gy = F.conv2d(x, self.ky, padding=1, groups=x.size(1))
        g = torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-8)         # per-channel gradient magnitude
        s = self.fr(x)
        return x + self.fuse(s + self.w_g * g)


class BlockFBD(nn.Module):
    """(fbd) band-complementarity block (self-made innovation #2): partition-of-unity frequency decomposition.
    x = Lo + Hi exactly (Lo = 3x3 average pool, Hi = x - Lo), so the smooth low-frequency thermal-radiation field
    and the high-frequency complement are processed separately and re-mixed with learned weights.
    Theory: identity partition in frequency; apt for IR (energy concentrated in low bands). Stable and cheap."""
    def __init__(self, c):
        super().__init__()
        self.lo = nn.Conv2d(c, c, 1)                                  # smooth op on low band
        self.hi = nn.Conv2d(c, c, 3, padding=1, groups=c)             # detail op on high band
        self.w = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.f = nn.Conv2d(c, c, 1)
    def forward(self, x):
        lo = F.avg_pool2d(x, 3, stride=1, padding=1)                  # low-pass (zero-param)
        hi = x - lo                                                   # high-pass complement (partition of unity)
        y = self.w[0] * self.lo(lo) + self.w[1] * self.hi(hi)
        return x + self.f(y)


class BlockWav(nn.Module):
    """(wav) Haar-wavelet band block (literature-backed: DDMN/DDSRNet wavelet SR; FABIMNet warns against fully
    independent band modeling -> high-frequency bands share one refinement conv). Conv-only (fixed haar kernels)."""
    def __init__(self, c):
        super().__init__()
        self.ll = nn.Conv2d(c, c, 1)                                # low-freq (thermal field) smooth branch
        self.hf = nn.Conv2d(c, c, 3, padding=1, groups=c)           # shared HF refinement (LH/HL/HH)
        self.f = nn.Conv2d(c, c, 1)
        w = torch.tensor([[[[1., 1.], [1., 1.]]], [[[1., 1.], [-1., -1.]]], [[[1., -1.], [1., -1.]]], [[[1., -1.], [-1., 1.]]]]) / 2.0
        self.register_buffer('w', w)                                # orthonormal Haar (4,1,2,2)
    def forward(self, x):
        B, c, H, W = x.shape
        y = F.conv2d(x.reshape(B * c, 1, H, W), self.w, stride=2).view(B, c, 4, H // 2, W // 2)
        ll, lh, hl, hh = y[:, :, 0], y[:, :, 1], y[:, :, 2], y[:, :, 3]
        ll2 = self.ll(ll)
        lh2, hl2, hh2 = self.hf(lh), self.hf(hl), self.hf(hh)
        z = torch.stack([ll2, lh2, hl2, hh2], dim=2).reshape(B * c, 4, H // 2, W // 2)
        x2 = F.conv_transpose2d(z, self.w, stride=2).view(B, c, H, W)
        return x + self.f(x2)


class BlockPSB(nn.Module):
    """(psb) Physical Spectral-Budget block (hallucination control): bounds recovered HF by a learned budget
    derived from the LR low-band spectral envelope (IR 1/f^alpha prior; wavelet-domain artifact control
    CVPR'24/FDMM-backed). DWT -> HF subbands shrunk by budget -> iDWT. Blocks hallucination in-structure."""
    def __init__(self, c):
        super().__init__()
        self.ll = nn.Conv2d(c, c, 1)
        self.hf = nn.Conv2d(c, c, 3, padding=1, groups=c)
        self.budget = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, c // 4, 1), nn.PReLU(c // 4), nn.Conv2d(c // 4, c, 1), nn.Sigmoid())
        self.f = nn.Conv2d(c, c, 1)
        w = torch.tensor([[[[1., 1.], [1., 1.]]], [[[1., 1.], [-1., -1.]]], [[[1., -1.], [1., -1.]]], [[[1., -1.], [-1., 1.]]]]) / 2.0
        self.register_buffer('w', w)
    def forward(self, x):
        B, c, H, W = x.shape
        y = F.conv2d(x.reshape(B * c, 1, H, W), self.w, stride=2).view(B, c, 4, H // 2, W // 2)
        ll, lh, hl, hh = y[:, :, 0], y[:, :, 1], y[:, :, 2], y[:, :, 3]
        ll2 = self.ll(ll)
        g = self.budget(ll)                       # per-channel HF-trust budget in (0,1) from low-band
        lh2 = g * torch.tanh(self.hf(lh))
        hl2 = g * torch.tanh(self.hf(hl))
        hh2 = g * torch.tanh(self.hf(hh))
        z = torch.stack([ll2, lh2, hl2, hh2], dim=2).reshape(B * c, 4, H // 2, W // 2)
        x2 = F.conv_transpose2d(z, self.w, stride=2).view(B, c, H, W)
        return x + self.f(x2)


class BlockGSL(nn.Module):
    """(gsl) Gradient-Structure gate (our innovation add-on, safe-by-construction): a zero-parameter Sobel
    structure prior gates the features; a learnable scalar (init 0 = exact identity) controls the injection, so it
    can never hurt the baseline early and only learns to add structure-aware modulation if useful.
    Motivation: edge/structure priors help SR (Scharr/Sobel guidance e.g. MODE-GAN); zero-param keeps it light."""
    def __init__(self, c):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(1))
        self.f = nn.Conv2d(c, c, 1)
        kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 4.0
        ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]) / 4.0
        self.register_buffer('kx', kx.view(1, 1, 3, 3).repeat(c, 1, 1, 1))
        self.register_buffer('ky', ky.view(1, 1, 3, 3).repeat(c, 1, 1, 1))
    def forward(self, x):
        c = x.size(1)
        gx = F.conv2d(x, self.kx, padding=1, groups=c)
        gy = F.conv2d(x, self.ky, padding=1, groups=c)
        g = torch.sigmoid((gx * gx + gy * gy).mean(dim=1, keepdim=True))     # structure-strength map
        return x + self.w * self.f(x * g)


_BLOCKS = dict(cs=BlockCS, dw=BlockDW, gs=BlockGS, ms=BlockMS, rp=BlockRP, sl=BlockSL,
               gf=BlockGF, kr=BlockKR, wp=BlockWP, mo=BlockMO, dy=BlockDY,
               ode=BlockODE, gc=BlockGC, hy=BlockHY, ssm=BlockSSM, lp=BlockLP, d2=BlockD2,
               rg=BlockRG, cd=BlockCD, od=BlockOD, ma=BlockMA, mp=BlockMP, fr=BlockFR,
               ga=BlockGA, mc=BlockMC, lr=BlockLR, sa=BlockSA, eq=BlockEQ, pr=BlockPR, pg=BlockPG, bd=BlockBD,
               msd=BlockMSD, af=BlockAF, dr13=lambda c: BlockDR(c, 1 / 3), dr12=lambda c: BlockDR(c, 0.5), dr23=lambda c: BlockDR(c, 2 / 3),
               ird=BlockIRD, ird_nose=lambda c: BlockIRD(c, 'nose'), ird_norank=lambda c: BlockIRD(c, 'norank'),
               rd=BlockRD, sd=BlockSD, ib=BlockIB, ibd=BlockIBD, lk=BlockLK, sp=BlockSP, dag=BlockDAG,
               phy=BlockPHY, radcons=BlockRadCons, gsp=BlockGSP, fbd=BlockFBD, wav=BlockWav, psb=BlockPSB, rfdx=RecirculatedDistillBlock, gsl=BlockGSL,
               veg=BlockVEG, vsg=BlockVSG, crm=BlockCRM, rfdxrep=FoldableRepBlock, rfdxdw=FoldableDepthwiseBlock, naf=BlockNAF, rfdxrepe=FoldableRepEdgeBlock, msrdb=MultiScaleDenseBlock, degmoe=DegradationAwareMoEBlock, csab=BlockCSAB,
               rfds=BlockRFDS, rfds_nosp=lambda c: BlockRFDS(c, 'nosp'), rfds_nodag=lambda c: BlockRFDS(c, 'nodag'), rfds_none=lambda c: BlockRFDS(c, 'none'))


@ARCH_REGISTRY.register()
class FDRNet(nn.Module):
    def __init__(self, in_nc=3, nf=48, num_blocks=4, out_nc=3, upscale=4, block_type='cs', block_types=None, dual_reg=False, phys_prior=False, tscf=False, cmf='', hfls=False, hfls2=False, hfls3=False, hfls4=False, hfls4_noquant=False, hfls4_nolip=False, degaware=False, sghf=False, baseline=None, baseline_width=1.0, deg_sem=None, deg_sem_codes=64, heavy_guided=None, aux_w=0.05, degcodec=False, degcodec_true=False, degcodec_coarse=False, use_clip=False, con_weight=0.1,
                 clip_path='D:/basicSR/pretrained/clip_vitb32'):
        super().__init__()
        self.in_nc = in_nc; self.nf = nf; self.upscale = upscale
        self.phys_prior = phys_prior
        # learnable radiometric estimate head (embedded, not external) so phys_loss is an in-network term
        # ONLY created when phys_prior=True, so non-physics runs keep the exact architecture of their checkpoints.
        if phys_prior:
            self.rad_head = nn.Sequential(nn.Conv2d(out_nc, out_nc, 1), nn.PReLU(out_nc), nn.Conv2d(out_nc, out_nc, 1), nn.Sigmoid())
        self.fea_conv = nn.Conv2d(in_nc, nf, 3, padding=1)
        if block_types is not None and len(block_types) > 0:
            # COMBINED innovation: stack different mechanism blocks in sequence
            self.block_types = list(block_types)
            self.num_blocks = len(self.block_types)
        else:
            self.block_types = [str(block_type)] * int(num_blocks)
            self.num_blocks = int(num_blocks)
        self.blocks = nn.ModuleList([_BLOCKS[t](nf) for t in self.block_types])
        self.c = nn.Sequential(nn.Conv2d(nf * self.num_blocks, nf, 1), _act(nf))
        self.lr_conv = nn.Conv2d(nf, nf, 3, padding=1)
        self.upsampler = nn.Sequential(nn.Conv2d(nf, out_nc * upscale * upscale, 3, padding=1), nn.PixelShuffle(upscale))
        # dual-regression (DRL): learn an inverse HR->LR mapping to constrain the SR mapping space
        self.dual_reg = dual_reg
        self.aux_loss = None
        self.dual_head = nn.Conv2d(out_nc, in_nc, 3, stride=upscale, padding=1) if dual_reg else None
        # ---- cross-modal (visible-guided) support: only built when veg/vsg requested (pure-IR path untouched) ----
        self.cross_modal = any(t in ('veg', 'vsg') for t in self.block_types) or bool(tscf) or bool(cmf) or bool(sghf)
        self.use_crm = ('crm' in self.block_types)
        self.tscf_on = bool(tscf) or (cmf == 'tscf')
        self.caf_on = (cmf == 'caf')
        if self.cross_modal:
            self.ctx_builder = CrossModalContext(nf, vi_in=in_nc)
        if self.tscf_on:
            self.tscf = CrossModalFusion(nf)
        self.baseline = baseline
        if baseline:
            self.tscf = build_baseline(baseline, nf, width=float(baseline_width))
            self.tscf_on = True
            self.baseline_on = True
        # ---- N-01..N-05 degradation-semantics modules (comma separated list) ----
        self.deg_sem = deg_sem
        self.ds_mods = torch.nn.ModuleDict()
        if deg_sem:
            from basicsr.archs.deg_sem_modules import VQDegCodebook, FiLM, DegCrossAttn, ContrastiveCode, WideMoE
            names = [n.strip() for n in str(deg_sem).split(',') if n.strip()]
            self.ds_names = names
            self.ds_code_dim = 64
            for nm in names:
                if nm == 'vq':
                    self.ds_mods['vq'] = VQDegCodebook(in_dim=2, code_dim=self.ds_code_dim, n_codes=int(deg_sem_codes))
                elif nm == 'film':
                    self.ds_mods['film'] = FiLM(code_dim=self.ds_code_dim, nf=nf)
                elif nm == 'xattn':
                    self.ds_mods['xattn'] = DegCrossAttn(code_dim=self.ds_code_dim, nf=nf)
                elif nm == 'moe8':
                    self.ds_mods['moe8'] = WideMoE(nf=nf, n_experts=8)
                elif nm == 'contrast':
                    self.ds_mods['contrast'] = ContrastiveCode(dim=self.ds_code_dim)
        # ---- N-06 heavy criterion-conditioned cross-modal attention ----
        self.heavy_guided = heavy_guided
        if heavy_guided == 'cghf_v2':
            from basicsr.archs.deg_sem_modules import CGHFv2
            self.tscf = CGHFv2(nf=nf)
            self.tscf_on = True
        self.aux_w = float(aux_w)
        self.sghf_on = bool(sghf)
        if self.sghf_on:
            self.tscf = StructureGatedHFFusion(nf)
        self.hfls_on = bool(hfls)
        if self.hfls_on:
            self.hfls = HFLS(nf, upscale=upscale, out_nc=out_nc)
        self.hfls2_on = bool(hfls2)
        if self.hfls2_on:
            self.hfls = HFLS2(nf, upscale=upscale, out_nc=out_nc)
        self.degcodec_on = bool(degcodec)
        self.con_weight = float(con_weight)
        if self.degcodec_on:
            clip_enc = None
            clip_dim = 768
            if use_clip:
                from transformers import CLIPVisionModel
                clip_enc = CLIPVisionModel.from_pretrained(clip_path, local_files_only=True)
                clip_enc.eval()
                for p_ in clip_enc.parameters():
                    p_.requires_grad_(False)
                clip_dim = clip_enc.config.hidden_size
            if degcodec_coarse:
                self.degcodec = DegCodecCoarse(nf, clip_encoder=clip_enc, use_clip=use_clip)
            else:
                self.degcodec = DegCodec(nf, clip_encoder=clip_enc, clip_dim=clip_dim, use_clip=use_clip)
            self.degcodec.true_code_injection = bool(degcodec_true)
        self.degaware_on = bool(degaware)
        if self.degaware_on:
            self.da = DegAware(nf, in_nc=in_nc)
        self.hfls3_on = bool(hfls3)
        if self.hfls3_on:
            self.hfls = HFLS3(nf, upscale=upscale, out_nc=out_nc)
        self.hfls4_on = bool(hfls4)
        if self.hfls4_on:
            self.hfls = HFLS4(nf, upscale=upscale, out_nc=out_nc,
                              use_quant=not hfls4_noquant, use_lip=not hfls4_nolip)
        if self.caf_on:
            self.tscf = CrossModalFusionSimple(nf)
    def switch_to_deploy(self):
        """One-time fold: cache the merged 3x3 kernel in every foldable block (export-time operation)."""
        n = 0
        for m in self.modules():
            if hasattr(m, '_fold'):
                m.deploy = True
                m._w_deploy = None
                n += 1
        return n

    def forward(self, x, vi=None, deg=None, clip_feat=None):
        B, C, H, W = x.shape
        if C != self.in_nc:
            x = x[:, 0:1, :, :] if self.in_nc == 1 else x
        ctx = None
        if self.cross_modal and vi is not None:
            ctx = self.ctx_builder(x, vi)
            if self.use_crm:
                ctx['gate'] = True        # reliability x spectral-budget gating (order-independent)
        self.last_ctx = ctx
        code_loss_v = None
        feat = self.fea_conv(x)
        outs = []; cur = feat
        for b in self.blocks:
            if ctx is not None and getattr(b, 'accepts_ctx', False):
                cur = b(cur, ctx)
            else:
                cur = b(cur)
            outs.append(cur)
        body = self.c(torch.cat(outs, dim=1))
        self._ds_aux = None
        if getattr(self, 'ds_mods', None) and len(self.ds_mods) > 0:
            _g = x.mean(dim=1, keepdim=True)          # grayscale descriptor, 2 dims
            _s = torch.cat([_g.mean(dim=(2, 3)), _g.std(dim=(2, 3))], 1)
            _aux = []
            _code = None
            if 'vq' in self.ds_mods:
                _code, _commit, _idx = self.ds_mods['vq'](_s)
                _aux.append(_commit)
            if 'film' in self.ds_mods and _code is not None:
                body = self.ds_mods['film'](body, _code)
            if 'xattn' in self.ds_mods and _code is not None:
                body = self.ds_mods['xattn'](body, _code)
            if 'moe8' in self.ds_mods:
                body, _sp = self.ds_mods['moe8'](body)
                _aux.append(0.01 * _sp)
            if 'contrast' in self.ds_mods and _code is not None:
                body = body + 0.0 * body.mean() * _code.mean()   # keep graph, no shape change
            self._ds_aux = _aux if _aux else None
        if getattr(self, 'degcodec_on', False):
            c_true = deg if (deg is not None and float(deg.abs().sum()) >= 0) else None
            body = self.degcodec(body, code=None, lr=x, clip_feat=clip_feat, c_true=c_true,
                                 code_weight=getattr(self, 'code_weight', 1.0),
                                 con_weight=getattr(self, 'con_weight', 0.1))
            code_loss_v = getattr(self.degcodec, 'code_loss', None)
        if getattr(self, 'degaware_on', False):
            gamma, beta = self.da(x)                     # M-D: degradation-aware FiLM (identity at init)
            body = body * (1.0 + gamma) + beta
        body = self.lr_conv(body) + feat
        if (self.tscf_on or self.caf_on or getattr(self, 'sghf_on', False)) and ctx is not None:
            body = self.tscf(body, ctx)      # cross-modal fusion (TSCF ours / CAFNet-style control)
        out = self.upsampler(body)
        out = out + F.interpolate(x, scale_factor=self.upscale, mode='nearest')
        if self.hfls_on or self.hfls2_on or self.hfls3_on or self.hfls4_on:
            out = out + self.hfls(body)      # HF latent distillation sampling (zero-init -> starts at baseline)
        if self.in_nc == 1:
            out = torch.cat([out, out, out], dim=1)
        if self.dual_reg and self.training:
            lr_recon = self.dual_head(out)
            self.aux_loss = F.l1_loss(lr_recon, x)
        else:
            self.aux_loss = None
        if self.hfls_on or self.hfls2_on or self.hfls3_on or self.hfls4_on:      # expose HF-latent reconstruction loss (HFLS2) to SRModel
            self.aux_loss = getattr(self.hfls, 'aux_loss', None)
        if getattr(self, '_ds_aux', None):
            _tot = sum(self._ds_aux)
            self.aux_loss = _tot if self.aux_loss is None else (self.aux_loss + self.aux_w * _tot)
        if getattr(self, 'sghf_on', False):                                       # SG-HF gate supervision (survives later resets)
            self.aux_loss = getattr(self.tscf, 'aux_loss', None)
        if getattr(self, 'degcodec_on', False) and code_loss_v is not None:       # SDC degradation-code loss (highest precedence)
            self.aux_loss = code_loss_v
        # physics prior (C1): SR output's radiometric quantity (from in-network rad_head) must conserve the LR's
        self.phys_loss = None
        if self.training and self.phys_prior:
            R = lambda v: F.adaptive_avg_pool2d(self.rad_head(v), x.shape[-2:])   # radiance estimate, down to LR grid
            self.phys_loss = F.l1_loss(R(out), R(x))
        return out


@ARCH_REGISTRY.register()
class IRLiteTopo(nn.Module):
    """Topologically-optimized IR SR: three parallel branches (main spatial distillation / spectral / physics)
    fused cross-branch with a channel-spatial attention. Not a flat stack."""
    def __init__(self, in_nc=3, nf=56, out_nc=3, upscale=4, main_blocks=2):
        super().__init__()
        self.in_nc = in_nc; self.nf = nf; self.upscale = upscale
        self.fea_conv = nn.Conv2d(in_nc, nf, 3, padding=1)
        # main spatial branch: RFDB distillation backbone
        self.main = nn.Sequential(*[BlockRFDS(nf) for _ in range(int(main_blocks))])
        # spectral branch
        self.spectral = nn.Sequential(BlockSP(nf), BlockFR(nf))
        # physics branch
        self.physics = nn.Sequential(BlockRadCons(nf), BlockDAG(nf))
        # fusion: pool each (normalized) branch -> per-branch scalar gates (3), direct weighted sum
        self.fuse = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(3 * nf, 16, 1), nn.PReLU(16), nn.Conv2d(16, 3, 1), nn.Sigmoid())
        self.tail = nn.Conv2d(nf, nf, 3, padding=1)
        self.upsampler = nn.Sequential(nn.Conv2d(nf, out_nc * upscale * upscale, 3, padding=1), nn.PixelShuffle(upscale))
    def forward(self, x):
        B, C, H, W = x.shape
        if C != self.in_nc:
            x = x[:, 0:1] if self.in_nc == 1 else x
        feat = self.fea_conv(x)
        m = self.main(feat)
        s = self.spectral(feat)
        p = self.physics(feat)
        # RMS-normalize each branch so energies are comparable (no single branch swamps)
        def rms(t): return t / (t.pow(2).mean(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-6)
        mn, sn, pn = rms(m), rms(s), rms(p)
        w = self.fuse(torch.cat([mn, sn, pn], dim=1))          # B x 3 x 1 x 1 per-branch gates
        body = self.tail(mn * w[:, 0:1] + sn * w[:, 1:2] + pn * w[:, 2:3] + feat)
        out = self.upsampler(body)
        out = out + F.interpolate(x, scale_factor=self.upscale, mode='nearest')
        if self.hfls_on or self.hfls2_on or self.hfls3_on or self.hfls4_on:
            out = out + self.hfls(body)      # HF latent distillation sampling (zero-init -> starts at baseline)
        if self.in_nc == 1:
            out = torch.cat([out, out, out], dim=1)
        return out


@ARCH_REGISTRY.register()
class IRTransfer(nn.Module):
    """Efficient domain-transfer module: a frozen pre-trained base (any FDRNet/Topo weights) + tiny per-block
    FiLM adapters + one adapter conv at the fusion. A new domain is adapted by training ONLY the adapters
    (a few % of params, RFDN-style in-domain fast transfer, no catastrophic forgetting of the base domain)."""
    def __init__(self, in_nc=3, nf=56, out_nc=3, upscale=4, block_types=None, base_key='params'):
        super().__init__()
        self.base = FDRNet(in_nc=in_nc, nf=nf, num_blocks=len(block_types), out_nc=out_nc, upscale=upscale, block_types=block_types)
        nb = self.base.num_blocks
        self.ada_scale = nn.ParameterList([nn.Parameter(torch.ones(nf)) for _ in range(nb)])
        self.ada_bias = nn.ParameterList([nn.Parameter(torch.zeros(nf)) for _ in range(nb)])
        self.ada_fuse = nn.Conv2d(nf, nf, 1)                       # small fusion adapter (identity-friendly)
        nn.init.zeros_(self.ada_fuse.weight); nn.init.zeros_(self.ada_fuse.bias)
        for p in self.base.parameters():
            p.requires_grad = False                                # freeze base
        self.frozen = True
    def train_adapter_only(self):
        self.frozen = True
        for p in self.base.parameters(): p.requires_grad = False
        for p in list(self.ada_scale.parameters()) + list(self.ada_bias.parameters()) + list(self.ada_fuse.parameters()):
            p.requires_grad = True
    def forward(self, x):
        b, base = self, self.base
        cur = base.fea_conv(x)
        outs = []
        for i, blk in enumerate(base.blocks):
            cur = blk(cur)
            cur = cur * self.ada_scale[i].view(1, -1, 1, 1) + self.ada_bias[i].view(1, -1, 1, 1)
            outs.append(cur)
        body = base.c(torch.cat(outs, dim=1))
        body = base.lr_conv(body) + self.ada_fuse(cur)
        out = base.upsampler(body)
        out = out + F.interpolate(x, scale_factor=base.upscale, mode='nearest')
        if base.in_nc == 1:
            out = torch.cat([out, out, out], dim=1)
        return out
