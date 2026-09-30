# -*- coding: utf-8 -*-
"""N-01..N-06: degradation-semantics and heavy-guidance modules.

RKNN-friendly: conv / linear / PReLU / Sigmoid / elementwise / matmul only.
No softmax over a dynamic set, no top-k inside the graph, no FFT.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import os as _os
_RICH = _os.environ.get('IR_DESC', 'rich') != 'basic'

def _gate_in(c, basic=False):
    return 2 * c if (basic or not _RICH) else 6 * c + 2

def _desc(x, basic=False):
    flat = torch.cat([x.mean(dim=(2, 3)), x.std(dim=(2, 3))], 1)
    if basic or not _RICH:
        return flat
    gx = x[:, :, 1:, :] - x[:, :, :-1, :]
    gy = x[:, :, :, 1:] - x[:, :, :, :-1]
    gg = (gx[:, :, :, :-1] ** 2 + gy[:, :, :-1, :] ** 2).sqrt()
    st = [torch.stack([gg.mean(dim=(1, 2, 3)), gg.std(dim=(1, 2, 3))], 1)]
    for s_ in (2, 4):
        pp = F.avg_pool2d(x, s_) if min(x.shape[-2:]) >= s_ else x
        st += [pp.mean(dim=(2, 3)), pp.std(dim=(2, 3))]
    return torch.cat([flat] + st, 1)


class VQDegCodebook(nn.Module):
    """N-01  learned degradation codebook, nearest-neighbour quantisation, straight-through grad."""

    def __init__(self, in_dim=2, code_dim=64, n_codes=64, hidden=128):
        super().__init__()
        self.n_codes, self.code_dim = n_codes, code_dim
        self.enc = nn.Sequential(nn.Linear(in_dim, hidden), nn.PReLU(hidden), nn.Linear(hidden, code_dim))
        self.book = nn.Parameter(torch.randn(n_codes, code_dim) * (1.0 / code_dim ** 0.5))

    def forward(self, s):
        z = self.enc(s)
        # matmul form: ||z||^2 - 2 z K^T + ||K||^2   (RKNN-convertible, no cdist)
        d = (z * z).sum(1, keepdim=True) - 2.0 * (z @ self.book.t()) + (self.book * self.book).sum(1).unsqueeze(0)
        d = d.clamp(min=0.0).sqrt()
        idx = d.argmin(dim=1)
        q = self.book[idx]
        q_st = z + (q - z).detach()
        commit = F.mse_loss(z.detach(), q) + 0.25 * F.mse_loss(z, q.detach())
        return q_st, commit, idx


class FiLM(nn.Module):
    """N-02  degradation code -> per-channel affine modulation."""

    def __init__(self, code_dim=64, nf=56):
        super().__init__()
        self.to_film = nn.Sequential(nn.Linear(code_dim, nf * 2), nn.PReLU(nf * 2), nn.Linear(nf * 2, nf * 2))

    def forward(self, x, code):
        g, b = self.to_film(code).chunk(2, dim=1)
        return x * (1.0 + g[:, :, None, None]) + b[:, :, None, None]


class DegCrossAttn(nn.Module):
    """N-03  degradation embedding as query over the feature map, sigmoid attention (no softmax)."""

    def __init__(self, code_dim=64, nf=56, n_heads=4):
        super().__init__()
        self.h = n_heads
        self.dk = max(8, nf // n_heads)
        self.q = nn.Linear(code_dim, self.h * self.dk)
        self.k = nn.Conv2d(nf, self.h * self.dk, 1)
        self.v = nn.Conv2d(nf, nf, 1)
        self.att = nn.Sequential(nn.Conv2d(self.h, self.h, 1), nn.Sigmoid())
        self.proj = nn.Conv2d(nf, nf, 1)

    def forward(self, x, code):
        b, c, h, w = x.shape
        q = self.q(code).view(b, self.h, self.dk)
        k = self.k(x).view(b, self.h, self.dk, h * w)
        a = (q.unsqueeze(-1) * k).sum(2) / (self.dk ** 0.5)
        a = self.att(a.view(b, self.h, h, w))
        v = self.v(x).view(b, 1, c, h * w)
        y = (a.view(b, self.h, 1, h * w) * v).sum(1)
        return x + self.proj(y.view(b, c, h, w))


class ContrastiveCode(nn.Module):
    """N-04  InfoNCE over degradation codes; two views of one image are positives (label free)."""

    def __init__(self, dim=64, tau=0.1):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.PReLU(dim), nn.Linear(dim, dim))
        self.tau = tau

    def forward(self, z1, z2):
        f1 = F.normalize(self.proj(z1), dim=1)
        f2 = F.normalize(self.proj(z2), dim=1)
        logits = f1 @ f2.t() / self.tau
        labels = torch.arange(f1.shape[0], device=f1.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


class WideMoE(nn.Module):
    """N-05  K=8 widened expert pool, sigmoid gate (no softmax), returns a sparsity term."""

    def __init__(self, nf=56, n_experts=8, expert_mult=2, gate_hidden=64):
        super().__init__()
        self.n_experts = n_experts
        w = nf * expert_mult
        self.gate = nn.Sequential(nn.Linear(_gate_in(nf), gate_hidden), nn.PReLU(gate_hidden),
                                  nn.Linear(gate_hidden, n_experts), nn.Sigmoid())
        # ONE grouped path that is mathematically identical to K separate experts:
        #   expand (K*w) -> per-channel DW 3x3 -> per-group 1x1 back to c
        self.e1 = nn.Conv2d(nf, n_experts * w, 1)
        self.eact1 = nn.PReLU(n_experts * w)
        self.edw = nn.Conv2d(n_experts * w, n_experts * w, 3, padding=1, groups=n_experts * w)
        self.eact2 = nn.PReLU(n_experts * w)
        self.e2 = nn.Conv2d(n_experts * w, n_experts * nf, 1, groups=n_experts)
        for m in (self.e2,):
            nn.init.zeros_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        s = _desc(x)
        g = self.gate(s)                                    # [B,K]
        b, c, h, wd = x.shape
        z = self.eact1(self.e1(x))                           # [B,K*w,h,w]
        z = self.eact2(self.edw(z))
        z = self.e2(z).view(b, self.n_experts, c, h, wd)     # [B,K,c,h,w]
        y = (g[:, :, None, None, None] * z).sum(1)           # weighted combine (one grouped path)
        return x + y, g.abs().mean()

class CGHFv2(nn.Module):
    """N-06  criterion-conditioned cross-modal attention (heavy guided version)."""

    def __init__(self, nf=56, n_heads=4, n_crit=3):
        super().__init__()
        self.h = n_heads
        self.dk = max(8, nf // n_heads)
        self.q = nn.Conv2d(nf, self.h * self.dk, 1)
        self.k = nn.Conv2d(nf, self.h * self.dk, 1)
        self.v = nn.Conv2d(nf, nf, 1)
        self.crit_w = nn.Parameter(torch.ones(n_crit) / n_crit)
        self.bias = nn.Conv2d(n_crit, self.h, 1)
        self.proj = nn.Sequential(nn.Conv2d(nf, nf, 1), nn.PReLU(nf), nn.Conv2d(nf, nf, 1))
        self.gate = nn.Sequential(nn.Conv2d(2 * nf, 32, 1), nn.PReLU(32), nn.Conv2d(32, nf, 1), nn.Sigmoid())
        self.lp = nn.AvgPool2d(3, 1, 1)
        self.down = 4          # attention resolution divisor (#2)

    @staticmethod
    def _grad(x):
        kx = torch.tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]],
                          dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
        g = x.mean(1, keepdim=True)
        return F.conv2d(g, kx, padding=1), F.conv2d(g, kx.transpose(2, 3), padding=1)

    def criteria(self, ir, vi):
        gx1, gy1 = self._grad(ir)
        gx2, gy2 = self._grad(vi)
        n1 = (gx1 ** 2 + gy1 ** 2).sqrt() + 1e-6
        n2 = (gx2 ** 2 + gy2 ** 2).sqrt() + 1e-6
        cos = (gx1 * gx2 + gy1 * gy2) / (n1 * n2)
        hi_ir = ir.mean(1, keepdim=True) - self.lp(ir.mean(1, keepdim=True))
        hi_vi = vi.mean(1, keepdim=True) - self.lp(vi.mean(1, keepdim=True))
        freq = 1.0 - (hi_ir - hi_vi).abs() / (hi_ir.abs() + hi_vi.abs() + 1e-6)
        mag = torch.sigmoid(4.0 * (n2 - 0.05))
        crit = torch.cat([cos, freq, mag], 1)
        a = (cos * (freq + mag) * 0.5).clamp(0.0, 1.0)
        return crit, a

    def forward(self, body, ctx):
        crit, a = self.criteria(body, ctx)
        w = (self.crit_w.abs() / self.crit_w.abs().sum().clamp(min=1e-6)).view(1, -1, 1, 1)
        crit = crit * w
        b, c, h, wd = body.shape
        ds = self.down                                    # attention at 1/ds resolution
        h2, w2 = max(1, h // ds), max(1, wd // ds)
        body_s = F.avg_pool2d(body, ds) if ds > 1 else body
        ctx_s = F.avg_pool2d(ctx, ds) if ds > 1 else ctx
        crit_s = F.avg_pool2d(crit, ds) if ds > 1 else crit
        q = self.q(body_s).view(b, self.h, self.dk, h2 * w2)              # [B,h,d,HW']
        k = self.k(ctx_s).view(b, self.h, self.dk, h2 * w2)
        logits = torch.matmul(q.transpose(2, 3), k) / (self.dk ** 0.5)     # [B,h,HW',HW']
        logits = logits + self.bias(crit_s).view(b, self.h, h2 * w2).unsqueeze(2)
        att = torch.sigmoid(logits)
        v = self.v(ctx_s).view(b, c, h2 * w2)
        y = torch.matmul(att, v.transpose(1, 2)).mean(1)                   # [B,HW',c]
        y = y.transpose(1, 2).view(b, c, h2, w2)
        if ds > 1:
            y = F.interpolate(y, size=(h, wd), mode='bilinear', align_corners=False)
        fused = self.lp(body)
        hf = self.gate(torch.cat([fused, y], 1)) * (y - self.lp(y))
        return fused + hf + self.proj(y), a.mean()
