# -*- coding: utf-8 -*-
"""Literature-style guided-fusion baselines, all drop-in for FDRNet (same IR backbone, x4, M3FD).

Every module fuses (body: IR features [B,nf,h,w], ctx: guide features [B,nf,h,w]) and returns the fused
IR features.  `width` scales the module so the total network can be matched to a target parameter count.

  cat_in     early fusion: concat LR+guide at the input, single backbone            (§2.3.1 第一条路线)
  concat1x1  concat features then 1x1 fuse                                         (SwinFusion/U2Fusion 式)
  ca_msff    逐级通道注意力（GAP+GMP 双路 + 3x3），f_ir*(1+g)                      (MSFF-CT [48])
  coarse2fine 粗到细渐进引导，三级由粗到细                                         (CPGNet [37])
  nogate     固定强度高频注入（无判据）：f_ir + a*HP(f_vi)                          (SwinFuSR [40] 简化)
  caf_gate   学习式通道门控（文献式门控，已有对照）                                 (CAFNet 式)
  spatattn   空间注意力图引导（免对齐）                                            (AlignFreeNet [49])
  corrmatch  相关体软匹配 + 变形采样对齐                                           (MASA-SR [50] 轻量版)
  bidir      双向互惠门控                                                          (RRSR [52] 轻量版)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

BASELINES = ['cat_in', 'concat1x1', 'ca_msff', 'coarse2fine', 'nogate',
             'caf_gate', 'spatattn', 'corrmatch', 'bidir']


def _c(in_c, out_c, k=1):
    return nn.Conv2d(in_c, out_c, k, padding=k // 2, bias=True)


class Concat1x1(nn.Module):
    """特征拼接 + 1x1 融合。"""
    def __init__(self, nf, width=1.0):
        super().__init__()
        w = max(16, int(nf * width))
        self.fuse = nn.Sequential(_c(2 * nf, w), nn.PReLU(w), _c(w, nf))

    def forward(self, body, ctx):
        return self.fuse(torch.cat([body, ctx], 1))


class CAMSFF(nn.Module):
    """逐级通道注意力引导：GAP+GMP 双路 + 3x3 局部分支，输出 f_ir*(1+g)。"""
    def __init__(self, nf, width=1.0):
        super().__init__()
        w = max(8, int(nf * width))
        self.mlp = nn.Sequential(nn.Linear(2 * nf, w), nn.PReLU(w), nn.Linear(w, nf))
        self.local = nn.Sequential(_c(nf, w, 3), nn.PReLU(w), _c(w, nf, 3))

    def forward(self, body, ctx):
        g_gap = F.adaptive_avg_pool2d(ctx, 1).flatten(1)
        g_gmp = F.adaptive_max_pool2d(ctx, 1).flatten(1)
        g = torch.sigmoid(self.mlp(torch.cat([g_gap, g_gmp], 1))).unsqueeze(-1).unsqueeze(-1)
        return body * (1.0 + g) + 0.1 * self.local(ctx)


class Coarse2Fine(nn.Module):
    """粗到细渐进引导：三级尺度上先粗后细地注入引导特征。"""
    def __init__(self, nf, width=1.0):
        super().__init__()
        w = max(16, int(nf * width))
        self.f3 = nn.Sequential(_c(2 * nf, w), nn.PReLU(w), _c(w, nf))
        self.f2 = nn.Sequential(_c(2 * nf, w), nn.PReLU(w), _c(w, nf))
        self.f1 = nn.Sequential(_c(2 * nf, w), nn.PReLU(w), _c(w, nf))

    def forward(self, body, ctx):
        out = body
        for f, s in ((self.f3, 4), (self.f2, 2), (self.f1, 1)):
            b = F.interpolate(out, scale_factor=1.0 / s, mode='bilinear', align_corners=False) if s > 1 else out
            c = F.interpolate(ctx, size=b.shape[-2:], mode='bilinear', align_corners=False)
            y = f(torch.cat([b, c], 1))
            out = out + F.interpolate(y, size=out.shape[-2:], mode='bilinear', align_corners=False) if s > 1 else out + y
        return out


class NoGateHF(nn.Module):
    """固定强度高频注入（无判据）：f_ir + a * HP(f_vi)，a 为可学习标量，但没有输入相关判据。"""
    def __init__(self, nf, width=1.0):
        super().__init__()
        self.a = nn.Parameter(torch.ones(1))
        self.proj = _c(nf, nf, 3)
        self.pool = nn.AvgPool2d(3, 1, 1)

    def forward(self, body, ctx):
        hf = ctx - self.pool(ctx)
        return body + self.a * self.proj(hf)


class CafGate(nn.Module):
    """学习式通道门控（文献式门控融合）。"""
    def __init__(self, nf, width=1.0):
        super().__init__()
        w = max(8, int(nf * width))
        self.gate = nn.Sequential(_c(2 * nf, w), nn.PReLU(w), _c(w, nf), nn.Sigmoid())
        self.proj = _c(nf, nf, 3)

    def forward(self, body, ctx):
        g = self.gate(torch.cat([body, ctx], 1))
        return body + g * self.proj(ctx)


class SpatialAttn(nn.Module):
    """空间注意力图引导（免对齐）：空间图 a ∈ [0,1]，逐像素加权引导特征。"""
    def __init__(self, nf, width=1.0):
        super().__init__()
        w = max(8, int(nf * width))
        self.att = nn.Sequential(_c(2 * nf, w, 3), nn.PReLU(w), _c(w, 1, 1), nn.Sigmoid())
        self.proj = _c(nf, nf, 3)

    def forward(self, body, ctx):
        a = self.att(torch.cat([body, ctx], 1))
        return body + a * self.proj(ctx)


class CorrMatch(nn.Module):
    """相关体软匹配 + 网格采样对齐（MASA-SR 轻量版）：先在低分辨率上算相关并软 argmax，再变形采样。"""
    def __init__(self, nf, width=1.0, r=4):
        super().__init__()
        w = max(16, int(nf * width))
        self.r = r
        self.q = _c(nf, w)
        self.k = _c(nf, w)
        self.fuse = nn.Sequential(_c(2 * nf, w), nn.PReLU(w), _c(w, nf))

    def forward(self, body, ctx):
        B, C, H, W = body.shape
        r = self.r
        h, w = max(1, H // r), max(1, W // r)
        q = F.normalize(self.q(F.adaptive_avg_pool2d(body, (h, w))).flatten(2), dim=1)   # B,w,HW
        k = F.normalize(self.k(F.adaptive_avg_pool2d(ctx, (h, w))).flatten(2), dim=1)
        corr = torch.bmm(q.transpose(1, 2), k) / (q.shape[1] ** 0.5)                      # B,HW,HW
        a = torch.softmax(corr, dim=-1)
        idx = torch.arange(h * w, device=body.device, dtype=torch.float32)
        ys = torch.div(idx, w, rounding_mode='floor'); xs = idx - ys * w
        gy = (a * ys).sum(-1) / max(1, w - 1) * 2 - 1
        gx = (a * xs).sum(-1) / max(1, h - 1) * 2 - 1
        grid = torch.stack([gx, gy], -1).view(B, h, w, 2)
        ctx_small = F.adaptive_avg_pool2d(ctx, (h, w))
        ctx_al = F.grid_sample(ctx_small, grid, align_corners=True)
        ctx_al = F.interpolate(ctx_al, size=(H, W), mode='bilinear', align_corners=False)
        return body + self.fuse(torch.cat([body, ctx_al], 1))


class BiDir(nn.Module):
    """双向互惠门控：IR→VI 与 VI→IR 两个方向的门控后相加。"""
    def __init__(self, nf, width=1.0):
        super().__init__()
        w = max(8, int(nf * width))
        self.g_iv = nn.Sequential(_c(2 * nf, w), nn.PReLU(w), _c(w, nf), nn.Sigmoid())
        self.g_vi = nn.Sequential(_c(2 * nf, w), nn.PReLU(w), _c(w, nf), nn.Sigmoid())
        self.fuse = _c(nf, nf, 3)

    def forward(self, body, ctx):
        cat = torch.cat([body, ctx], 1)
        return body + self.fuse(self.g_iv(cat) * ctx + self.g_vi(cat) * body)


def build_baseline(name, nf, width=1.0, **kw):
    table = {'concat1x1': Concat1x1, 'ca_msff': CAMSFF, 'coarse2fine': Coarse2Fine,
             'nogate': NoGateHF, 'caf_gate': CafGate, 'spatattn': SpatialAttn,
             'corrmatch': CorrMatch, 'bidir': BiDir}
    if name not in table:
        raise KeyError('unknown baseline %s（可选：%s）' % (name, BASELINES))
    return table[name](nf, width=width, **kw)
