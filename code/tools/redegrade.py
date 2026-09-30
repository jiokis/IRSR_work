# -*- coding: utf-8 -*-
"""#14 Differentiable re-degradation operator — the core of the cycle loss (and of PhysIR).

Physical chain we model (so a restored image can be pushed back to the LR domain and compared
with the real LR input):

    Gaussian PSF blur -> detector noise -> 8x8 block quantisation (JPEG proxy)
        -> radiometric (gain/bias/clip) -> atmosphere (transmission tau + haze)

Everything is differentiable; rounding uses a straight-through estimator so gradients flow to the
parameters (sigma, q, tau, gamma, ...), which is what lets the model *learn* the degradation.

    from tools.physir.redegrade import redegrade, cycle_loss
    lr_hat = redegrade(sr, dict(sigma=1.5, noise=0.02, q=0.5, tau=0.8, haze=0.5))
    loss   = cycle_loss(sr, lr, params)        # L1 in the LR domain
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _is_on(v):
    """True when the value should be applied (tensor -> assume on; float -> >0)."""
    return True if torch.is_tensor(v) else float(v) > 0


def _bcast(v, x, lo=0.0):
    """accept scalar or per-image tensor [B] / [B,1] -> reshape to [B,1,1,1] for broadcasting."""
    if torch.is_tensor(v):
        if v.dim() == 0 or v.numel() == 1:
            return v.reshape(1, 1, 1, 1)
        return torch.clamp(v.reshape(-1, 1, 1, 1), min=lo)
    return max(float(v), lo) if lo else float(v)


def _pos(v, lo=1e-3):
    """positive scalar: works for float and for 0-dim tensor (keeps the graph)."""
    if torch.is_tensor(v):
        return torch.clamp(v.reshape(()), min=lo)
    return max(float(v), lo)


def _scalar_sigma(sigma):
    """accept float or 0-dim / 1-elem tensor (keeps the graph for trainable sigma)."""
    if torch.is_tensor(sigma):
        return torch.clamp(sigma.reshape(()), min=1e-3)
    return max(float(sigma), 1e-3)


def _gauss_kernel(sigma, ksize=9, device=None, dtype=None):
    ax = torch.arange(ksize, device=device, dtype=dtype) - (ksize - 1) / 2.0
    sig = _scalar_sigma(sigma)
    g = torch.exp(-(ax ** 2) / (2.0 * sig ** 2))
    return g / g.sum()


def gaussian_blur(x, sigma):
    """Separable depthwise Gaussian blur; differentiable w.r.t. sigma."""
    c = x.shape[1]
    sig = sigma.mean() if torch.is_tensor(sigma) else sigma      # 逐图 sigma 取批均值建核
    g = _gauss_kernel(sig, 9, x.device, x.dtype)
    kx = g.view(1, 1, 1, -1).expand(c, 1, 1, -1)
    ky = g.view(1, 1, -1, 1).expand(c, 1, -1, 1)
    x = F.conv2d(x, kx, padding=(0, 4), groups=c)
    return F.conv2d(x, ky, padding=(4, 0), groups=c)


def add_noise(x, sigma):
    if not _is_on(sigma):
        return x
    return x + _bcast(sigma, x) * torch.randn_like(x)


def block_quant(x, q):
    """JPEG proxy: 8x8 block flattening with straight-through DC quantisation (blockiness + flatness)."""
    if not _is_on(q):
        return x
    b, c, h, w = x.shape
    xp = F.pad(x, (0, (-w) % 8, 0, (-h) % 8), mode='replicate')
    H8, W8 = xp.shape[2] // 8, xp.shape[3] // 8
    blk = xp.view(b, c, H8, 8, W8, 8)
    _q = _bcast(q, x)
    _q = torch.as_tensor(_q, dtype=x.dtype, device=x.device).reshape(-1)
    if _q.numel() == 1:
        _q = _q.expand(b)
    elif _q.numel() != b:
        _q = _q.reshape(b, -1)[:, 0]
    qb = _q.reshape(b, 1, 1, 1, 1, 1)              # [B,1,1,1,1,1] for 6-D broadcasting
    dc = blk.mean(dim=(3, 5), keepdim=True)
    step = 1.0 / qb
    dc_q = dc + (torch.round(dc / step) * step - dc).detach()      # straight-through
    out = blk + (dc_q.expand_as(blk) - blk) * qb
    return out.reshape(b, c, xp.shape[2], xp.shape[3])[..., :h, :w]


def radiometric(x, gamma=1.0, bias=0.0):
    return (gamma * x + bias).clamp(0.0, 1.0)


def atmosphere(x, tau=1.0, haze=0.5):
    return tau * x + (1.0 - tau) * haze


def redegrade(sr, p):
    """p: dict(sigma, noise, q, gamma, bias, tau, haze); every key optional."""
    y = sr
    if _is_on(p.get('sigma', 0)):
        y = gaussian_blur(y, p['sigma'])
    if _is_on(p.get('noise', 0)):
        y = add_noise(y, p['noise'])
    if _is_on(p.get('q', 0)):
        y = block_quant(y, p['q'])
    y = radiometric(y, p.get('gamma', 1.0), p.get('bias', 0.0))
    if not _is_on(p.get('tau', 1.0)) or (not torch.is_tensor(p.get('tau', 1.0)) and float(p.get('tau', 1.0)) < 1.0) or (torch.is_tensor(p.get('tau', 1.0)) and float(p['tau'].min()) < 1.0):
        y = atmosphere(y, _bcast(p['tau'], y), p.get('haze', 0.5))
    return y


def cycle_loss(sr, lr, p):
    """L1 between the re-degraded SR and the real LR (both in the LR domain)."""
    lr_hat = redegrade(sr, p)
    if lr_hat.shape[-2:] != lr.shape[-2:]:
        lr_hat = F.interpolate(lr_hat, size=lr.shape[-2:], mode='area')
    return F.l1_loss(lr_hat, lr)


def cycle_loss_from_logits(sr, lr, raw, cfg=None):
    """Convenience for training loops: raw = dict of tensors (sigma, noise, q, tau, gamma, bias),
    each already a leaf that requires grad -> returns (loss, dict of detached values)."""
    cfg = cfg or {}
    p = {}
    for k in ('sigma', 'noise', 'q', 'tau', 'gamma', 'bias', 'haze'):
        if k in raw:
            p[k] = torch.clamp(raw[k], min=cfg.get(k + '_min', 0.0)) if k != 'bias' else raw[k]
    return cycle_loss(sr, lr, p)
