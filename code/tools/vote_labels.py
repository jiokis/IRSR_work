# -*- coding: utf-8 -*-
"""S1..S7 weighted voting for open-vocabulary labels.

Public API
    vote_label_dists(list_of_dists, weights, abstain_thresh) -> (q, conf)
        list_of_dists : list of [H, W, L] arrays (one per source; abstaining pixels = uniform)
        weights       : [S, L] per-source per-class reliability
    smooth(q, k)      : light spatial smoothing (stand-in for DenseCRF)
    cascade_mask(conf, q, thresh) -> bool map of "disputed" pixels to escalate to a second VLM

The module is model free: it consumes whatever distributions the sources produce, so the whole
voting logic can be validated before any VLM is downloaded (see demo_pipeline.py).
"""
from __future__ import annotations

import numpy as np


def normalize(d):
    s = d.sum(-1, keepdims=True)
    return d / np.maximum(s, 1e-8)


def vote_label_dists(dists, weights, abstain_thresh=0.0):
    """Weighted vote with per-source per-class reliability.

    dists   : list of [H,W,L] (uniform row = abstain)
    weights : [S,L] reliability of source s on class l
    returns q [H,W,L] (normalised), conf [H,W]
    """
    H, W, L = dists[0].shape
    q = np.zeros((H, W, L), np.float32)
    cov = np.zeros((H, W, L), np.float32)                      # which classes got any vote
    for s, d in enumerate(dists):
        w = weights[s][None, None, :]
        q += d * w
        cov += (d > (1.0 / L) * 1.5).astype(np.float32)         # informative (non-abstain) votes
    q = normalize(q)
    ent = -(q * np.log(np.maximum(q, 1e-8))).sum(-1)
    conf = 1.0 - ent / np.log(L)
    conf = np.where(cov.sum(-1) > 0, conf, 0.0)                 # no votes -> no confidence
    return q.astype(np.float32), conf.astype(np.float32)


def smooth(q, k=5):
    """Box blur over the label dimension (light spatial regularisation)."""
    if k <= 1:
        return q
    pad = k // 2
    out = np.zeros_like(q)
    for c in range(q.shape[-1]):
        p = np.pad(q[..., c], pad, mode='edge')
        acc = np.zeros_like(q[..., c])
        for dy in range(k):
            for dx in range(k):
                acc += p[dy:dy + q.shape[0], dx:dx + q.shape[1]]
        out[..., c] = acc / (k * k)
    return normalize(out)


def cascade_mask(q, conf, conf_thresh=0.35, margin_thresh=0.15):
    """Pixels worth escalating to a stronger (7B) arbiter: low confidence or small top-2 margin."""
    top2 = np.sort(q, axis=-1)[..., -2:]
    margin = top2[..., 1] - top2[..., 0]
    return (conf < conf_thresh) | (margin < margin_thresh)


def make_confusion(L, acc, bias=None, rng=None, bias_strength=0.5):
    """Row = GT, col = predicted. Per-class accuracy + optional systematic bias mass."""
    rng = rng or np.random.RandomState(0)
    C = np.zeros((L, L), np.float32)
    for i in range(L):
        a = acc[i] if isinstance(acc, (list, np.ndarray)) else acc
        rest = 1.0 - a
        C[i, i] = a
        if bias is not None and i in bias:
            j = bias[i]
            C[i, j] += rest * bias_strength
            rest *= (1.0 - bias_strength)
        idx = [j for j in range(L) if j != i and (bias is None or bias.get(i) != j)]
        p = rng.dirichlet(np.ones(len(idx)))
        C[i, idx] = rest * p
    return C


def sample_source(C, gt, abstain=0.0, rng=None):
    """gt [H,W] ints -> distribution [H,W,L] (uniform where the source abstains)."""
    rng = rng or np.random.RandomState(1)
    L = C.shape[0]
    H, W = gt.shape
    out = np.full((H, W, L), 1.0 / L, np.float32)
    flat = gt.reshape(-1)
    pick = np.stack([rng.choice(L, p=C[c]) for c in flat]).reshape(H, W)
    abs_mask = rng.rand(H, W) < abstain
    oh = np.zeros((H, W, L), np.float32)
    oh[np.arange(H)[:, None], np.arange(W)[None, :], pick] = 1.0
    out[~abs_mask] = oh[~abs_mask]
    return out


def estimate_reliability(sources, gt_cal, L, eps=1e-3):
    """w_s(l): empirical accuracy of source s on class l, from a *calibration* split."""
    S = len(sources)
    W = np.zeros((S, L), np.float32)
    for s, d in enumerate(sources):
        pred = d.argmax(-1)
        informative = d.max(-1) > (1.0 / L) * 1.5
        for l in range(L):
            m = (gt_cal == l) & informative
            W[s, l] = (pred[m] == gt_cal[m]).mean() if m.sum() > 5 else 0.5
    return np.clip(W, eps, 1.0)
