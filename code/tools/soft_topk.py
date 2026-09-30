# -*- coding: utf-8 -*-
"""Top-k soft constraint for open-vocabulary semantic voting + physics adjustment.

Replaces brittle hard thresholds by a scale-free top-k formulation:

  1) topk_candidate_set : union of each source's top-k informative labels  -> small candidate set
  2) restrict_to_candidates : renormalise the vote inside that set (mass outside is dropped)
  3) physics_score : per-candidate physical plausibility, expressed in *sigma units*
                     s_phys(l) = -dist(eps_measured, [lo_l, hi_l]) / sigma_eps
  4) soft_adjust : q' = softmax( log q + lambda * s_phys )   <- differentiable soft re-weighting,
                   i.e. the "veto" becomes a soft constraint instead of a hard rule
  5) abstain_topk : abstain when physics cannot separate the top-2 candidates
                    (gap < delta, again in sigma units -> scale free)

Everything is plain numpy so it can be unit-tested without any model.
"""
from __future__ import annotations

import numpy as np


def topk_candidate_set(dists, k=2, informative_eps=1.5, top_k_final=3):
    """Union of per-source top-k labels (sources that abstain contribute nothing)."""
    H, W, L = dists[0].shape
    cand = np.zeros((H, W, L), bool)
    for d in dists:
        informative = d.max(-1) > informative_eps / L
        idx = np.argsort(-d, axis=-1)[..., :k]
        row = np.arange(H)[:, None, None]
        col = np.arange(W)[None, :, None]
        sel = np.zeros((H, W, L), bool)
        for j in range(idx.shape[-1]):
            sel[row, col, idx[..., j]] = True
        cand |= sel & informative[..., None]
    # 并集后按聚合票质量只保留前 top_k_final 个（否则 6 源×top2 会覆盖全集）
    if top_k_final and top_k_final < cand.shape[-1]:
        mass = np.zeros(cand.shape, np.float32)
        for d in dists:
            mass += d
        mass = np.where(cand, mass, -1.0)
        keep = np.argsort(-mass, axis=-1)[..., :top_k_final]
        cand2 = np.zeros_like(cand)
        r = np.arange(cand.shape[0])[:, None, None]
        c_ = np.arange(cand.shape[1])[None, :, None]
        for j in range(keep.shape[-1]):
            cand2[r, c_, keep[..., j]] = True
        cand = cand2
    return cand


def restrict_to_candidates(q, cand):
    q2 = np.where(cand, q, 0.0)
    s = q2.sum(-1, keepdims=True)
    empty = s[..., 0] <= 0
    q2 = np.where(s > 0, q2 / np.maximum(s, 1e-8), 1.0 / q.shape[-1])
    return q2.astype(np.float32), empty


def physics_score(eps_measured, classes, eps_tab, sigma):
    """s_phys [H,W,L]: 0 if the measurement lies inside the class interval, negative sigma distance otherwise."""
    lo = np.array([eps_tab[c]['lo'] for c in classes], np.float32)
    hi = np.array([eps_tab[c]['hi'] for c in classes], np.float32)
    v = eps_measured[..., None]
    d = np.maximum(np.maximum(lo - v, v - hi), 0.0) / max(sigma, 1e-6)
    return (-d).astype(np.float32)


def soft_adjust(q, cand, s_phys, lam=1.0):
    """Differentiable soft physics re-weighting inside the candidate set."""
    logq = np.log(np.maximum(q, 1e-8))
    z = logq + lam * s_phys
    z = np.where(cand, z, -1e9)
    m = z.max(-1, keepdims=True)
    e = np.exp(z - m)
    e = np.where(cand, e, 0.0)
    s = e.sum(-1, keepdims=True)
    q2 = np.where(s > 0, e / np.maximum(s, 1e-8), q)
    return q2.astype(np.float32)


def abstain_topk(q, s_phys, delta=1.0, margin=0.15):
    """Abstain where the top-2 candidates are within `delta` sigma of each other in physics plausibility."""
    idx = np.argsort(-q, axis=-1)[..., :2]
    row = np.arange(q.shape[0])[:, None]
    col = np.arange(q.shape[1])[None, :]
    s1 = s_phys[row, col, idx[..., 0]]
    s2 = s_phys[row, col, idx[..., 1]]
    gap = np.abs(s1 - s2)                      # 物理可分性（sigma 单位）
    qs = np.sort(q, axis=-1)[..., -2:]
    vote_margin = qs[..., 1] - qs[..., 0]      # 语义可分性（票差）
    return (gap < delta) & (vote_margin < margin)


def confidence_from(q, cand):
    """Entropy-based confidence inside the candidate set (no absolute threshold needed)."""
    L = q.shape[-1]
    ent = -(q * np.log(np.maximum(q, 1e-8))).sum(-1)
    n = np.maximum(cand.sum(-1), 1)
    # normalise by the entropy of a uniform distribution over the candidate set size
    conf = 1.0 - ent / np.log(np.maximum(n, 2))
    return np.clip(conf, 0.0, 1.0).astype(np.float32)
