# -*- coding: utf-8 -*-
"""Turn voted label distributions into the 7-channel physical prior field the network consumes.

P = [ eps_hat, var_eps, r_phys, tau_hat, conf, var_r, var_tau ]
    eps_hat : expected emissivity           (from the knowledge table)
    var_eps : emissivity variance           <- vote disagreement (free uncertainty!)
    r_phys  : cross-modal physical usability (from spectral table)
    tau_hat : atmospheric transmission prior (from the class' typical path length)
    conf    : semantic confidence
"""
from __future__ import annotations

import numpy as np

# 典型大气路径长度类 -> 透过率先验（LWIR）；真实系统由 atmospheric_prior.yaml 提供
TAU_BY_CLASS = {
    'sky_clear': 0.35, 'sky_cloud': 0.60, 'vegetation': 0.80, 'water': 0.70,
    'person': 0.90, 'asphalt': 0.85, 'concrete': 0.85, 'metal': 0.80,
    'glass': 0.75, 'unknown': 0.70,
}


def build_prior_fields(q, conf, classes, eps_tab, r_phys, tau_by_class=None):
    """q [H,W,L], conf [H,W] -> P [H,W,7] float32 and per-class expectation/variance."""
    tau_by_class = tau_by_class or TAU_BY_CLASS
    eps_mid = np.array([eps_tab[c]['mid'] for c in classes], np.float32)
    r_vec = np.array([r_phys[c] for c in classes], np.float32)
    tau_vec = np.array([tau_by_class.get(c, 0.7) for c in classes], np.float32)

    eps_hat = (q * eps_mid).sum(-1)
    var_eps = (q * (eps_mid[None, None, :] - eps_hat[..., None]) ** 2).sum(-1)
    r_hat = (q * r_vec).sum(-1)
    var_r = (q * (r_vec[None, None, :] - r_hat[..., None]) ** 2).sum(-1)
    tau_hat = (q * tau_vec).sum(-1)
    var_tau = (q * (tau_vec[None, None, :] - tau_hat[..., None]) ** 2).sum(-1)

    P = np.stack([eps_hat, var_eps, r_hat, tau_hat, conf, var_r, var_tau], -1).astype(np.float32)
    return P, dict(eps_hat=eps_hat, var_eps=var_eps, r_hat=r_hat, tau_hat=tau_hat)


def table_consistency_loss(q, eps_hat, classes, eps_tab):
    """hinge: penalise eps_hat outside the [lo,hi] interval of the voted class (innovation #2)."""
    tot = 0.0
    n = 0
    for i, c in enumerate(classes):
        w = q[..., i]
        if w.sum() <= 0:
            continue
        lo, hi = eps_tab[c]['lo'], eps_tab[c]['hi']
        below = np.maximum(0.0, lo - eps_hat)
        above = np.maximum(0.0, eps_hat - hi)
        tot += float((w * (below + above)).sum())
        n += float(w.sum())
    return tot / max(n, 1e-6)
