# -*- coding: utf-8 -*-
"""S7: physics veto — let measured physics overrule an implausible semantic label.

Rules are documented in knowledge/veto_rules.yaml; implemented here explicitly so every rule is
auditable and unit-testable. Veto is conservative: unresolved -> unknown (abstain beats guess).
"""
from __future__ import annotations

import numpy as np


def veto_labels(label_map, eps_measured, eps_lookup, structure=None, upper_half=None,
                sigma=None, tau_veto=2.0, ratio=0.5, prefer=None):
    """Apply epsilon-based veto rules.

    label_map    : [H,W] int (index into class list)
    eps_measured : [H,W] float  (physics head / probe estimate of emissivity)
    eps_lookup   : {class_index: (lo, hi)} from knowledge table
    structure    : optional [H,W] bool (structure present)
    upper_half   : optional [H,W] bool (True if pixel lies in the upper half of the frame)
    returns new_label_map, stats dict
    """
    out = label_map.copy()
    sigma = 0.03 if sigma is None else sigma
    prefer = prefer or {}

    def z_out(v, lo, hi):
        """how many sigmas the measurement lies OUTSIDE the interval"""
        return np.maximum(np.maximum(lo - v, v - hi), 0.0) / sigma

    def inside(v, lo, hi):
        return (v >= lo) & (v <= hi)
    name = eps_lookup['names']
    rules_fired = {}
    H, W = label_map.shape

    for c, idx in eps_lookup['index'].items():
        m = label_map == idx
        if not m.any():
            continue
        lo, hi = eps_lookup['interval'][c]

        if c == 'water':                                     # water_but_metal_like
            zc = z_out(eps_measured, lo, hi)
            bad = m & (zc > tau_veto) & (eps_measured < lo)   # 实测远低于水区间才算证据
            out[bad] = eps_lookup['index']['unknown']
            if bad.sum():
                rules_fired['water_but_metal_like'] = int(bad.sum())

        elif c == 'metal':                                   # metal_but_human_thermal
            zc = z_out(eps_measured, lo, hi)
            tgt = eps_lookup['index']['person']
            tl, th = eps_lookup['interval']['person']
            za = z_out(eps_measured, tl, th)
            bad = m & (zc > tau_veto) & (za < ratio * zc) & inside(eps_measured, tl, th)
            out[bad] = tgt
            if bad.sum():
                rules_fired['metal_but_human_thermal'] = int(bad.sum())

        elif c == 'sky_clear':                               # clear_sky_low_position
            if upper_half is not None and structure is not None:
                bad = m & (~upper_half) & structure
                out[bad] = eps_lookup['index']['unknown']
                if bad.sum():
                    rules_fired['clear_sky_low_position'] = int(bad.sum())

        elif c == 'sky_cloud':                               # cloud_but_hot
            bad = m & (eps_measured < 0.6)
            out[bad] = eps_lookup['index']['unknown']
            if bad.sum():
                rules_fired['cloud_but_hot'] = int(bad.sum())

        elif c == 'person':                                  # person_but_water_eps
            zc = z_out(eps_measured, lo, hi)
            tl, th = eps_lookup['interval']['water']
            za = z_out(eps_measured, tl, th)
            bad = m & (zc > tau_veto) & (za < ratio * zc) & inside(eps_measured, tl, th)
            out[bad] = eps_lookup['index']['water']
            if bad.sum():
                rules_fired['person_but_water_eps'] = int(bad.sum())
    return out, dict(rules_fired=rules_fired, changed=int((out != label_map).sum()), total=int(label_map.size))


def lo_water(lk):
    return lk['interval']['water'][0]


def hi_water(lk):
    return lk['interval']['water'][1]
