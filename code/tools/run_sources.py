# -*- coding: utf-8 -*-
"""S1 semantic pipeline: run the multi-source label estimators and chain them into prior fields.

Sources (each returns [H, W, L] label distributions over VOCAB; uniform row = abstain)
  S1 evf_sam   : text -> mask        (primary segmentation)                 [needs model]
  S2 clipseg   : text -> mask        (second, architecturally different)    [needs model]
  S3 qwen_vl   : scene/material attributes -> broadcast hints               [needs model]
  S4 siglip    : text-image similarity scores                               [needs model]
  S5 crossmodal: visible-vs-infrared structure agreement                    [free, model-free]
  S6 temporal  : label stability across neighbouring frames                 [free, model-free]

Everything heavy is imported lazily, so this module imports and runs on a CPU-only machine.
`--mock` fabricates source distributions so the whole chain (sources -> vote -> veto -> prior
fields) can be validated before any weights are downloaded.

Usage
    python tools/vlm/run_sources.py --mock --out experiments/PHYSIR/demo2
    python tools/vlm/run_sources.py --ir img.png --vis guide.png --out ...   # real sources (TODO)
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

VOCAB = ['person', 'vegetation', 'water', 'asphalt', 'concrete', 'metal', 'glass', 'sky_clear', 'unknown']
L = len(VOCAB)


# --------------------------------------------------------------------------- helper
def uniform(H, W):
    return np.full((H, W, L), 1.0 / L, np.float32)


def onehot_from_labels(labels, H, W, conf=0.9):
    """labels: [H,W] ints -> peaked distribution with the given confidence."""
    d = np.full((H, W, L), (1.0 - conf) / max(L - 1, 1), np.float32)
    r = np.arange(H)[:, None]
    c = np.arange(W)[None, :]
    d[r, c, labels] = conf
    return d


# --------------------------------------------------------------------------- S5 / S6 (model-free)
def source_crossmodal(vis_gray, ir_gray, thresh=0.15):
    """Visible-vs-IR structure agreement -> per-pixel confidence multiplier.

    Both are normalised to [0,1]. Where the two modalities disagree about structure we return a
    uniform row (abstain) rather than a guess.
    """
    if vis_gray is None or ir_gray is None:
        return None
    v = (vis_gray - vis_gray.mean()) / (vis_gray.std() + 1e-6)
    i = (ir_gray - ir_gray.mean()) / (ir_gray.std() + 1e-6)
    gx = lambda z: (np.diff(z, axis=1, prepend=z[:, :1]) ** 2 + np.diff(z, axis=0, prepend=z[:1, :]) ** 2) ** 0.5
    gv, gi = gx(v), gx(i)
    agree = (gv > np.quantile(gv, 0.6)) & (gi > np.quantile(gi, 0.6))
    H, W = agree.shape
    d = uniform(H, W)
    d[agree] = 1.0 / L - 1e-6            # keeps it uninformative but marks agreement
    return d, agree


def source_temporal(prev_label, cur_label):
    """Where consecutive frames agree, boost that label; where they disagree, abstain."""
    if prev_label is None:
        return None
    H, W = cur_label.shape
    same = (prev_label == cur_label)
    d = uniform(H, W)
    conf = np.where(same, 0.80, 0.0).astype(np.float32)
    r = np.arange(H)[:, None]
    c = np.arange(W)[None, :]
    d[r, c, cur_label] = conf
    return d


# --------------------------------------------------------------------------- real sources (lazy)
class Sources:
    """Holds model handles; every model is loaded lazily so a missing dependency is not fatal."""

    def __init__(self, weights_root=None, device='cuda'):
        self.w = weights_root or os.path.join(os.path.dirname(ROOT), 'weights')
        self.device = device
        self._evf = self._clipseg = self._qwen = self._siglip = None

    # ---- S1
    def evf_sam(self, vis_rgb, vocab=VOCAB):
        """TODO(real): EVF-SAM text-conditioned mask -> per-vocab score maps."""
        try:
            import torch  # noqa
        except Exception:
            return None
        raise NotImplementedError('EVF-SAM inference not wired yet (see design doc S1)')

    # ---- S2
    def clipseg(self, vis_rgb, vocab=VOCAB):
        """TODO(real): CLIPSeg dense scores per text prompt."""
        raise NotImplementedError('CLIPSeg inference not wired yet')

    # ---- S3
    def qwen_vl_attrs(self, vis_rgb):
        """Return the attribute JSON only (no per-pixel prediction). Schema is fixed here."""
        raise NotImplementedError('Qwen2.5-VL inference not wired yet')

    # ---- S4
    def siglip_scores(self, vis_rgb, vocab=VOCAB):
        raise NotImplementedError('SigLIP inference not wired yet')


ATTR_SCHEMA = {
    'scene': ['urban', 'suburban', 'highway', 'rural', 'indoor'],
    'time': ['day', 'night', 'dusk'],
    'weather': ['clear', 'haze', 'fog', 'rain'],
    'materials': VOCAB[:-1],
    'notes': 'free text',
}


def attrs_to_hint(attrs, H, W):
    """S3 -> a weak global prior: emphasise materials the VLM named, down-weight the rest."""
    if not attrs:
        return uniform(H, W)
    d = np.full((H, W, L), 0.5 / L, np.float32)
    named = [m for m in attrs.get('materials', []) if m in VOCAB]
    for m in named:
        d[..., VOCAB.index(m)] += 0.5 / max(len(named), 1)
    d /= d.sum(-1, keepdims=True)
    if attrs.get('time') == 'night':
        d[..., VOCAB.index('sky_clear')] *= 0.3      # night policy from abstain_policy.yaml
        d /= d.sum(-1, keepdims=True)
    return d


# --------------------------------------------------------------------------- chain
def run_chain(ir, vis=None, prev_label=None, mock=False, out=None, srcs=None):
    H, W = ir.shape[:2]
    dists, names = [], []

    if mock:
        rng = np.random.RandomState(7)
        gt = rng.randint(0, L, size=(H, W))
        specs = [('S1 evf_sam', 0.72, None, 0.0), ('S2 clipseg', 0.65, None, 0.05),
                 ('S3 qwen_vl', 0.74, None, 0.35), ('S4 siglip', 0.58, None, 0.0),
                 ('S5 crossmodal', 0.80, None, 0.20), ('S6 temporal', 0.78, None, 0.10)]
        for nm, acc, _b, ab in specs:
            flip = rng.rand(H, W) > acc
            lab = np.where(flip, rng.randint(0, L, size=(H, W)), gt)
            ab_m = rng.rand(H, W) < ab
            d = onehot_from_labels(lab, H, W, conf=0.85)
            d[ab_m] = 1.0 / L
            dists.append(d); names.append(nm)
    else:
        s = srcs or Sources()
        for fn, nm in ((s.evf_sam, 'S1 evf_sam'), (s.clipseg, 'S2 clipseg'),
                       (s.siglip_scores, 'S4 siglip')):
            try:
                d = fn(vis if vis is not None else ir)
                if d is not None:
                    dists.append(d); names.append(nm)
            except NotImplementedError as e:
                print('  [skip] %s: %s' % (nm, str(e)[:60]))
        try:
            dists.append(attrs_to_hint(s.qwen_vl_attrs(vis if vis is not None else ir), H, W))
            names.append('S3 qwen_vl')
        except NotImplementedError as e:
            print('  [skip] S3: %s' % str(e)[:60])
        cm = source_crossmodal(None, None)
        if cm:
            dists.append(cm[0]); names.append('S5 crossmodal')
        tm = source_temporal(prev_label, np.zeros((H, W), np.int64))
        if tm is not None:
            dists.append(tm); names.append('S6 temporal')

    if not dists:
        raise RuntimeError('no source produced output')

    from tools.vlm.vote_labels import vote_label_dists, smooth, estimate_reliability
    from tools.vlm.soft_topk import (topk_candidate_set, restrict_to_candidates, physics_score,
                                     soft_adjust, abstain_topk, confidence_from)
    from tools.physir import phys_table
    from tools.physir.build_prior_fields import build_prior_fields
    from tools.physir.physics_veto import veto_labels

    # reliability: without a labelled calibration split we fall back to uniform weights
    weights = np.ones((len(dists), L), np.float32) * 0.6

    q, conf = vote_label_dists(dists, weights)
    q = smooth(q, 5)
    cand = topk_candidate_set(dists, k=2, top_k_final=3)
    qc, empty = restrict_to_candidates(q, cand)

    eps_tab = phys_table.eps_table()
    sigma_eps = 0.05
    eps_measured = np.full((H, W), 0.9, np.float32)          # TODO: from the physics head
    s_phys = physics_score(eps_measured, VOCAB, eps_tab, sigma_eps)
    q_soft = soft_adjust(qc, cand, s_phys, lam=1.0)
    abst = abstain_topk(q_soft, s_phys, delta=0.5, margin=0.15)
    conf_soft = confidence_from(q_soft, cand)

    lab = q_soft.argmax(-1)
    lk = dict(names=VOCAB, index={c: i for i, c in enumerate(VOCAB)},
              interval={c: (eps_tab[c]['lo'], eps_tab[c]['hi']) for c in VOCAB})
    vetoed, vstats = veto_labels(lab, eps_measured, lk, sigma=sigma_eps)

    r_phys = phys_table.prior_reliability()
    P, aux = build_prior_fields(q_soft, conf_soft, VOCAB, eps_tab, r_phys)

    info = dict(names=names, n_sources=len(dists), abstain_ratio=float(abst.mean()),
                veto=vstats, P_shape=list(P.shape))
    if out:
        os.makedirs(out, exist_ok=True)
        np.savez_compressed(os.path.join(out, 'prior_field.npz'),
                            P=P, q_soft=q_soft, cand=cand, conf=conf_soft, label=vetoed,
                            abstain=abst, eps_measured=eps_measured)
        io.open(os.path.join(out, 'info.json'), 'w', encoding='utf-8').write(
            json.dumps(info, ensure_ascii=False, indent=1))
    return P, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ir', default=None)
    ap.add_argument('--vis', default=None)
    ap.add_argument('--mock', action='store_true')
    ap.add_argument('--hw', type=int, default=96)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    if a.mock or not a.ir:
        ir = np.random.rand(a.hw, a.hw, 3).astype(np.float32)
        vis = np.random.rand(a.hw, a.hw, 3).astype(np.float32)
        print('MOCK 模式：用合成源跑通全链路 ✓')
    else:
        from PIL import Image
        ir = np.asarray(Image.open(a.ir).convert('RGB'), np.float32) / 255.0
        vis = np.asarray(Image.open(a.vis).convert('RGB'), np.float32) / 255.0 if a.vis else None
    P, info = run_chain(ir, vis, mock=(a.mock or not a.ir), out=a.out)
    print(json.dumps(info, ensure_ascii=False, indent=1)[:700])
    print('先验场 P: %s 通道=[eps, var_eps, r_phys, tau, conf, var_r, var_tau]' % (P.shape,))


if __name__ == '__main__':
    main()
