# -*- coding: utf-8 -*-
"""Serial evaluation of the unified M3FD guidance batch (10 rows), one model at a time.

Why this script exists instead of eval_full_table.py:
  * eval_full_table.py is hard-coded to FLIR-IISR and calls `m(lr)` WITHOUT the visible guide, which
    is wrong for every guided row (they need `vi=`).
  * Here each row's own yml is parsed, so the model is rebuilt with exactly the architecture kwargs
    it was trained with (nf / block_types / sghf / baseline / ...). The checkpoint is then loaded and
    evaluated on the M3FD val split (gt / lq / vi, 300 pairs) with crop_border=4 for every row, so
    all numbers share one protocol.

Usage:
    python eval_m3fd_uni.py                # all rows that have a checkpoint
    python eval_m3fd_uni.py U00 U09        # subset (prefix match)
"""
from __future__ import annotations

import glob
import io
import os
import re
import sys

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from basicsr.archs.irlite_variants_arch import FDRNet          # noqa: E402
from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim  # noqa: E402

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
VAL = 'D:/datasets/M3FD_SR/val'
CROP = 4
ROWS = ['U00_noguide', 'U01_concat1x1', 'U02_ca_msff', 'U03_coarse2fine', 'U04_caf_gate',
        'U05_spatattn', 'U06_corrmatch', 'U07_bidir', 'U08_nogate', 'U09_ours']


def parse_yml(path):
    t = io.open(path, encoding='utf-8').read()
    kw = {}
    m = re.search(r'(?m)^\s*nf:\s*(\d+)', t)
    if m:
        kw['nf'] = int(m.group(1))
    m = re.search(r'(?m)^\s*upscale:\s*(\d+)', t)
    if m:
        kw['upscale'] = int(m.group(1))
    m = re.search(r'(?m)^\s*block_types:\s*\[([^\]]*)\]', t)
    if m:
        kw['block_types'] = [x.strip() for x in m.group(1).split(',') if x.strip()]
    m = re.search(r'(?m)^\s*sghf:\s*(\S+)', t)
    if m:
        kw['sghf'] = m.group(1).lower() == 'true'
    m = re.search(r'(?m)^\s*baseline:\s*(\S+)', t)
    if m and m.group(1).lower() not in ('none', 'null', '~'):
        kw['baseline'] = m.group(1)
    for key in ('heavy_guided', 'deg_sem'):
        m = re.search(r'(?m)^\s*%s:\s*(\S+)' % key, t)
        if m and m.group(1).lower() not in ('none', 'null', '~'):
            kw[key] = m.group(1)
    return kw


def np8(t):
    return (t[0].detach().cpu().numpy().transpose(1, 2, 0).clip(0, 1) * 255).round().astype(np.uint8)


def pick_ckpt(row):
    fs = glob.glob(os.path.join(ROOT, 'experiments', row, 'models', 'net_g_*.pth'))
    if not fs:
        return None
    def it(p):
        m = re.search(r'net_g_(\d+)\.pth', os.path.basename(p))
        return int(m.group(1)) if m else -1
    return max(fs, key=it)


def main():
    want = sys.argv[1:]
    gts = sorted(glob.glob(os.path.join(VAL, 'gt', '*')))
    print('val pairs: %d   device: %s   crop_border=%d' % (len(gts), DEV, CROP))
    print('%-16s %-9s %-9s %s' % ('row', 'PSNR', 'SSIM', 'checkpoint'))
    for row in ROWS:
        if want and not any(row.startswith(w) for w in want):
            continue
        ck = pick_ckpt(row)
        yml = os.path.join(ROOT, 'options/train/M3FD_UNI', row + '.yml')
        if not ck or not os.path.exists(yml):
            print('%-16s %-9s %-9s %s' % (row, '-', '-', 'no ckpt/yml'))
            continue
        kw = parse_yml(yml)
        net = FDRNet(in_nc=3, out_nc=3, **kw).to(DEV).eval()
        sd = torch.load(ck, map_location='cpu')
        net.load_state_dict(sd['params'] if isinstance(sd, dict) and 'params' in sd else sd, strict=False)
        ps, ss, n = [], [], 0
        for hp in gts:
            stem = os.path.splitext(os.path.basename(hp))[0]
            lp = os.path.join(VAL, 'lq', stem + '.png')
            vp = os.path.join(VAL, 'vi', stem + '.png')
            if not os.path.exists(lp):
                continue
            gt = TF.to_tensor(Image.open(hp).convert('RGB')).unsqueeze(0).to(DEV)
            lr = TF.to_tensor(Image.open(lp).convert('RGB')).unsqueeze(0).to(DEV)
            with torch.no_grad():
                if kw.get('sghf') and os.path.exists(vp):
                    vi = TF.to_tensor(Image.open(vp).convert('RGB')).unsqueeze(0).to(DEV)
                    out = net(lr, vi=vi)
                else:
                    out = net(lr)
            h = min(out.shape[-2], gt.shape[-2]); w = min(out.shape[-1], gt.shape[-1])
            a, b = np8(out)[:h, :w], np8(gt)[:h, :w]
            ps.append(calculate_psnr(a, b, crop_border=CROP, test_y_channel=False))
            ss.append(calculate_ssim(a, b, crop_border=CROP, test_y_channel=False))
            n += 1
        print('%-16s %-9.4f %-9.4f %s (n=%d)' % (row, float(np.mean(ps)), float(np.mean(ss)),
                                                 os.path.basename(ck), n))


if __name__ == '__main__':
    main()
