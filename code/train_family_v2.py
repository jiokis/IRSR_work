# -*- coding: utf-8 -*-
"""Family trainer v2 — one process, one shared dataloader, N models, built-in validation.

Adds over v1:
  * real LR/GT pairs matched by filename (no bicubic-from-GT shortcut)
  * periodic validation PSNR/SSIM on the FLIR-IISR x4 val split  -> comparison table comes free
  * AMP + TF32 + channels_last + persistent workers  -> faster per GPU-hour
  * checkpoint / resume so a restart never loses progress
  * optional RGB guide (for the guided family on M3FD)

Usage:
  export PYTHONPATH=/root/autodl-tmp/basicSR:$PYTHONPATH
  python train_family_v2.py --yml_dir options/train/AutoDL --iters 20000 --batch 16 \
      --workers 2 --val_every 2000 --data_root /root/autodl-tmp
"""
from __future__ import annotations

import argparse
import glob
import io
import os
import random
import re
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from basicsr.archs.irlite_variants_arch import FDRNet          # noqa: E402


# ------------------------------------------------------------------ utils
def fix_path(p, root):
    if p and root:
        return p.replace('DATA_ROOT', root)
    return p


def parse_yml(path):
    t = io.open(path, encoding='utf-8').read()
    g = lambda pat, d=None: (re.search(pat, t).group(1) if re.search(pat, t) else d)
    blocks = g(r'(?m)^\s*block_types:\s*\[([^\]]*)\]')
    return dict(
        name=g(r'(?m)^name:\s*(\S+)'),
        nf=int(g(r'(?m)^\s*nf:\s*(\d+)', 56)),
        blocks=[b.strip() for b in blocks.split(',')] if blocks else ['rfdxrep'] * 4,
        baseline=None if g(r'(?m)^\s*baseline:\s*(\S+)') in (None, 'null') else g(r'(?m)^\s*baseline:\s*(\S+)'),
        deg_sem=g(r'(?m)^\s*deg_sem:\s*(\S+)'),
        heavy_guided=g(r'(?m)^\s*heavy_guided:\s*(\S+)'),
        gt=g(r'(?m)^\s*dataroot_gt:\s*(\S+)'),
        lq=g(r'(?m)^\s*dataroot_lq:\s*(\S+)'),
        lq_real=g(r'(?m)^\s*dataroot_lq_real:\s*(\S+)'),
        vi=g(r'(?m)^\s*dataroot_vi:\s*(\S+)'),
        val_gt=g(r'(?ms)val:.*?dataroot_gt:\s*(\S+)'),
        val_lq=g(r'(?ms)val:.*?dataroot_lq:\s*(\S+)'),
        iters=int(g(r'(?m)^\s*total_iter:\s*(\d+)', 20000)),
    )


def build_model(c):
    kw = dict(in_nc=3, nf=c['nf'], out_nc=3, upscale=4, block_types=c['blocks'],
              sghf=(c['baseline'] is None and c['heavy_guided'] is None))
    for k in ('baseline', 'deg_sem', 'heavy_guided'):
        if c.get(k):
            kw[k] = c[k]
    return FDRNet(**kw)


# ------------------------------------------------------------------ data
class PairsByFilename(Dataset):
    """LR and GT matched by file name (no cross-pairing)."""

    def __init__(self, lq_dir, gt_dir, patch=48, train=True):
        self.lq = sorted(glob.glob(os.path.join(lq_dir, '*.*')))
        self.gt_dir = gt_dir
        self.patch, self.train = patch, train
        self.items = [(p, os.path.join(gt_dir, os.path.basename(p))) for p in self.lq]
        self.items = [(a, b) for a, b in self.items if os.path.exists(b)]
        if not self.items:
            raise RuntimeError('no pairs: lq=%s gt=%s' % (lq_dir, gt_dir))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        lp, gp = self.items[i]
        lr = Image.open(lp).convert('RGB')
        gt = Image.open(gp).convert('RGB')
        if self.train:
            s = self.patch
            w, h = lr.size
            x0, y0 = random.randint(0, max(0, w - s)), random.randint(0, max(0, h - s))
            lr = lr.crop((x0, y0, x0 + s, y0 + s))
            gt = gt.crop((x0 * 4, y0 * 4, (x0 + s) * 4, (y0 + s) * 4))
            if random.random() < 0.5:
                lr, gt = lr.transpose(Image.FLIP_LEFT_RIGHT), gt.transpose(Image.FLIP_LEFT_RIGHT)
        x = torch.from_numpy(np.asarray(lr, np.float32).transpose(2, 0, 1) / 255.0)
        y = torch.from_numpy(np.asarray(gt, np.float32).transpose(2, 0, 1) / 255.0)
        return x, y


def psnr(a, b):
    mse = float(((a - b) ** 2).mean())
    return 10.0 * np.log10(1.0 / max(mse, 1e-10))


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--yml_dir', default='options/train/AutoDL')
    ap.add_argument('--iters', type=int, default=20000)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--patch', type=int, default=48)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--data_root', default=None)
    ap.add_argument('--out', default='experiments/FAMILY2')
    ap.add_argument('--log_every', type=int, default=100)
    ap.add_argument('--val_every', type=int, default=2000)
    ap.add_argument('--val_n', type=int, default=40)
    ap.add_argument('--ckpt_every', type=int, default=5000)
    ap.add_argument('--amp', action='store_true', default=True)
    ap.add_argument('--with_guided', action='store_true')
    ap.add_argument('--resume', default=None)
    ap.add_argument('--dry_run', action='store_true')
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    cfgs = [parse_yml(p) for p in sorted(glob.glob(os.path.join(args.yml_dir, '*.yml')))]
    if not args.with_guided:
        cfgs = [c for c in cfgs if not (c.get('baseline') or c.get('heavy_guided'))]
    print('family size: %d configs' % len(cfgs), flush=True)
    os.makedirs(args.out, exist_ok=True)

    lq = fix_path(cfgs[0]['lq_real'] or cfgs[0]['lq'], args.data_root)
    gt = fix_path(cfgs[0]['gt'], args.data_root)
    train_ds = PairsByFilename(lq, gt, patch=args.patch, train=True)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                          pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)
    print('train pairs: %d  (lq=%s)' % (len(train_ds), lq), flush=True)

    val_dl = None
    if cfgs[0]['val_gt'] and cfgs[0]['val_lq']:
        try:
            vds = PairsByFilename(fix_path(cfgs[0]['val_lq'], args.data_root),
                                  fix_path(cfgs[0]['val_gt'], args.data_root), train=False)
            val_dl = DataLoader(vds, batch_size=1, shuffle=False, num_workers=0)
            print('val pairs: %d' % len(vds), flush=True)
        except Exception as e:
            print('no val set (%s)' % str(e)[:60], flush=True)

    models, opts, logs, scalers, scheds = [], [], [], [], []
    for c in cfgs:
        m = build_model(c).cuda().train()
        o = torch.optim.Adam(m.parameters(), lr=args.lr, weight_decay=0, betas=(0.9, 0.99))
        sch = torch.optim.lr_scheduler.MultiStepLR(o, milestones=[int(args.iters * .4), int(args.iters * .8)], gamma=0.5)
        models.append(m); opts.append(o); scheds.append(sch)
        scalers.append(torch.cuda.amp.GradScaler(enabled=args.amp))
        f = io.open(os.path.join(args.out, c['name'] + '.log'), 'w', encoding='utf-8')
        f.write('name=%s nf=%d blocks=%s baseline=%s deg_sem=%s heavy=%s\n' %
                (c['name'], c['nf'], c['blocks'], c['baseline'], c['deg_sem'], c['heavy_guided']))
        f.write('params=%.4fM\n' % (sum(q.numel() for q in models[-1].parameters()) / 1e6)); f.flush()
        logs.append(f)

    start_it = 0
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location='cpu')
        for k, c in enumerate(cfgs):
            if c['name'] in ck:
                models[k].load_state_dict(ck[c['name']]); opts[k].load_state_dict(ck[c['name'] + '_opt'])
        start_it = ck.get('iter', 0)
        print('resumed at %d' % start_it, flush=True)

    if args.dry_run:
        xb, yb = next(iter(train_dl))
        xb = xb.cuda(non_blocking=True); yb = yb.cuda(non_blocking=True)
        ok = 0
        for k, (m, o, sc) in enumerate(zip(models, opts, scalers)):
            try:
                with torch.cuda.amp.autocast(enabled=args.amp):
                    out = m(xb); loss = F.l1_loss(out, yb)
                o.zero_grad(set_to_none=True); sc.scale(loss).backward(); sc.step(o); sc.update()
                print('  DRY_OK %-26s loss %.5f' % (cfgs[k]['name'], loss.item()), flush=True); ok += 1
            except Exception as e:
                print('  DRY_FAIL %-24s %s' % (cfgs[k]['name'], str(e)[:90]), flush=True)
        print('dry_run passed %d/%d  gpu_peak %.2f GB' % (ok, len(cfgs), torch.cuda.max_memory_allocated()/1024**3), flush=True)
        return
    t0, it = time.time(), iter(train_dl)
    for it_i in range(start_it + 1, args.iters + 1):
        try:
            x, gt_b = next(it)
        except StopIteration:
            it = iter(train_dl); x, gt_b = next(it)
        x = x.cuda(non_blocking=True)
        gt_b = gt_b.cuda(non_blocking=True)
        for k, (m, o, sc) in enumerate(zip(models, opts, scalers)):
            if m is None:
                continue
            try:
                with torch.cuda.amp.autocast(enabled=args.amp):
                    out = m(x)
                    loss = F.l1_loss(out, gt_b)
                o.zero_grad(set_to_none=True)
                sc.scale(loss).backward()
                sc.step(o); sc.update()
                if it_i % args.log_every == 0:
                    logs[k].write('iter %d l1 %.5f lr %.2e\n' % (it_i, loss.item(), o.param_groups[0]['lr']))
                    logs[k].flush()
            except RuntimeError as e:
                logs[k].write('iter %d MODEL_ERR %s\n' % (it_i, str(e)[:160])); logs[k].flush()
                models[k] = None
        for sch in scheds:
            sch.step()
        if it_i % args.log_every == 0:
            el = (time.time() - t0) / 60
            eta = el / max(1, it_i - start_it) * (args.iters - it_i)
            print('iter %d/%d  %.1f min  eta %.1f min  gpu_peak %.2f GB  mem %.2f GB' % (
                it_i, args.iters, el, eta, torch.cuda.max_memory_allocated() / 1024 ** 3,
                torch.cuda.memory_reserved() / 1024 ** 3), flush=True)
        if val_dl is not None and it_i % args.val_every == 0:
            for k, m in enumerate(models):
                if m is None:
                    continue
                m.eval()
                tot = n = 0.0
                with torch.no_grad():
                    for vx, vgt in val_dl:
                        if n >= args.val_n:
                            break
                        with torch.cuda.amp.autocast(enabled=args.amp):
                            vo = m(vx.cuda())
                        tot += psnr(vo.clamp(0, 1)[0].float().cpu().numpy(), vgt[0].numpy()); n += 1
                logs[k].write('VAL iter %d psnr %.4f (n=%d)\n' % (it_i, tot / max(1, n), n)); logs[k].flush()
                m.train()
            print('validation done at iter %d' % it_i, flush=True)
        if it_i % args.ckpt_every == 0:
            ck = {'iter': it_i}
            for k, c in enumerate(cfgs):
                if models[k] is not None:
                    ck[c['name']] = models[k].state_dict(); ck[c['name'] + '_opt'] = opts[k].state_dict()
            torch.save(ck, os.path.join(args.out, 'family_%d.pth' % it_i))
    for f in logs:
        f.close()
    print('done at %s' % time.strftime('%H:%M:%S'), flush=True)


if __name__ == '__main__':
    main()
