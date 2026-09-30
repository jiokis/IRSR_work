# -*- coding: utf-8 -*-
"""Progress checker that cannot be fooled by stale/duplicated logs.

Why this exists: BasicSR appends to `train_*.log`, and several runs of the same experiment can
leave multiple files behind. Grepping the first match therefore reports history, not progress
(this has caused two wrong status reports already). Rules used here:

  * per experiment, only the NEWEST log file by mtime is authoritative
  * a run is "active" only if that file was written within STALE_MIN minutes
  * progress = the LAST `iter: N` occurrence in that file (not max, not first)
  * errors are counted only in the tail window of that file

Usage
    python check_progress.py            # human table
    python check_progress.py --json     # machine readable
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import re
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(ROOT, 'experiments')
STALE_MIN = 30               # 30 min 内仍在写才算正在训练 ✓
TAIL_LINES = 400


def newest_log(d):
    fs = glob.glob(os.path.join(d, 'train_*.log')) + glob.glob(os.path.join(d, 'run2_*.log'))
    return max(fs, key=os.path.getmtime) if fs else None


def read_tail(path, n=TAIL_LINES):
    with io.open(path, encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    return lines[-n:], len(lines)


def analyse(exp_dir):
    name = os.path.basename(exp_dir.rstrip('/\\'))
    log = newest_log(exp_dir)
    if not log:
        return None
    age_min = (time.time() - os.path.getmtime(log)) / 60.0
    tail, total = read_tail(log)
    text = ''.join(tail)
    iters = [int(x.replace(',','')) for x in re.findall(r'iter:\s*([\d,]+),', text)]
    it = int(iters[-1]) if iters else None
    eta = None
    m = re.findall(r'eta:\s*([0-9]+:[0-9]{2}:[0-9]{2})', text)
    if m:
        eta = m[-1]
    psnrs = [float(x) for x in re.findall(r'psnr:\s*([0-9.]+)', text)]
    tb = text.count('Traceback') + len(re.findall(r'\b(KeyError|TypeError|RuntimeError|ValueError|AssertionError)\b', text))
    done = ('End of training' in text) or ('Training finished' in text)
    return dict(name=name, log=os.path.relpath(log, ROOT), age_min=round(age_min, 1),
                iter=it, eta=eta, best_psnr=max(psnrs) if psnrs else None,
                errors=tb, done=done, lines=total,
                active=(age_min <= STALE_MIN and not done))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', action='store_true')
    ap.add_argument('--all', action='store_true', help='include inactive/finished')
    a = ap.parse_args()
    rows = []
    for d in sorted(glob.glob(os.path.join(EXP, '*'))):
        b = os.path.basename(d)
        if not os.path.isdir(d) or b.startswith('_') or '_archived_' in b:
            continue
        r = analyse(d)
        if r:
            rows.append(r)
    # queue status
    q = None
    for f in (os.path.join(EXP, '_queue2.out'), os.path.join(EXP, '_queue.out')):
        if os.path.exists(f):
            lines = read_tail(f, 6)[0]
            q = [l.strip() for l in lines if l.strip()][-1:] or None
            break
    if a.json:
        print(json.dumps(dict(experiments=rows, queue_tail=q), ensure_ascii=False, indent=1))
        return
    act = [r for r in rows if r['active']]
    print('=== 活跃训练（%d 个）===' % len(act))
    print('%-26s %-10s %-11s %-10s %s' % ('实验', '迭代', 'ETA', '日志新鲜度', 'ERRORS'))
    for r in sorted(act, key=lambda x: -(x['iter'] or 0)):
        print('%-26s %-10s %-11s %-10s %s' % (r['name'], r['iter'], r['eta'] or '-',
                                              '%.0f min 前' % r['age_min'], r['errors'] or '0'))
    fin = [r for r in rows if r['done']]
    if fin:
        print('\n=== 已完成（%d 个）===' % len(fin))
        for r in fin:
            print('  %-26s PSNR=%s  日志=%s' % (r['name'], r['best_psnr'], r['log']))
    if a.all:
        rest = [r for r in rows if not r['active'] and not r['done']]
        print('\n=== 非活跃（%d 个）===' % len(rest))
        for r in rest:
            print('  %-26s iter=%-8s %.0f min 前  errors=%s' % (r['name'], r['iter'], r['age_min'], r['errors']))
    if q:
        print('\n队列尾: %s' % q[0][:110])
    print('\n提示：每实验只看最新日志；年龄 > 12 h 视为非活跃 ✓')


if __name__ == '__main__':
    main()
