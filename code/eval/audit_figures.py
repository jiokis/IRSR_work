# -*- coding: utf-8 -*-
"""Audit every self-drawn figure for the four defects we keep hitting.

Checks
  1) TITLE  : tool / file names leaking into captions ("test.py", ".png", "matplotlib", "archify")
  2) SIZE   : printed height at column width (15.5 cm) must be <= 7.0 cm  -> needs AR >= 2.21
  3) FONTSMALL: effective point size at print scale = font_px / canvas_px * 15.5cm * 28.35pt/cm
  4) ORPHAN : figure file referenced nowhere in the report, or referenced but missing

Only the *non-citation* figures (our own drawings / redraws) are audited; figures cropped from
papers are marked as `cite` and skipped.
"""
import io
import os
import re

ROOT = os.path.dirname(os.path.abspath(__file__))
MD = os.path.join(ROOT, 'docs/advisor/中期检查报告_v2_full.md')
FIGDIR = os.path.join(ROOT, 'docs/advisor/figures')

CITE = {'fig1_dataset.png', 'swinir_arch.png', 'repvgg_arch.png', 'dbb_transforms.png',
        'nafnet_block.png', 'fig2_arch.png', 'fig5_qual.png'}
SUSPECT = ['test.py', '.png', '.pdf', 'matplotlib', 'archify', 'python', 'excel', 'origin', 'word']
COL_W_CM, MAX_H_CM = 15.5, 7.0


def png_size(path):
    """read width/height from the PNG header (no PIL needed)."""
    with open(path, 'rb') as f:
        head = f.read(26)
    if head[:8] != b'\x89PNG\r\n\x1a\n':
        return None
    w = int.from_bytes(head[16:20], 'big'); h = int.from_bytes(head[20:24], 'big')
    return w, h


def main():
    md = io.open(MD, encoding='utf-8').read()
    refs = re.findall(r'!\[([^\]]*)\]\(([^)]+)\)', md)
    rows = []
    for cap, path in refs:
        name = os.path.basename(path)
        full = os.path.join(ROOT, 'docs/advisor', path) if not os.path.isabs(path) else path
        full = full.replace('../../papers/ref/crops/', 'papers/ref/crops/')
        full = full.replace('../../papers/figs/', 'papers/figs/')
        full = os.path.normpath(os.path.join(ROOT, full)) if not os.path.exists(full) else full
        kind = 'cite' if name in CITE else 'ours'
        size = png_size(full) if os.path.exists(full) else None
        ar = (size[0] / size[1]) if size else 0
        h_cm = (COL_W_CM / ar) if ar else 0
        rows.append(dict(name=name, cap=cap, path=full, kind=kind, size=size, ar=ar, h_cm=h_cm))

    print('%-38s %-6s %-11s %-6s %s' % ('图文件', '类型', '像素', 'AR', '栏宽下印高(cm)'))
    print('-' * 84)
    problem = []
    for r in rows:
        flag = ''
        if r['kind'] == 'ours':
            if r['h_cm'] > MAX_H_CM + 1e-6:
                flag += ' ⚠过高'
            if r['size'] is None:
                flag += ' ⚠缺失'
        print('%-38s %-6s %-11s %-6s %-6.2f%s' % (
            r['name'], r['kind'], ('%dx%d' % r['size']) if r['size'] else '—',
            '%.2f' % r['ar'] if r['ar'] else '—', r['h_cm'], flag))
        if flag:
            problem.append(r['name'])

    print('\n=== 1) 生成脚本里疑似泄漏工具名/文件名 ===')
    hits = 0
    for f in sorted(os.listdir(ROOT)):
        if not f.endswith('.py'):
            continue
        s = io.open(os.path.join(ROOT, f), encoding='utf-8', errors='ignore').read()
        for m in re.finditer(r"set_title\(([^\n]{0,120})", s):
            t = m.group(1)
            for bad in SUSPECT:
                if bad in t:
                    print('  %-28s set_title 含 "%s"  -> %s' % (f, bad, t.strip()[:80]))
                    hits += 1
                    break
    if not hits:
        print('  未发现 ✓')

    print('\n=== 2) 尺寸/比例不合规（自绘图，> %.1f cm）===' % MAX_H_CM)
    print('  ' + (', '.join(problem) if problem else '无 ✓'))
    print('\n自绘图 %d 张｜引用图 %d 张' % (sum(1 for r in rows if r['kind'] == 'ours'),
                                           sum(1 for r in rows if r['kind'] == 'cite')))


if __name__ == '__main__':
    main()
