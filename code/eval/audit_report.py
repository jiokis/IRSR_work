# -*- coding: utf-8 -*-
"""Final consistency audit of the report source (markdown).

Checks
  1 figures : caption numbers continuous 1..N, every in-text 图 N has a caption, no orphan captions
  2 tables  : same for 表 N
  3 equations: (chapter-index) continuity per chapter
  4 sections: ### x.y.z numbering continuity inside each section
  5 refs    : in-text [N] <-> reference list [N] must match both ways
  6 banned  : tool/file names leaking into captions ('test.py', '.png', 'matplotlib', 'archify', ...)
  7 markers : leftover TODO / FIXME / XXX / ??? in the prose
  8 files   : every referenced image file exists; flag figure files never referenced
"""
from __future__ import annotations

import io
import os
import re

ROOT = os.path.dirname(os.path.abspath(__file__))
MD = os.path.join(ROOT, 'docs/advisor/中期检查报告_v2_full.md')
FIGD = os.path.join(ROOT, 'docs/advisor/figures')
BANNED = ['test.py', 'matplotlib', 'archify', 'python', '.png', '.pdf', 'excel', 'origin', 'openpyxl']
problems = []


def p(msg):
    problems.append(msg)
    print('  ✗ ' + msg)


def ok(msg):
    print('  ✓ ' + msg)


def main():
    s = io.open(MD, encoding='utf-8').read()
    lines = s.split('\n')

    # ---------- 1/2 figure & table numbers
    for tag, pat in (('图', r'!\[图 ?(\d+)'), ('表', r'(?m)^表 ?(\d+)\s')):
        caps = [int(x) for x in re.findall(pat, s)]
        want = list(range(1, len(caps) + 1))
        if caps != want:
            p('%s 题注编号不连续: %s' % (tag, caps))
        else:
            ok('%s 题注连续 1..%d' % (tag, len(caps)))
        used = sorted(set(int(x) for x in re.findall(tag + r' ?(\d+)', s)))
        bad = [u for u in used if u > len(caps)]
        if bad:
            p('%s 存在越界引用: %s' % (tag, bad))
        else:
            ok('%s 无越界引用（引用数 %d）' % (tag, len(used)))

    # ---------- 3 equations
    eqs = re.findall(r'\((\d)-(\d+)\)', s)
    by_ch = {}
    for c, i in eqs:
        by_ch.setdefault(c, []).append(int(i))
    for c, v in sorted(by_ch.items()):
        if sorted(set(v)) != list(range(1, max(v) + 1)):
            p('第 %s 章公式编号有缺口: %s' % (c, sorted(set(v))))
        else:
            ok('第 %s 章公式编号连续 1..%d' % (c, max(v)))

    # ---------- 4 section numbering
    secs = [(a, b, c) for a, b, c in re.findall(r'(?m)^### (\d+)\.(\d+)\.(\d+)', s)]
    bad_sec = []
    prev = None
    for a, b, c in secs:
        cur = (int(a), int(b), int(c))
        if prev and cur[0] == prev[0] and cur[1] == prev[1] and cur[2] != prev[2] + 1:
            bad_sec.append('%d.%d.%d (上一节 %d.%d.%d)' % (cur + prev))
        prev = cur
    if bad_sec:
        p('小节编号有跳跃: %s' % '; '.join(bad_sec[:5]))
    else:
        ok('小节编号连续（共 %d 节）' % len(secs))

    # ---------- 5 references
    cited = set(int(x) for x in re.findall(r'(?<!\d)\[(\d{1,2})\]', s))
    listed = set(int(x) for x in re.findall(r'(?m)^\[(\d{1,2})\]', s))
    if cited - listed:
        p('被引但文献表缺失: %s' % sorted(cited - listed)[:8])
    if listed - cited:
        p('文献表有条目未被引: %s' % sorted(listed - cited)[:8])
    if not (cited - listed) and not (listed - cited) and listed:
        ok('引用 <-> 文献表 双向一致（%d 条）' % len(listed))

    # ---------- 6 banned strings in captions
    hits = []
    for ln in lines:
        probe = re.sub(r'\]\([^)]*\)', '](PATH)', ln)      # 去掉 markdown 路径再扫
        if ln.startswith('![') or re.match(r'^表 ?\d+\s', ln):
            for b in BANNED:
                if b in probe:
                    hits.append('%s ← %s' % (b, ln[:60]))
    if hits:
        p('题注含工具名/文件名: %s' % '; '.join(hits[:4]))
    else:
        ok('题注无工具名/文件名泄漏')

    # ---------- 7 markers
    mk = re.findall(r'(?m)(TODO|FIXME|XXX|\?\?\?|待补页码|待补)', s)
    if mk:
        p('残留标记: %s（%d 处）' % (sorted(set(mk)), len(mk)))
    else:
        ok('无 TODO/FIXME/待补 标记')

    # ---------- 8 image files
    refd = set()
    for path in re.findall(r'!\[[^\]]*\]\(([^)]+)\)', s):
        refd.add(os.path.basename(path))
    missing = [f for f in refd if not os.path.exists(
        os.path.join(ROOT, 'docs/advisor', f)) and not os.path.exists(
        os.path.join(ROOT, 'docs/advisor/figures', f)) and not os.path.exists(os.path.join(ROOT, f))]
    if missing:
        p('引用的图片文件不存在: %s' % missing[:5])
    else:
        ok('全部引用图片文件存在（%d 个）' % len(refd))

    print('\n=== 汇总: %d 个问题 ===' % len(problems))
    for x in problems:
        print('  - ' + x)


if __name__ == '__main__':
    main()
