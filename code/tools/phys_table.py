# -*- coding: utf-8 -*-
"""Physics knowledge table loader.

Reads knowledge/*.yaml. Uses PyYAML when available, otherwise a minimal built-in parser that
handles the exact shape we use (nested maps, scalars, inline lists). Fail loud if a key is missing.
"""
from __future__ import annotations

import io
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KN = os.path.join(ROOT, 'knowledge')


def _mini_yaml(text):
    """Parse the subset used in knowledge/: 'a:', 'a: b', 'a: [x, y]', 'a: "s"', nested by indent."""
    root, stack = {}, [(-1, root)]
    for raw in text.split('\n'):
        if not raw.strip() or raw.lstrip().startswith('#'):
            continue
        indent = len(raw) - len(raw.lstrip(' '))
        line = raw.strip()
        if ':' not in line:
            continue
        k, v = line.split(':', 1)
        k, v = k.strip(), v.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if v == '':
            parent[k] = {}
            stack.append((indent, parent[k]))
        else:
            v = v.split('#')[0].strip()
            if v.startswith('[') and v.endswith(']'):
                items = [x.strip().strip('"').strip("'") for x in v[1:-1].split(',') if x.strip()]
                try:
                    v = [float(x) for x in items]
                except ValueError:
                    v = items
            elif re.fullmatch(r'-?\d+\.\d+', v):
                v = float(v)
            elif re.fullmatch(r'-?\d+', v):
                v = int(v)
            else:
                v = v.strip('"').strip("'")
            parent[k] = v
    return root


def load(name):
    p = os.path.join(KN, name if name.endswith('.yaml') else name + '.yaml')
    text = io.open(p, encoding='utf-8').read()
    try:
        import yaml                                        # noqa
        return yaml.safe_load(text)
    except Exception:
        return _mini_yaml(text)


def eps_table():
    """{class -> (lo, hi, mid, conf, zh)}"""
    t = load('emissivity_table')
    out = {}
    for k, v in t['classes'].items():
        lo, hi = v['eps']
        out[k] = dict(lo=lo, hi=hi, mid=0.5 * (lo + hi), conf=float(v['conf']), zh=v.get('zh', k))
    return out


def spectral_table():
    t = load('spectral_behavior')
    return t['classes']


def prior_reliability():
    """r_physics per class: 1.0 = 物理上适合用跨模态结构引导；0.0 = 不可用（镜面/天空/水）"""
    out = {}
    for k, v in spectral_table().items():
        out[k] = 1.0 if str(v.get('cross_modal_reliable')).lower() == 'true' else 0.0
    return out


if __name__ == '__main__':
    e = eps_table()
    r = prior_reliability()
    print('%-12s %-16s %-6s %-6s %s' % ('class', 'eps[lo,hi]', 'mid', 'conf', 'r_phys'))
    for k, v in e.items():
        print('%-12s [%.2f, %.2f]     %.3f  %.2f   %.1f' % (k, v['lo'], v['hi'], v['mid'], v['conf'], r[k]))
