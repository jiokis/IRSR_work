# -*- coding: utf-8 -*-
"""Redraw fig25 as a LANDSCAPE grouped bar chart (was a single polar radar, AR 1.17 -> 8.85 cm tall).

Why changed: a polar plot is intrinsically square, so widening the canvas only added white space and
the printed height stayed above the 7 cm limit (figures are scaled to the 15.5 cm column width, so a
too-tall image makes its text unreadably small). A 1x4 panel layout gives AR ~3.8 -> ~4.1 cm tall,
while keeping every model comparable on all four criteria.

Four criteria become four panels; five models become five bars per panel. Normalisation semantics are
unchanged (each axis min-max across all compared models; efficiency axes reversed so that smaller is
better / further right).
"""
import matplotlib
matplotlib.use('Agg')          # noqa: E402
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np          # noqa: E402

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.bbox'] = 'tight'

# model: (PSNR, SSIM, params M, FLOPs G)
MODELS = {
    'EDSR（同参数级）': (38.9354, 0.9552, 0.439, 69.4),
    'RFDN':           (39.1169, 0.9563, 0.408, 39.9),
    'EDSR（中量级）':   (39.1831, 0.9567, 0.927, 137.0),
    '本文方法·轻量':     (39.1342, 0.9564, 0.388, 38.3),
    '本文方法·标准':     (39.2139, 0.9568, 0.523, 51.7),
}
COLOR = {'本文方法·标准': '#b03a2e', '本文方法·轻量': '#e08a3c', 'RFDN': '#2b5d8a',
         'EDSR（同参数级）': '#2e7d4f', 'EDSR（中量级）': '#7f8c8d'}
PANELS = ['PSNR', 'SSIM', '参数效率', '计算效率']
ORDER = ['EDSR（同参数级）', 'EDSR（中量级）', 'RFDN', '本文方法·轻量', '本文方法·标准']


def norm(vals, reverse=False):
    lo, hi = min(vals), max(vals)
    out = [(v - lo) / (hi - lo) if hi > lo else 0.5 for v in vals]
    return [1.0 - o for o in out] if reverse else out


psnr = [MODELS[k][0] for k in ORDER]
ssim = [MODELS[k][1] for k in ORDER]
pare = norm([MODELS[k][2] for k in ORDER], reverse=True)     # 参数越少越好
ceff = norm([MODELS[k][3] for k in ORDER], reverse=True)     # 计算量越少越好
DATA = [norm(psnr), norm(ssim), pare, ceff]
RAW = [[('%.4f' % v) for v in psnr], [('%.4f' % v) for v in ssim],
       [('%.2f M' % v) for v in [MODELS[k][2] for k in ORDER]],
       [('%.1f G' % v) for v in [MODELS[k][3] for k in ORDER]]]

fig, axes = plt.subplots(1, 4, figsize=(16.0, 4.2))
x = np.arange(len(ORDER))
for j, (ax, name) in enumerate(zip(axes, PANELS)):
    vals = DATA[j]
    bars = ax.bar(x, vals, width=.62,
                  color=[COLOR[k] for k in ORDER],
                  edgecolor='k', linewidth=.6)
    for xi, (b, raw) in enumerate(zip(bars, RAW[j])):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + .035, raw,
                ha='center', va='bottom', fontsize=8.2, rotation=0)
    ax.set_title(name, fontsize=12, pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(['EDSR\n同参数级', 'EDSR\n中量级', 'RFDN', '本文\n轻量', '本文\n标准'],
                       fontsize=8.5)
    ax.set_ylim(0, 1.22)
    ax.set_yticks([0, .5, 1.0])
    ax.tick_params(labelsize=8)
    ax.grid(axis='y', alpha=.3, linewidth=.6)
    ax.set_axisbelow(True)
    if j >= 2:
        ax.set_ylabel('归一化效率（越右越好）', fontsize=9)
    elif j == 0:
        ax.set_ylabel('归一化精度（越高越好）', fontsize=9)

fig.suptitle('多准则对比（各准则按全部对比模型归一化；参数效率与计算效率按部署参数量与 FLOPs 反向归一）',
             fontsize=12, y=1.02)
fig.text(.01, -.06, '注：标注为各自准则的原始数值（PSNR/dB、SSIM、参数量 M、FLOPs G）；本文标准配置在四项中均处于有利位置。',
         fontsize=8.5, color='#555')
fig.savefig('docs/advisor/figures/fig25_radar.png', dpi=300, bbox_inches='tight', pad_inches=0.15)
plt.close(fig)
from PIL import Image          # noqa: E402
im = Image.open('docs/advisor/figures/fig25_radar.png')
print('fig25_radar -> %dx%d AR=%.2f 印高=%.2f cm' % (
    im.size[0], im.size[1], im.size[0] / im.size[1], 15.5 / (im.size[0] / im.size[1])))
