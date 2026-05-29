"""
Load exp1_plot_data.pt and produce a NeurIPS-ready PDF figure.

Usage:
    python plt-loss.py                          # all models, both panels
    python plt-loss.py --panel left             # all models, left panel only
    python plt-loss.py --model resnet18         # one model, both panels
    python plt-loss.py --model vgg11 --panel right
"""

import argparse
import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

parser = argparse.ArgumentParser()
parser.add_argument('--panel', choices=['left', 'right', 'both'], default='left',
                    help='Which panel(s) to plot')
# parser.add_argument('--model', choices=['resnet18', 'vgg11', 'densenet121', 'all'],
#                     default='all', help='Which model to plot')
parser.add_argument('--model', choices=['resnet18', 'vgg11', 'all'],
                    default='all', help='Which model to plot')

args = parser.parse_args()

# ── Config ────────────────────────────────────────────────────────────────────
BASE      = '/home/mmoslem3/scratch/UE-DD'
SMOOTH_W  = 7   # moving-average window for cosine curve

MODELS = {
    'resnet18':    'extraEXP',
    'vgg11':       'extraEXP-vgg11',
    # 'densenet121': 'extraEXP-DenseNet121',
}

# NeurIPS single-column ≈ 3.25 in; two-panel ≈ 6.75 in
# Single-panel uses a wider aspect ratio to avoid a too-tall look
BOTH_W,   BOTH_H   = 6.75, 2.6
SINGLE_W, SINGLE_H = 3.75, 2.4
FIG_W = BOTH_W   if args.panel == 'both' else SINGLE_W
FIG_H = BOTH_H   if args.panel == 'both' else SINGLE_H

# Colorblind-safe palette (Wong 2011)
COLORS = {
    'clean': '#009E73',   # green
    'DGC':   '#0072B2',   # blue
    'EMN':   '#D55E00',   # vermillion  (SW → EMN label)
    'SW':    '#D55E00',
    'GUE':   '#E69F00',   # orange
    'TUE':   '#CC79A7',   # pink-purple
}
LABELS = {
    'clean': 'Clean',
    'DGC':   'DGC (ours)',
    'EMN':   'EMN',
    'SW':    'EMN',
    'GUE':   'GUE',
    'TUE':   'TUE',
}
# Draw DGC on top so it stays visible
ZORDERS = {'clean': 1, 'SW': 2, 'EMN': 2, 'GUE': 2, 'TUE': 2, 'DGC': 3}

# ── NeurIPS matplotlib style ──────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':        'serif',
    'font.serif':         ['Times New Roman', 'DejaVu Serif'],
    'font.size':          8,
    'axes.titlesize':     12,
    'axes.labelsize':     12,
    'xtick.labelsize':    10,
    'ytick.labelsize':    10,
    'legend.fontsize':    10,
    'legend.framealpha':  0.85,
    'legend.edgecolor':   '0.8',
    'lines.linewidth':    1,
    'axes.linewidth':     1,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'xtick.direction':    'out',
    'ytick.direction':    'out',
    'xtick.major.size':   6,
    'ytick.major.size':   6,
    'xtick.major.width':  0.8,
    'ytick.major.width':  0.8,
    'grid.linewidth':     0.2,
    'grid.alpha':         0.2,
    'figure.dpi':         300,
    'savefig.dpi':        300,
    'pdf.fonttype':       42,   # embed fonts as Type 42 (TrueType) — required by NeurIPS
    'ps.fonttype':        42,
})


# ── Helpers ───────────────────────────────────────────────────────────────────

def smooth(x, w):
    if w <= 1 or len(x) < w:
        return np.asarray(x)
    kernel = np.ones(w) / w
    pad = w // 2
    return np.convolve(np.pad(x, pad, mode='edge'), kernel, mode='valid')


# ── Per-model loop ────────────────────────────────────────────────────────────

model_names = list(MODELS.keys()) if args.model == 'all' else [args.model]

for model_name in model_names:
    exp_dir   = os.path.join(BASE, MODELS[model_name])
    data_file = os.path.join(exp_dir, 'exp1_plot_data.pt')
    out_file  = os.path.join(exp_dir, f'exp1_{model_name}_{args.panel}.pdf')

    data = torch.load(data_file, map_location='cpu', weights_only=False)
    print(f"[{model_name}] Loaded keys: {list(data.keys())}")

    ncols = 2 if args.panel == 'both' else 1
    fig, axes = plt.subplots(1, ncols, figsize=(FIG_W, FIG_H))
    if ncols == 1:
        axes = [axes]

    ax_idx = 0

    # ── Left panel: training loss ──────────────────────────────────────────────
    if args.panel in ('left', 'both'):
        ax = axes[ax_idx]; ax_idx += 1
        for nt, d in data.items():
            epochs = np.asarray(d['epochs'])
            loss   = np.asarray(d['loss'])
            ax.plot(epochs, loss,
                    color=COLORS.get(nt, 'black'),
                    label=LABELS.get(nt, nt),
                    zorder=ZORDERS.get(nt, 1))
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Training Loss (CE)')
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=5))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        ax.grid(True, linestyle='--')
        if model_name == 'resnet18':
            ax.legend(loc='upper right', ncol=1)

    # ── Right panel: gradient cosine similarity ────────────────────────────────
    if args.panel in ('right', 'both'):
        ax = axes[ax_idx]
        for nt, d in data.items():
            cos_epochs = np.asarray(d['cosine_epochs'])
            cosine     = smooth(np.asarray(d['cosine']), SMOOTH_W)
            ax.plot(cos_epochs, cosine,
                    color=COLORS.get(nt, 'black'),
                    label=LABELS.get(nt, nt),
                    zorder=ZORDERS.get(nt, 1))
        ax.axhline(0, color='0.5', linestyle=':', linewidth=0.8, zorder=0)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(r'$\cos(\mathbf{g}_\mathrm{rel},\,\mathbf{g}_\mathrm{tar})$')
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=5))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        ax.grid(True, linestyle='--')
        if model_name == 'resnet18':
            ax.legend(loc='upper right', ncol=1)

    plt.tight_layout(pad=0.5, w_pad=1.5)
    plt.savefig(out_file, bbox_inches='tight')
    print(f"Saved: {out_file}")
    plt.close()
