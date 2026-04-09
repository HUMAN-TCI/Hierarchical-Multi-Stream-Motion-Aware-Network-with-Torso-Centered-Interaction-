"""
make_comparison_figure.py

Assembles multiple query ghost figures into a single paper-ready
comparison row — like the MDM/MotionDiffuse paper figures.

Usage:
    python make_comparison_figure.py \
        --query_dirs outputs/renders_smpl/test/22 \
                     outputs/renders_smpl/test/45 \
                     outputs/renders_smpl/test/44 \
                     outputs/renders_smpl/test/51 \
                     outputs/renders_smpl/test/34 \
                     outputs/renders_smpl/test/56 \
        --output     outputs/renders_smpl/test/comparison_figure.png \
        --highlight  0          # index of box to highlight with green border (query)
        --dpi        200
"""

import os
import re
import argparse
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch


BORDER_GREEN  = '#4CAF50'
BORDER_RED    = '#cc3333'   # dashed divider colour (not used as box border)
BOX_BG        = '#f0efec'
CAPTION_COLOR = '#1a1a1a'


def _read_caption(query_dir):
    """Read desc.txt and return the first clean human sentence."""
    desc_file = os.path.join(query_dir, 'desc.txt')
    if not os.path.exists(desc_file):
        return os.path.basename(query_dir)
    with open(desc_file, encoding='utf-8') as f:
        lines = f.readlines()
    # First line is the description
    raw = lines[0].strip() if lines else ''
    # Strip metrics
    raw = re.sub(r'\(nDCG[^)]*\)', '', raw)
    raw = re.sub(r'spacy\s*=[\d.]+', '', raw)
    raw = re.sub(r'spice\s*=[\d.]+', '', raw)
    raw = re.sub(r'^\d+;\s*', '', raw)
    if ';' in raw:
        parts = raw.split(';')
        for p in parts:
            p = p.strip()
            if not any(k in p for k in ['nDCG','spacy','spice','=']):
                if len(p) > 8:
                    raw = p
                    break
    raw = raw.strip(' .,;)(')
    raw = ' '.join(raw.split())
    if raw:
        raw = raw[0].upper() + raw[1:]
    return raw


def _find_ghost_png(query_dir):
    """Find the ghost PNG in a query directory."""
    for f in sorted(os.listdir(query_dir)):
        if f.endswith('_ghost.png') or f.endswith('ghost.png'):
            return os.path.join(query_dir, f)
    # Fall back: any PNG
    for f in sorted(os.listdir(query_dir)):
        if f.endswith('.png'):
            return os.path.join(query_dir, f)
    return None


def _wrap(text, max_chars=28):
    """Wrap text to max_chars per line."""
    words = text.split()
    lines, line = [], []
    for w in words:
        if sum(len(x)+1 for x in line) + len(w) > max_chars:
            lines.append(' '.join(line))
            line = [w]
        else:
            line.append(w)
    if line:
        lines.append(' '.join(line))
    return '\n'.join(lines)


def make_comparison_figure(
    query_dirs,
    output_path,
    highlight_idx=0,          # which box gets the green border (the query)
    box_width=2.8,            # inches per box
    box_height=3.2,           # inches for the image area
    caption_height=1.0,       # inches for caption below each box
    dpi=200,
    add_dividers=True,        # red dashed dividers between groups
    divider_every=None,       # e.g. 3 = divider after every 3 boxes
    bg_color='#e8e7e4',
):
    n = len(query_dirs)
    fig_w = box_width * n + 0.3
    fig_h = box_height + caption_height + 0.4

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=bg_color, dpi=dpi)

    pad       = 0.12 / fig_w    # outer padding fraction
    box_w_frac = (1.0 - 2*pad) / n
    box_h_img = box_height / fig_h
    box_h_cap = caption_height / fig_h
    bot_cap   = 0.04
    bot_img   = bot_cap + box_h_cap

    for col, qdir in enumerate(query_dirs):
        ghost_path = _find_ghost_png(qdir)
        caption    = _read_caption(qdir)

        left  = pad + col * box_w_frac
        right = left + box_w_frac

        # ── Image axes ──────────────────────────────────────────────────────
        img_margin = 0.008
        ax_img = fig.add_axes([
            left  + img_margin,
            bot_img,
            box_w_frac - 2*img_margin,
            box_h_img
        ])

        if ghost_path and os.path.exists(ghost_path):
            img = np.array(Image.open(ghost_path).convert('RGB'))
            ax_img.imshow(img)
        else:
            ax_img.set_facecolor('#cccccc')
            ax_img.text(0.5, 0.5, 'No image', ha='center', va='center',
                        transform=ax_img.transAxes, fontsize=8, color='#666')

        ax_img.set_xticks([]); ax_img.set_yticks([])
        for spine in ax_img.spines.values():
            spine.set_visible(False)

        # ── Box outline ──────────────────────────────────────────────────────
        is_highlight = (col == highlight_idx)
        border_color = BORDER_GREEN if is_highlight else '#bbbbbb'
        border_lw    = 2.5          if is_highlight else 1.0

        box_rect = FancyBboxPatch(
            (left + 0.003, bot_img - 0.004),
            box_w_frac - 0.006,
            box_h_img  + 0.008,
            boxstyle='square,pad=0',
            linewidth=border_lw,
            edgecolor=border_color,
            facecolor='none',
            transform=fig.transFigure,
            clip_on=False
        )
        fig.add_artist(box_rect)

        # ── Caption ──────────────────────────────────────────────────────────
        wrapped = _wrap(caption, max_chars=26)
        cx = left + box_w_frac / 2
        cy = bot_cap + box_h_cap * 0.45

        fig.text(
            cx, cy, wrapped,
            ha='center', va='center',
            fontsize=6.5,
            fontfamily='DejaVu Serif',
            fontstyle='italic',
            color=CAPTION_COLOR,
            transform=fig.transFigure,
            wrap=False
        )

    # ── Red dashed dividers ──────────────────────────────────────────────────
    if add_dividers and divider_every:
        for k in range(1, n):
            if k % divider_every == 0:
                x_div = pad + k * box_w_frac
                line = plt.Line2D(
                    [x_div, x_div],
                    [bot_img - 0.01, bot_img + box_h_img + 0.01],
                    transform=fig.transFigure,
                    color=BORDER_RED, linewidth=1.2,
                    linestyle='--', clip_on=False
                )
                fig.add_artist(line)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved comparison figure: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_dirs', nargs='+', required=True,
                        help='List of query output directories')
    parser.add_argument('--output', type=str,
                        default='outputs/renders_smpl/test/comparison_figure.png')
    parser.add_argument('--highlight', type=int, default=0,
                        help='Index of box to highlight with green border')
    parser.add_argument('--dpi', type=int, default=200)
    parser.add_argument('--divider_every', type=int, default=None,
                        help='Add red dashed divider after every N boxes')
    parser.add_argument('--box_width',   type=float, default=2.8)
    parser.add_argument('--box_height',  type=float, default=3.2)
    args = parser.parse_args()

    make_comparison_figure(
        query_dirs    = args.query_dirs,
        output_path   = args.output,
        highlight_idx = args.highlight,
        dpi           = args.dpi,
        divider_every = args.divider_every,
        box_width     = args.box_width,
        box_height    = args.box_height,
    )
