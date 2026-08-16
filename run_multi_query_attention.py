"""
run_multi_query_attention.py
=============================
Batch-runs attention visualization over MULTIPLE queries in one go,
and adds a "Torso Importance" summary that ties directly to the same
keyframes you already show in your skeleton/SMPL ghost figures.

WHY THIS MATTERS FOR YOUR PAPER:
Your AttentionFuse uses Q=limb, K/V=torso — so `att` IS torso attention:
it tells you which TORSO FRAMES each limb relied on. Instead of just
4 separate per-limb curves, this script adds one extra combined panel:
  "Torso Frame Importance" = mean attention across all 4 limbs,
  with the SAME keyframe markers (t=0, peak, ..., t=T-1) used in your
  SMPL/skeleton ghost qualitative figures — so readers can directly
  cross-reference: "the model attended most to frame 65 → here is
  that exact pose in the SMPL figure."

USAGE:
    python run_multi_query_attention.py \
        --cfg  "runs/.../.hydra/config.yaml" \
        --ckpt "runs/.../best_models/best_model_metric_all.pth" \
        --set  test \
        --query_ids 22 45 44 51 34 56 86 97 \
        --out_dir outputs/attention_maps \
        --n_keyframes 6
"""

import os, sys, argparse
from pathlib import Path

import torch
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap

# ── import your existing, already-working functions ──────────────────────────
# Put this file in the SAME folder as step_by_step_attention.py
from step_by_step_attention import (
    load_model,
    patch_attention_fuse,
    motion_from_dataset,
    extract_attention,
)

PART_COLORS = {
    'Right Arm': '#c0392b',
    'Left Arm':  '#e74c3c',
    'Right Leg': '#27ae60',
    'Left Leg':  '#2ecc71',
}
BG = '#fafafa'


# ══════════════════════════════════════════════════════════════════════════════
# Keyframe extraction — SAME algorithm you use for the SMPL/skeleton ghost figs
# so the frame numbers line up across all your qualitative results
# ══════════════════════════════════════════════════════════════════════════════
def extract_keyframes_from_attention(total_att, n_keys=6):
    T = len(total_att)
    cum = np.cumsum(total_att)
    if cum[-1] == 0:
        return list(np.linspace(0, T - 1, n_keys, dtype=int))
    cum_n = cum / cum[-1]
    targets = np.linspace(0, 1, n_keys)
    idxs = sorted(set(
        [0] + [int(np.argmin(np.abs(cum_n - a))) for a in targets] + [T - 1]
    ))
    while len(idxs) > n_keys:
        gaps = [cum[idxs[i+1]] - cum[idxs[i]] for i in range(1, len(idxs)-1)]
        idxs.pop(1 + int(np.argmin(gaps)))
    return idxs


# ══════════════════════════════════════════════════════════════════════════════
# Per-query figure — 4 limb curves + Torso Importance panel (with keyframes)
# ══════════════════════════════════════════════════════════════════════════════
def plot_query_attention(att_weights, caption, T, out_dir, tag, n_keyframes=6):
    os.makedirs(out_dir, exist_ok=True)
    parts  = list(att_weights.keys())
    matrix = np.stack([att_weights[p] for p in parts], axis=0)
    frames = np.arange(T)

    # Torso importance = mean attention across all 4 limbs at each torso frame
    torso_importance = matrix.mean(axis=0)
    s = torso_importance.sum()
    if s > 0:
        torso_importance = torso_importance / s

    keyframes = extract_keyframes_from_attention(torso_importance, n_keyframes)

    cmap_heat = LinearSegmentedColormap.from_list(
        'a', ['#ffffff','#fef9e7','#f9ca74','#e67e22','#c0392b','#7b241c'])

    fig = plt.figure(figsize=(18, 11), facecolor=BG, dpi=120)
    gs  = gridspec.GridSpec(2, 4, figure=fig,
                            hspace=0.50, wspace=0.35,
                            left=0.06, right=0.97, top=0.90, bottom=0.08)

    short = (caption[:85] + '...') if len(caption) > 85 else caption
    fig.suptitle(f'Torso-Attention Map — {tag}\n"{short}"',
                 fontsize=12, fontweight='bold',
                 fontfamily='DejaVu Serif', color='#1a1a1a', y=0.97)

    # ── Row 0: per-limb attention over torso frames ───────────────────────────
    for pi, part in enumerate(parts):
        c  = PART_COLORS[part]
        ax = fig.add_subplot(gs[0, pi])
        w  = att_weights[part] * 100
        ax.fill_between(frames, w, alpha=0.22, color=c)
        ax.plot(frames, w, color=c, lw=1.8)
        pk = int(np.argmax(att_weights[part]))
        ax.axvline(pk, color=c, lw=1.0, ls='--', alpha=0.6)
        ax.scatter([pk], [w[pk]], s=50, color=c, zorder=5,
                   edgecolors='white', lw=1)
        ax.set_title(part, fontsize=9.5, fontweight='bold',
                     color=c, fontfamily='DejaVu Serif')
        ax.set_xlabel('Torso Frame', fontsize=7.5, color='#555')
        ax.set_ylabel('Attention (%)', fontsize=7.5, color='#555')
        ax.set_xlim(0, T-1); ax.set_ylim(bottom=0)
        ax.tick_params(labelsize=7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_facecolor('#ffffff')
        ax.grid(axis='y', alpha=0.2, lw=0.4)

    # ── Row 1 LEFT (span 3): TORSO IMPORTANCE — key panel for your paper ──────
    ax_t = fig.add_subplot(gs[1, :3])
    ax_t.fill_between(frames, torso_importance*100, alpha=0.30, color='#2c3e50')
    ax_t.plot(frames, torso_importance*100, color='#2c3e50', lw=2.3,
              label='Mean torso attention (all limbs)')

    # Mark keyframes — SAME frames as your SMPL/skeleton ghost figure
    for ki, kf in enumerate(keyframes):
        is_end = (ki == 0 or ki == len(keyframes)-1)
        color  = '#c0392b' if is_end else '#e67e22'
        ax_t.axvline(kf, color=color, lw=1.3,
                     ls='-' if is_end else '--', alpha=0.85)
        ax_t.scatter([kf], [torso_importance[kf]*100], s=90,
                     color=color, zorder=6, marker='D',
                     edgecolors='white', lw=1.2)
        ax_t.text(kf, torso_importance[kf]*100*1.06, f't={kf}',
                  fontsize=7.5, color=color, ha='center',
                  fontfamily='monospace', fontweight='bold')

    ax_t.set_title('Torso Frame Importance — keyframes match SMPL/skeleton ghost figure',
                   fontsize=10.5, fontweight='bold',
                   fontfamily='DejaVu Serif', color='#1a1a1a')
    ax_t.set_xlabel('Torso Frame', fontsize=9, color='#444')
    ax_t.set_ylabel('Mean Attention (%)', fontsize=9, color='#444')
    ax_t.set_xlim(0, T-1); ax_t.set_ylim(bottom=0)
    ax_t.spines['top'].set_visible(False)
    ax_t.spines['right'].set_visible(False)
    ax_t.set_facecolor('#ffffff')
    ax_t.grid(axis='y', alpha=0.25, lw=0.5)
    ax_t.tick_params(labelsize=8)
    ax_t.legend(fontsize=8, loc='upper right', framealpha=0.9)

    # ── Row 1 RIGHT: per-limb peak frame vs the shared keyframes ──────────────
    ax_k = fig.add_subplot(gs[1, 3])
    peak_frames = [int(np.argmax(att_weights[p])) for p in parts]
    colors_list = [PART_COLORS[p] for p in parts]
    ax_k.barh(parts, peak_frames, color=colors_list, alpha=0.85)
    for kf in keyframes:
        ax_k.axvline(kf, color='#888', lw=0.8, ls=':', alpha=0.6)
    for i, pf in enumerate(peak_frames):
        ax_k.text(pf+1, i, f't={pf}', va='center', fontsize=7.5, color='#333')
    for tick, part in zip(ax_k.get_yticklabels(), parts):
        tick.set_color(PART_COLORS[part]); tick.set_fontweight('bold')
    ax_k.set_xlabel('Peak Attention Frame', fontsize=8, color='#444')
    ax_k.set_title('Peak Frame\nper Limb', fontsize=9, fontweight='bold',
                   fontfamily='DejaVu Serif')
    ax_k.spines['top'].set_visible(False)
    ax_k.spines['right'].set_visible(False)
    ax_k.set_facecolor('#ffffff')
    ax_k.tick_params(labelsize=7.5)

    fig.add_artist(FancyBboxPatch(
        (0.005,0.005), 0.990,0.990, boxstyle='square,pad=0',
        linewidth=3, edgecolor='#4CAF50', facecolor='none',
        transform=fig.transFigure, clip_on=False))

    out = os.path.join(out_dir, f'attention_{tag}.png')
    fig.savefig(out, dpi=120, facecolor=BG, format='png')
    plt.close(fig)
    print(f"  Saved: {out}")
    return out, keyframes


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — batch over multiple query_ids
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg',  type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--set',  type=str, default='test')
    parser.add_argument('--query_ids', type=int, nargs='+', required=True,
                        help='List of query ids, e.g. --query_ids 22 45 44 51')
    parser.add_argument('--out_dir', type=str, default='outputs/attention_maps')
    parser.add_argument('--n_keyframes', type=int, default=6)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}\nBATCH TORSO-ATTENTION VISUALIZATION\n{'='*60}")
    print(f"Device: {device}")
    print(f"Queries: {args.query_ids}")

    # Load model ONCE — reused across all queries (much faster)
    print("\n[Loading model once for all queries...]")
    model, cfg = load_model(args.cfg, args.ckpt, device)
    part_modules = patch_attention_fuse(model)

    all_keyframes = {}
    for qid in args.query_ids:
        print(f"\n{'-'*60}\nQuery {qid}\n{'-'*60}")
        motion, T, caption = motion_from_dataset(cfg, args.set, qid, device)
        att = extract_attention(model, motion, part_modules, device)
        tag = f'q{qid}'
        _, keyframes = plot_query_attention(
            att, caption, T, args.out_dir, tag, n_keyframes=args.n_keyframes)
        all_keyframes[qid] = {'caption': caption, 'keyframes': keyframes}

    # Save keyframe manifest — use these SAME frame numbers when rendering
    # your SMPL/skeleton ghost figures for perfect cross-referencing
    import json
    manifest_path = os.path.join(args.out_dir, 'keyframe_manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(all_keyframes, f, indent=2)
    print(f"\n{'='*60}")
    print(f"DONE — {len(args.query_ids)} queries processed")
    print(f"Keyframe manifest saved: {manifest_path}")
    print("Use these EXACT frame numbers as n_keys reference points")
    print("when generating your SMPL/skeleton ghost figures for this query,")
    print("so the attention figure and qualitative figure agree.")
    print(f"{'='*60}")
