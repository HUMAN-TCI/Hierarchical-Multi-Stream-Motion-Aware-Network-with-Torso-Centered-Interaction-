# """
# COMPLETE STEP-BY-STEP ATTENTION VISUALIZATION
# =============================================
# Drop this file in your project root alongside render_smpl.py
# Run: python step_by_step_attention.py --run "runs/<your_config>" --query_id 22
# """

# # ══════════════════════════════════════════════════════════════════════════════
# # STEP 1: IMPORTS
# # ══════════════════════════════════════════════════════════════════════════════
# import os
# import torch
# import numpy as np
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# from matplotlib.patches import FancyBboxPatch
# from matplotlib.colors import LinearSegmentedColormap
# from pathlib import Path
# from omegaconf import OmegaConf
# import hydra
# import argparse


# # ══════════════════════════════════════════════════════════════════════════════
# # STEP 2: PATCH AttentionFuse TO CAPTURE WEIGHTS
# # This monkey-patches the forward() method to save attention weights
# # WITHOUT changing any model outputs or affecting retrieval metrics
# # ══════════════════════════════════════════════════════════════════════════════
# def patch_attention_fuse(model):
#     """
#     Patches all 4 AttentionFuse modules in UpperLowerGRU.
#     After any forward pass, weights are stored in module.last_att
#     Shape: [B, T]  (batch x time)
#     """
#     part_modules = {
#         'Right Arm': model.motion_encoder.att_rarm,
#         'Left Arm':  model.motion_encoder.att_larm,
#         'Right Leg': model.motion_encoder.att_rleg,
#         'Left Leg':  model.motion_encoder.att_lleg,
#     }

#     for name, mod in part_modules.items():
#         def make_patched_forward(m, part_name):
#             def patched_forward(part_feat, torso_feat):
#                 # Original AttentionFuse logic — unchanged
#                 Q   = m.query(part_feat)                              # [B,T,h]
#                 K   = m.key(torso_feat)                               # [B,T,h]
#                 V   = m.value(torso_feat)                             # [B,T,h]
#                 att = torch.softmax(
#                     (Q * K).sum(-1, keepdim=True), dim=1)             # [B,T,1]
#                 fused = att * V + part_feat                           # residual

#                 # ← SAVE weights here (detached from graph)
#                 m.last_att  = att.squeeze(-1).detach().cpu().numpy()  # [B,T]
#                 m.part_name = part_name
#                 return m.out(fused)
#             return patched_forward

#         mod.forward = make_patched_forward(mod, name)
#         print(f"  ✓ Patched AttentionFuse for: {name}")

#     return part_modules


# # ══════════════════════════════════════════════════════════════════════════════
# # STEP 3: LOAD MODEL + RUN ONE FORWARD PASS
# # ══════════════════════════════════════════════════════════════════════════════
# def load_model_and_extract_attention(run_path, set_name='test',
#                                       query_id=22, device=None):
#     """
#     Loads the trained model, patches it, runs a forward pass,
#     and returns the attention weights for each body part.

#     Returns:
#         att_weights: dict {part_name: np.array shape (T,)}
#         caption:     str  — the query text
#         T:           int  — sequence length
#     """
#     if device is None:
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"\nStep 3a: Loading model from {run_path}")

#     # Load Hydra config
#     run_path = Path(run_path)
#     hydra_cfg = OmegaConf.load(run_path / '.hydra' / 'hydra.yaml')['hydra']
#     OmegaConf.register_new_resolver(
#         "hydra", lambda x: OmegaConf.select(hydra_cfg, x))
#     cfg = OmegaConf.load(run_path / '.hydra' / 'config.yaml')

#     # Build dataloader
#     dataset_cfg = getattr(cfg.data, set_name)
#     dataloader  = hydra.utils.call(dataset_cfg, batch_size=64)

#     # Build + load model
#     from models.model import MatchingModel
#     model = MatchingModel(cfg).to(device)
#     model.eval()

#     ckpt_path = run_path / 'best_models' / 'best_model_metric_all.pth'
#     ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
#     model.load_state_dict(
#         {k: v for k, v in ckpt['model'].items()
#          if k in model.state_dict() and
#          v.shape == model.state_dict()[k].shape})
#     print("  ✓ Model loaded")

#     # ── STEP 3b: PATCH ────────────────────────────────────────────────────────
#     print("\nStep 3b: Patching AttentionFuse modules")
#     part_modules = patch_attention_fuse(model)

#     # ── STEP 3c: FIND THE QUERY ───────────────────────────────────────────────
#     print(f"\nStep 3c: Finding query {query_id}")
#     all_desc = [dataloader.dataset[i]['desc']
#                 for i in range(len(dataloader.dataset))]
#     _, q_idx = np.unique(np.asarray(all_desc), return_index=True)
#     useful   = np.sort(q_idx)
#     real_idx = useful[query_id]
#     caption  = all_desc[real_idx]
#     print(f"  Query text: {caption[:80]}...")

#     # Get the motion for this query
#     sample  = dataloader.dataset[real_idx]
#     motion  = torch.tensor(sample['motion']).unsqueeze(0).float().to(device)
#     lengths = torch.tensor([motion.shape[1]])

#     # ── STEP 3d: FORWARD PASS (attention weights are captured here) ───────────
#     print("\nStep 3d: Running forward pass to capture attention")
#     with torch.no_grad():
#         _ = model.motion_encoder(motion, lengths)

#     # ── STEP 3e: COLLECT WEIGHTS ──────────────────────────────────────────────
#     att_weights = {}
#     for name, mod in part_modules.items():
#         if hasattr(mod, 'last_att'):
#             # last_att shape: [1, T] → take batch index 0
#             att_weights[name] = mod.last_att[0]           # shape (T,)
#             # Normalize to sum=1 (softmax already does this but ensure it)
#             att_weights[name] /= att_weights[name].sum()
#         else:
#             print(f"  WARNING: No attention captured for {name}")

#     T = motion.shape[1]
#     print(f"  ✓ Captured attention for {len(att_weights)} parts, T={T} frames")
#     return att_weights, caption, T


# # ══════════════════════════════════════════════════════════════════════════════
# # STEP 4: PLOT EVERYTHING
# # ══════════════════════════════════════════════════════════════════════════════
# PART_COLORS = {
#     'Right Arm': '#c0392b',
#     'Left Arm':  '#e74c3c',
#     'Right Leg': '#27ae60',
#     'Left Leg':  '#2ecc71',
# }
# BG = '#fafafa'

# def plot_all(att_weights, caption, T, out_dir, query_id):
#     os.makedirs(out_dir, exist_ok=True)
#     parts  = list(att_weights.keys())
#     matrix = np.stack([att_weights[p] for p in parts], axis=0)  # (4,T)
#     frames = np.arange(T)

#     cmap_heat = LinearSegmentedColormap.from_list(
#         'attn', ['#ffffff','#fef9e7','#f9ca74',
#                  '#e67e22','#c0392b','#7b241c'], N=256)

#     fig = plt.figure(figsize=(18, 14), facecolor=BG, dpi=120)
#     gs  = gridspec.GridSpec(3, 4, figure=fig,
#                             hspace=0.55, wspace=0.38,
#                             left=0.07, right=0.97,
#                             top=0.90, bottom=0.07)

#     # Short caption for title
#     short_cap = caption[:90]+'...' if len(caption)>90 else caption
#     fig.suptitle(
#         f'Attention Map  —  Query {query_id}\n"{short_cap}"',
#         fontsize=12, fontweight='bold',
#         fontfamily='DejaVu Serif', color='#1a1a1a', y=0.97)

#     # ── ROW 0: individual curves ───────────────────────────────────────────────
#     for pi,(part,color) in enumerate(zip(parts,
#                                 [PART_COLORS[p] for p in parts])):
#         ax = fig.add_subplot(gs[0, pi])
#         w  = att_weights[part] * 100
#         ax.fill_between(frames, w, alpha=0.25, color=color)
#         ax.plot(frames, w, color=color, lw=2.2, zorder=4)

#         peak = int(np.argmax(att_weights[part]))
#         ax.axvline(peak, color=color, lw=1.2, ls='--', alpha=0.6)
#         ax.scatter([peak],[w[peak]],s=70,color=color,
#                    zorder=5,edgecolors='white',lw=1)
#         ax.text(peak+0.5, w[peak]*1.03, f't={peak}',
#                 fontsize=7, color=color, fontfamily='monospace')

#         entropy = -np.sum(att_weights[part] *
#                           np.log(att_weights[part]+1e-12))
#         ax.text(0.97,0.94,f'H={entropy:.2f}',
#                 transform=ax.transAxes, fontsize=7.5,
#                 ha='right',va='top',color='#555',
#                 bbox=dict(boxstyle='round,pad=0.25',
#                           fc='white',ec='#ccc',lw=0.8))

#         ax.set_title(part, fontsize=10, fontweight='bold',
#                      color=color, fontfamily='DejaVu Serif', pad=4)
#         ax.set_xlabel('Torso Frame', fontsize=8, color='#555')
#         ax.set_ylabel('Attention (%)', fontsize=8, color='#555')
#         ax.set_xlim(0,T-1); ax.set_ylim(bottom=0)
#         ax.tick_params(labelsize=7.5)
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)
#         ax.set_facecolor('#ffffff')
#         ax.grid(axis='y',alpha=0.25,lw=0.5)

#     # ── ROW 1 LEFT: Main heatmap ───────────────────────────────────────────────
#     ax_heat = fig.add_subplot(gs[1, :3])
#     im = ax_heat.imshow(matrix*100, aspect='auto',
#                         cmap=cmap_heat, interpolation='bilinear',
#                         vmin=0, vmax=matrix.max()*110)
#     for ri,part in enumerate(parts):
#         peak = int(np.argmax(matrix[ri]))
#         ax_heat.scatter([peak],[ri],marker='D',s=90,
#                         color='#1a237e',zorder=6,lw=0)
#         ax_heat.text(peak+0.7,ri-0.3,f'{peak}',
#                      fontsize=6.5,color='#1a237e',
#                      fontfamily='monospace')

#     cbar = plt.colorbar(im, ax=ax_heat, shrink=0.9, pad=0.01)
#     cbar.set_label('Attention Weight (%)', fontsize=8, color='#444')
#     cbar.ax.tick_params(labelsize=7)
#     ax_heat.set_yticks(range(4))
#     ax_heat.set_yticklabels(parts, fontsize=9.5,
#                              fontfamily='DejaVu Serif')
#     for tick,color in zip(ax_heat.get_yticklabels(),
#                           [PART_COLORS[p] for p in parts]):
#         tick.set_color(color); tick.set_fontweight('bold')
#     ax_heat.set_xlabel('Torso Frame Index', fontsize=9, color='#444')
#     ax_heat.set_title('Attention Heatmap: Body Parts → Torso Frames',
#                       fontsize=11,fontweight='bold',
#                       fontfamily='DejaVu Serif',color='#1a1a1a',pad=6)
#     ax_heat.tick_params(axis='x',labelsize=8)

#     # ── ROW 1 RIGHT: Bar chart ─────────────────────────────────────────────────
#     ax_bar = fig.add_subplot(gs[1, 3])
#     means = matrix.mean(axis=1)*100
#     stds  = matrix.std(axis=1)*100
#     bars  = ax_bar.barh(parts, means, xerr=stds,
#                         color=[PART_COLORS[p] for p in parts],
#                         alpha=0.85,
#                         error_kw=dict(ecolor='#555',capsize=3,lw=1.2))
#     for bar,mv in zip(bars,means):
#         ax_bar.text(mv+0.02, bar.get_y()+bar.get_height()/2,
#                     f'{mv:.2f}%', va='center', fontsize=7.5, color='#333')
#     ax_bar.set_xlabel('Mean Attention (%)', fontsize=8.5, color='#444')
#     ax_bar.set_title('Mean ± Std\nper Part', fontsize=9.5,
#                      fontweight='bold',fontfamily='DejaVu Serif',pad=4)
#     ax_bar.spines['top'].set_visible(False)
#     ax_bar.spines['right'].set_visible(False)
#     ax_bar.set_facecolor('#ffffff'); ax_bar.tick_params(labelsize=8.5)
#     for tick,color in zip(ax_bar.get_yticklabels(),
#                           [PART_COLORS[p] for p in parts]):
#         tick.set_color(color); tick.set_fontweight('bold')

#     # ── ROW 2 LEFT: Overlay ────────────────────────────────────────────────────
#     ax_all = fig.add_subplot(gs[2, :2])
#     for part,color in zip(parts,[PART_COLORS[p] for p in parts]):
#         w = att_weights[part]*100
#         ax_all.plot(frames,w,color=color,lw=2.2,label=part,alpha=0.9)
#         ax_all.fill_between(frames,w,alpha=0.08,color=color)
#     ax_all.set_title('All Parts Overlaid',fontsize=10.5,fontweight='bold',
#                      fontfamily='DejaVu Serif',color='#1a1a1a',pad=5)
#     ax_all.set_xlabel('Torso Frame',fontsize=9,color='#444')
#     ax_all.set_ylabel('Attention (%)',fontsize=9,color='#444')
#     ax_all.set_xlim(0,T-1); ax_all.set_ylim(bottom=0)
#     ax_all.legend(fontsize=8.5,framealpha=0.92,
#                   edgecolor='#ddd',loc='upper right')
#     ax_all.spines['top'].set_visible(False)
#     ax_all.spines['right'].set_visible(False)
#     ax_all.set_facecolor('#ffffff')
#     ax_all.grid(axis='y',alpha=0.25,lw=0.5)
#     ax_all.tick_params(labelsize=8)

#     # ── ROW 2 RIGHT: Cumulative ────────────────────────────────────────────────
#     ax_cum = fig.add_subplot(gs[2, 2:])
#     total  = matrix.sum(axis=0); total /= total.sum()
#     cum    = np.cumsum(total)
#     ax_cum.fill_between(frames,total*100,alpha=0.3,color='#8e44ad')
#     ax_cum.plot(frames,total*100,color='#8e44ad',lw=2.2,
#                 label='Total attention')
#     ax2 = ax_cum.twinx()
#     ax2.plot(frames,cum*100,color='#2c3e50',lw=1.8,
#              ls='--',label='Cumulative (%)')
#     ax2.axhline(50,color='#27ae60',lw=1,ls=':',alpha=0.7)
#     ax2.axhline(90,color='#e67e22',lw=1,ls=':',alpha=0.7)
#     ax2.text(T-1,51,'50%',fontsize=7,color='#27ae60',ha='right')
#     ax2.text(T-1,91,'90%',fontsize=7,color='#e67e22',ha='right')
#     ax2.set_ylabel('Cumulative (%)',fontsize=8,color='#2c3e50')
#     ax2.tick_params(labelsize=7.5,colors='#2c3e50')
#     ax2.set_ylim(0,105)
#     f50 = int(np.searchsorted(cum,0.50))
#     f90 = int(np.searchsorted(cum,0.90))
#     for fx,fc2,lab in [(f50,'#27ae60','F₅₀'),(f90,'#e67e22','F₉₀')]:
#         ax_cum.axvline(fx,color=fc2,lw=1.2,ls=':',alpha=0.8)
#         ax_cum.text(fx+0.5,total.max()*90,f'{lab}={fx}',
#                     fontsize=7,color=fc2,fontfamily='monospace')
#     lines1,labs1 = ax_cum.get_legend_handles_labels()
#     lines2,labs2 = ax2.get_legend_handles_labels()
#     ax_cum.legend(lines1+lines2,labs1+labs2,
#                   fontsize=8,framealpha=0.9,edgecolor='#ddd',
#                   loc='upper left')
#     ax_cum.set_title('Total + Cumulative Attention',fontsize=10.5,
#                      fontweight='bold',fontfamily='DejaVu Serif',
#                      color='#1a1a1a',pad=5)
#     ax_cum.set_xlabel('Torso Frame',fontsize=9,color='#444')
#     ax_cum.set_ylabel('Total Attention (%)',fontsize=9,color='#444')
#     ax_cum.set_xlim(0,T-1); ax_cum.set_ylim(bottom=0)
#     ax_cum.spines['top'].set_visible(False)
#     ax_cum.set_facecolor('#ffffff')
#     ax_cum.grid(axis='y',alpha=0.25,lw=0.5)
#     ax_cum.tick_params(labelsize=8)

#     # Border
#     fig.add_artist(FancyBboxPatch(
#         (0.005,0.005),0.990,0.990,
#         boxstyle='square,pad=0',linewidth=3,
#         edgecolor='#4CAF50',facecolor='none',
#         transform=fig.transFigure,clip_on=False))

#     out_path = os.path.join(out_dir, f'attention_q{query_id}.png')
#     fig.savefig(out_path, dpi=120, facecolor=BG, format='png')
#     plt.close(fig)
#     print(f"\n  ✓ Saved: {out_path}")
#     return out_path


# # ══════════════════════════════════════════════════════════════════════════════
# # STEP 5: MAIN — tie it all together
# # ══════════════════════════════════════════════════════════════════════════════
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(
#         description='Step-by-step attention visualization')
#     parser.add_argument('--run', type=str, required=True,
#         help='Path to trained run directory')
#     parser.add_argument('--set', type=str, default='test')
#     parser.add_argument('--query_id', type=int, default=22,
#         help='Which query index to visualize')
#     parser.add_argument('--out_dir', type=str,
#         default='outputs/attention_maps')
#     args = parser.parse_args()

#     print("=" * 60)
#     print("ATTENTION VISUALIZATION — STEP BY STEP")
#     print("=" * 60)

#     print("\n[STEP 1] Imports done ✓")
#     print("\n[STEP 2] AttentionFuse patch ready ✓")
#     print("\n[STEP 3] Loading model and extracting attention weights...")

#     att_weights, caption, T = load_model_and_extract_attention(
#         run_path = args.run,
#         set_name = args.set,
#         query_id = args.query_id,
#     )

#     print(f"\n[STEP 4] Plotting {len(att_weights)} attention maps (T={T})...")
#     out = plot_all(att_weights, caption, T, args.out_dir, args.query_id)

#     print("\n[STEP 5] Done!")
#     print(f"         Output: {out}")
#     print("=" * 60)


"""
step_by_step_attention.py  — V2
Accepts either --run (full path) OR --ckpt + --npz directly.
No Hydra config needed when using --npz mode.

Usage A (full run):
    python step_by_step_attention.py \
        --run "runs/+data.num_workers=0/data=human-ml3d/motion_model=upper-lower-gru/optim.batch_size=64/optim=info-nce/text_model=clip/data_rep=cont_6d_plus_rifke/space-dim=256/run-42" \
        --query_id 22

Usage B (direct npz — simplest):
    python step_by_step_attention.py \
        --npz "outputs/renders_smpl/test/22/004965_smpl_fit.npz" \
        --ckpt "runs/.../best_models/best_model_metric_all.pth" \
        --cfg  "runs/.../.hydra/config.yaml"
"""

import os, sys, argparse
from pathlib import Path

import torch
import os

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


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — color config
# ══════════════════════════════════════════════════════════════════════════════
PART_COLORS = {
    'Right Arm': '#c0392b',
    'Left Arm':  '#e74c3c',
    'Right Leg': '#27ae60',
    'Left Leg':  '#2ecc71',
}
BG = '#fafafa'


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — patch AttentionFuse to capture weights
# ══════════════════════════════════════════════════════════════════════════════
# # def patch_attention_fuse(model):
#     # """
#     Monkey-patches all 4 AttentionFuse modules.
#     After forward(), weights stored in module.last_att  shape [B,T].
#     Model outputs are UNCHANGED.
#     """
#     # Try both common import paths
#     try:
#         motion_enc = model.motion_encoder          # MatchingModel wraps it here
#     except AttributeError:
#         motion_enc = model                         # passed directly

#     part_modules = {
#         'Right Arm': motion_enc.att_rarm,
#         'Left Arm':  motion_enc.att_larm,
#         'Right Leg': motion_enc.att_rleg,
#         'Left Leg':  motion_enc.att_lleg,
#     }

#     for name, mod in part_modules.items():
#         def _make(m, n):
#             def _fwd(part_feat, torso_feat):
#                 Q   = m.query(part_feat)
#                 K   = m.key(torso_feat)
#                 V   = m.value(torso_feat)
#                 att = torch.softmax((Q*K).sum(-1, keepdim=True), dim=1)
#                 m.last_att = att.squeeze(-1).detach().cpu().numpy()  # [B,T]
#                 fused = att * V + part_feat
#                 return m.out(fused)
#             return _fwd
#         mod.forward = _make(mod, name)
#         print(f"  ✓ Patched: {name}")

#     # return part_modules

def patch_attention_fuse(model):
    """
    Capture torso-centered attention weights from UpperLowerGRU.

    MatchingModel:
        model.pose_enc -> UpperLowerGRU

    Captured:
        att_rarm
        att_larm
        att_rleg
        att_lleg

    Output:
        module.last_att -> [B,T]
    """

    # Correct encoder name
    if hasattr(model, "pose_enc"):
        motion_enc = model.pose_enc
    else:
        motion_enc = model


    part_modules = {

        "Right Arm":
            motion_enc.att_rarm,

        "Left Arm":
            motion_enc.att_larm,

        "Right Leg":
            motion_enc.att_rleg,

        "Left Leg":
            motion_enc.att_lleg
    }


    for name, module in part_modules.items():


        def patched_forward(
            part_feat,
            torso_feat,
            m=module
        ):

            # Query = limb
            Q = m.query(part_feat)

            # Key/Value = torso
            K = m.key(torso_feat)
            V = m.value(torso_feat)


            # torso interaction attention
            att = torch.softmax(
                (Q*K).sum(-1, keepdim=True),
                dim=1
            )


            # save attention
            m.last_att = (
                att
                .squeeze(-1)
                .detach()
                .cpu()
                .numpy()
            )


            # original AttentionFuse operation
            fused = att * V + part_feat


            return m.out(fused)


        module.forward = patched_forward


        print(
            f"  ✓ Patched {name}"
        )


    return part_modules
# ══════════════════════════════════════════════════════════════════════════════
# STEP 3A — load model from config yaml + checkpoint
# ══════════════════════════════════════════════════════════════════════════════
def load_model(cfg_path, ckpt_path, device):
    from omegaconf import OmegaConf
    import hydra

    print(f"  Loading config: {cfg_path}")
    cfg = OmegaConf.load(cfg_path)

    # Register hydra resolver if needed
    try:
        hydra_yaml = Path(cfg_path).parent / 'hydra.yaml'
        if hydra_yaml.exists():
            hydra_cfg = OmegaConf.load(hydra_yaml)['hydra']
            OmegaConf.register_new_resolver(
                "hydra", lambda x: OmegaConf.select(hydra_cfg, x))
    except Exception:
        pass

    from models.model import MatchingModel
    model = MatchingModel(cfg).to(device)
    model.eval()

    print(f"  Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ckpt.get('model', ckpt)
    model_state = model.state_dict()
    compatible = {k: v for k, v in state.items()
                  if k in model_state and v.shape == model_state[k].shape}
    model_state.update(compatible)
    model.load_state_dict(model_state)
    print(f"  ✓ Loaded {len(compatible)}/{len(model_state)} parameters")
    return model, cfg


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3B — get motion tensor from NPZ (SMPL fit output)
# ══════════════════════════════════════════════════════════════════════════════
# def motion_from_npz(npz_path, device):
    """
    Load joints from a saved _smpl_fit.npz file.
    Returns tensor [1, T, 22, 3] and T.
    """
    data   = np.load(npz_path)
    joints = data['joints']              # (T, 45, 3) or (T, 24, 3)
    joints = joints[:, :22, :]          # keep first 22 joints
    T      = joints.shape[0]
    print(f"  Loaded joints from npz: shape={joints.shape}")
    tensor = torch.tensor(joints).float().unsqueeze(0).to(device)  # [1,T,22,3]
    return tensor, T

def motion_from_npz(npz_path, device):

    """
    Load SMPL joints.

    Converts:
        XYZ joints

    into:
        6D rotation representation
    """

    from data_loaders.humanml.scripts.motion_process import recover_rot


    data=np.load(npz_path)


    joints=data['joints']


    joints=joints[:,:22,:]


    T=joints.shape[0]


    print(
        "Original joints:",
        joints.shape
    )


    motion=torch.tensor(
        joints
    ).float()



    # --------------------------------------------------
    # IMPORTANT
    # Your model was trained on HumanML3D cont_6d.
    # If your NPZ already contains motion_6d use it.
    # --------------------------------------------------

    if 'motion' in data:

        motion=torch.tensor(
            data['motion']
        ).float()

        print(
            "Using stored motion representation:",
            motion.shape
        )


    else:

        raise ValueError(
            """
NPZ only contains xyz joints.

Your UpperLowerGRU expects cont_6d or cont_6d_plus_rifke.

Please save the HumanML3D processed motion representation
inside the NPZ or use --set dataset mode.
"""
        )


    motion=motion.unsqueeze(0).to(device)


    return motion,T
# ══════════════════════════════════════════════════════════════════════════════
# STEP 3C — get motion tensor from dataset (by query_id)
# ══════════════════════════════════════════════════════════════════════════════
# def motion_from_dataset(cfg, set_name, query_id, device):
    
#     import hydra
#     dataset_cfg = getattr(cfg.data, set_name)
#     dataloader  = hydra.utils.call(dataset_cfg, batch_size=64)
#     dataset     = dataloader.dataset

#     all_desc = [dataset[i]['desc'] for i in range(len(dataset))]
#     _, q_idx = np.unique(np.asarray(all_desc), return_index=True)
#     useful   = np.sort(q_idx)
#     real_idx = int(useful[query_id])
#     caption  = all_desc[real_idx]
#     print(f"  Query {query_id} → dataset index {real_idx}")
#     print(f"  Caption: {caption[:100]}")

#     sample  = dataset[real_idx]
#     motion  = torch.tensor(sample['motion']).float().unsqueeze(0).to(device)
#     T       = motion.shape[1]
#     return motion, T, caption


# # ══════════════════════════════════════════════════════════════════════════════
# # STEP 4 — run forward pass and extract attention
# # ══════════════════════════════════════════════════════════════════════════════
# # def extract_attention(model, motion_tensor, part_modules, device):
#     """
#     Runs one forward pass through motion_encoder only.
#     Returns dict {part_name: np.array(T,)}
#     """
#     T       = motion_tensor.shape[1]
#     lengths = torch.tensor([T])

#     with torch.no_grad():
#         try:
#             _ = model.motion_encoder(motion_tensor, lengths)
#         except Exception:
#             # Some configs expose forward differently
#             _ = model(motion_tensor, lengths)

#     att_weights = {}
#     for name, mod in part_modules.items():
#         if hasattr(mod, 'last_att'):
#             w = mod.last_att[0]          # shape (T,)
#             w = np.clip(w, 0, None)
#             s = w.sum()
#             att_weights[name] = w / s if s > 0 else w
#         else:
#             print(f"  WARNING: No attention for {name}")

#     print(f"  ✓ Captured attention: {list(att_weights.keys())}")
#     return att_weights
def motion_from_dataset(cfg, set_name, query_id, device):

    import hydra

    dataset_cfg = getattr(cfg.data, set_name)

    dataloader = hydra.utils.call(
        dataset_cfg,
        batch_size=64
    )

    dataset = dataloader.dataset


    all_desc = [
        dataset[i]['desc']
        for i in range(len(dataset))
    ]


    _, q_idx = np.unique(
        np.asarray(all_desc),
        return_index=True
    )

    useful = np.sort(q_idx)

    real_idx = int(useful[query_id])


    caption = all_desc[real_idx]


    print(
        f"Query {query_id} → dataset index {real_idx}"
    )

    print(
        f"Caption: {caption[:100]}"
    )


    sample = dataset[real_idx]


    motion = torch.tensor(
        sample['motion']
    ).float()


    print(
        "Original motion shape:",
        motion.shape
    )


    # -------------------------------------
    # Convert flattened representation
    # -------------------------------------

    if motion.ndim == 2:

        T, D = motion.shape

        if D == 251:

            # KIT cont_6d_plus_rifke
            # first 189 dims are joint rotations
            # remaining 62 dims are root/RIFKE

            motion = motion[:, :189]

            motion = motion.reshape(
                T,
                21,
                9
            )


        elif D == 263:

            # HumanML3D
            motion = motion[:, :198]

            motion = motion.reshape(
                T,
                22,
                9
            )


        else:
            raise ValueError(
                f"Unexpected motion dimension {D}"
            )

        

        print(
            "Encoder motion shape:",
            motion.shape
        )


        motion = (
            motion
            .unsqueeze(0)
            .to(device)
        )


        T = motion.shape[1]


    return motion, T, caption

def extract_attention(
        model,
        motion_tensor,
        part_modules,
        device
):

    """
    Run only motion encoder.

    Extract:
        Right Arm
        Left Arm
        Right Leg
        Left Leg
    """

    T = motion_tensor.shape[1]


    lengths = torch.tensor(
        [T],
        dtype=torch.long,
        device=device
    )


    with torch.no_grad():


        # MatchingModel
        if hasattr(model, "pose_enc"):


            _ = model.pose_enc(
                motion_tensor,
                lengths
            )


        else:

            _ = model(
                motion_tensor,
                lengths
            )



    attention = {}


    for name, module in part_modules.items():


        if hasattr(module, "last_att"):


            w = module.last_att[0]


            # avoid numerical problems
            w = np.maximum(w,0)


            if w.sum()>0:
                w = w / w.sum()


            attention[name]=w


        else:


            print(
                f"WARNING: missing {name}"
            )



    print(
        "✓ Captured attention:",
        list(attention.keys())
    )


    return attention

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — plot everything
# ══════════════════════════════════════════════════════════════════════════════
# def plot_attention(att_weights, caption, T, out_dir, tag='query'):
#     os.makedirs(out_dir, exist_ok=True)
#     parts  = list(att_weights.keys())
#     matrix = np.stack([att_weights[p] for p in parts], axis=0)   # (4,T)
#     frames = np.arange(T)

#     cmap_heat = LinearSegmentedColormap.from_list(
#         'attn', ['#ffffff','#fef9e7','#f9ca74',
#                  '#e67e22','#c0392b','#7b241c'], N=256)

#     fig = plt.figure(figsize=(18, 14), facecolor=BG, dpi=120)
#     gs  = gridspec.GridSpec(3, 4, figure=fig,
#                             hspace=0.55, wspace=0.38,
#                             left=0.07, right=0.97,
#                             top=0.90, bottom=0.07)

#     short = (caption[:85]+'...') if len(caption)>85 else caption
#     fig.suptitle(f'Attention Map — {tag}\n"{short}"',
#                  fontsize=12, fontweight='bold',
#                  fontfamily='DejaVu Serif', color='#1a1a1a', y=0.97)

#     # ── ROW 0: per-part curves ─────────────────────────────────────────────────
#     for pi, part in enumerate(parts):
#         color = PART_COLORS[part]
#         ax    = fig.add_subplot(gs[0, pi])
#         w     = att_weights[part] * 100

#         ax.fill_between(frames, w, alpha=0.22, color=color)
#         ax.plot(frames, w, color=color, lw=2.2, zorder=4)

#         peak = int(np.argmax(att_weights[part]))
#         ax.axvline(peak, color=color, lw=1.2, ls='--', alpha=0.6)
#         ax.scatter([peak], [w[peak]], s=70, color=color,
#                    zorder=5, edgecolors='white', lw=1.2)
#         ax.text(peak+0.5, w[peak]*1.03, f't={peak}',
#                 fontsize=7, color=color, fontfamily='monospace')

#         H = -np.sum(att_weights[part] * np.log(att_weights[part]+1e-12))
#         ax.text(0.97, 0.94, f'H={H:.2f}', transform=ax.transAxes,
#                 fontsize=7.5, ha='right', va='top', color='#555',
#                 bbox=dict(boxstyle='round,pad=0.25',
#                           fc='white', ec='#ccc', lw=0.8))

#         ax.set_title(part, fontsize=10, fontweight='bold',
#                      color=color, fontfamily='DejaVu Serif', pad=4)
#         ax.set_xlabel('Torso Frame', fontsize=8, color='#555')
#         ax.set_ylabel('Attention (%)', fontsize=8, color='#555')
#         ax.set_xlim(0, T-1); ax.set_ylim(bottom=0)
#         ax.tick_params(labelsize=7.5)
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)
#         ax.set_facecolor('#ffffff')
#         ax.grid(axis='y', alpha=0.25, lw=0.5)

#     # ── ROW 1 LEFT: heatmap ────────────────────────────────────────────────────
#     ax_h = fig.add_subplot(gs[1, :3])
#     im   = ax_h.imshow(matrix*100, aspect='auto', cmap=cmap_heat,
#                        interpolation='bilinear',
#                        vmin=0, vmax=matrix.max()*110)
#     for ri, part in enumerate(parts):
#         pk = int(np.argmax(matrix[ri]))
#         ax_h.scatter([pk], [ri], marker='D', s=90,
#                      color='#1a237e', zorder=6, lw=0)
#         ax_h.text(pk+0.7, ri-0.28, str(pk), fontsize=6.5,
#                   color='#1a237e', fontfamily='monospace')

#     plt.colorbar(im, ax=ax_h, shrink=0.9,
#                  pad=0.01).set_label('Attention (%)', fontsize=8)
#     ax_h.set_yticks(range(4))
#     ax_h.set_yticklabels(parts, fontsize=9.5, fontfamily='DejaVu Serif')
#     for tick, part in zip(ax_h.get_yticklabels(), parts):
#         tick.set_color(PART_COLORS[part]); tick.set_fontweight('bold')
#     ax_h.set_xlabel('Torso Frame', fontsize=9, color='#444')
#     ax_h.set_title('Heatmap: Body Part Attention over Torso Frames',
#                    fontsize=11, fontweight='bold',
#                    fontfamily='DejaVu Serif', pad=6)
#     ax_h.tick_params(labelsize=8)

#     # ── ROW 1 RIGHT: bar ───────────────────────────────────────────────────────
#     ax_b = fig.add_subplot(gs[1, 3])
#     means = matrix.mean(axis=1)*100
#     stds  = matrix.std(axis=1)*100
#     bars  = ax_b.barh(parts, means, xerr=stds,
#                       color=[PART_COLORS[p] for p in parts], alpha=0.85,
#                       error_kw=dict(ecolor='#555', capsize=3, lw=1.2))
#     for bar, mv in zip(bars, means):
#         ax_b.text(mv+0.02, bar.get_y()+bar.get_height()/2,
#                   f'{mv:.2f}%', va='center', fontsize=7.5)
#     for tick, part in zip(ax_b.get_yticklabels(), parts):
#         tick.set_color(PART_COLORS[part]); tick.set_fontweight('bold')
#     ax_b.set_xlabel('Mean Attention (%)', fontsize=8.5, color='#444')
#     ax_b.set_title('Mean ± Std\nper Part', fontsize=9.5,
#                    fontweight='bold', fontfamily='DejaVu Serif', pad=4)
#     ax_b.spines['top'].set_visible(False)
#     ax_b.spines['right'].set_visible(False)
#     ax_b.set_facecolor('#ffffff'); ax_b.tick_params(labelsize=8.5)

#     # ── ROW 2 LEFT: overlay ────────────────────────────────────────────────────
#     ax_a = fig.add_subplot(gs[2, :2])
#     for part in parts:
#         w = att_weights[part]*100
#         ax_a.plot(frames, w, color=PART_COLORS[part],
#                   lw=2.2, label=part, alpha=0.9)
#         ax_a.fill_between(frames, w, alpha=0.08, color=PART_COLORS[part])
#     ax_a.set_title('All Parts Overlaid', fontsize=10.5, fontweight='bold',
#                    fontfamily='DejaVu Serif', pad=5)
#     ax_a.set_xlabel('Torso Frame', fontsize=9, color='#444')
#     ax_a.set_ylabel('Attention (%)', fontsize=9, color='#444')
#     ax_a.set_xlim(0, T-1); ax_a.set_ylim(bottom=0)
#     ax_a.legend(fontsize=8.5, framealpha=0.92, edgecolor='#ddd',
#                 loc='upper right')
#     ax_a.spines['top'].set_visible(False)
#     ax_a.spines['right'].set_visible(False)
#     ax_a.set_facecolor('#ffffff')
#     ax_a.grid(axis='y', alpha=0.25, lw=0.5)
#     ax_a.tick_params(labelsize=8)

#     # ── ROW 2 RIGHT: cumulative ────────────────────────────────────────────────
#     ax_c  = fig.add_subplot(gs[2, 2:])
#     total = matrix.sum(axis=0); total /= total.sum()
#     cum   = np.cumsum(total)
#     ax_c.fill_between(frames, total*100, alpha=0.28, color='#8e44ad')
#     ax_c.plot(frames, total*100, color='#8e44ad', lw=2.2,
#               label='Total attention')
#     ax_c2 = ax_c.twinx()
#     ax_c2.plot(frames, cum*100, color='#2c3e50', lw=1.8,
#                ls='--', label='Cumulative (%)')
#     for pct, col, lab in [(50,'#27ae60','50%'),(90,'#e67e22','90%')]:
#         ax_c2.axhline(pct, color=col, lw=1, ls=':', alpha=0.7)
#         ax_c2.text(T-1, pct+1, lab, fontsize=7, color=col, ha='right')
#         fi = int(np.searchsorted(cum, pct/100))
#         ax_c.axvline(fi, color=col, lw=1.2, ls=':', alpha=0.8)
#         ax_c.text(fi+0.5, total.max()*85,
#                   f'F={fi}', fontsize=7, color=col,
#                   fontfamily='monospace')
#     ax_c2.set_ylabel('Cumulative (%)', fontsize=8, color='#2c3e50')
#     ax_c2.tick_params(labelsize=7.5, colors='#2c3e50')
#     ax_c2.set_ylim(0, 105)
#     l1,b1 = ax_c.get_legend_handles_labels()
#     l2,b2 = ax_c2.get_legend_handles_labels()
#     ax_c.legend(l1+l2, b1+b2, fontsize=8, framealpha=0.9,
#                 edgecolor='#ddd', loc='upper left')
#     ax_c.set_title('Total + Cumulative Attention', fontsize=10.5,
#                    fontweight='bold', fontfamily='DejaVu Serif', pad=5)
#     ax_c.set_xlabel('Torso Frame', fontsize=9, color='#444')
#     ax_c.set_ylabel('Total Attention (%)', fontsize=9, color='#444')
#     ax_c.set_xlim(0, T-1); ax_c.set_ylim(bottom=0)
#     ax_c.spines['top'].set_visible(False)
#     ax_c.set_facecolor('#ffffff')
#     ax_c.grid(axis='y', alpha=0.25, lw=0.5)
#     ax_c.tick_params(labelsize=8)

#     # Border
#     fig.add_artist(FancyBboxPatch(
#         (0.005,0.005), 0.990,0.990,
#         boxstyle='square,pad=0', linewidth=3,
#         edgecolor='#4CAF50', facecolor='none',
#         transform=fig.transFigure, clip_on=False))

#     out = os.path.join(out_dir, f'attention_{tag}.png')
#     fig.savefig(out, dpi=120, facecolor=BG, format='png')
#     plt.close(fig)
#     print(f"\n  ✓ Saved: {out}")
#     return out

# """
# Drop-in replacement for plot_attention() in step_by_step_attention.py

# HOW TO USE:
#   1. Open step_by_step_attention.py
#   2. Find the line:  def plot_attention(att_weights, caption, T, out_dir, tag):
#   3. Delete that function entirely (down to the next def or if __name__)
#   4. Paste everything from this file in its place
# """

# # def plot_attention(att_weights, caption, T, out_dir, tag):
# #     import os
# #     import numpy as np
# #     import matplotlib
# #     matplotlib.use("Agg")
# #     import matplotlib.pyplot as plt
# #     import matplotlib.gridspec as gridspec
# #     from matplotlib.patches import FancyBboxPatch
# #     from matplotlib.colors import LinearSegmentedColormap

# #     os.makedirs(out_dir, exist_ok=True)
# #     parts  = list(att_weights.keys())
# #     matrix = np.stack([att_weights[p] for p in parts], axis=0)   # (4, T)

# #     # ── AUTO-DETECT real sequence length ──────────────────────────────────────
# #     # Softmax over padding produces a perfectly flat plateau — detect and cut it
# #     total_att = matrix.mean(axis=0)
# #     window    = max(5, T // 20)
# #     T_real    = T
# #     for i in range(T - window):
# #         if total_att[i:i+window].std() < 1e-4:
# #             T_real = i
# #             break
# #     print(f"  Real sequence length detected: {T_real} / {T} frames")

# #     frames_r = np.arange(T_real)
# #     matrix_r = matrix[:, :T_real].copy()

# #     # Re-normalise over real frames only
# #     for i in range(matrix_r.shape[0]):
# #         s = matrix_r[i].sum()
# #         if s > 0:
# #             matrix_r[i] /= s

# #     att_r = {p: matrix_r[pi] for pi, p in enumerate(parts)}

# #     PART_COLORS = {
# #         "Right Arm": "#c0392b",
# #         "Left Arm":  "#e74c3c",
# #         "Right Leg": "#27ae60",
# #         "Left Leg":  "#2ecc71",
# #     }
# #     BG = "#fafafa"
# #     cmap_heat = LinearSegmentedColormap.from_list(
# #         "a", ["#ffffff","#fef9e7","#f9ca74",
# #               "#e67e22","#c0392b","#7b241c"])

# #     fig = plt.figure(figsize=(18, 14), facecolor=BG, dpi=120)
# #     gs  = gridspec.GridSpec(3, 4, figure=fig,
# #                             hspace=0.55, wspace=0.38,
# #                             left=0.07, right=0.97,
# #                             top=0.90, bottom=0.07)

# #     short = (caption[:85] + "...") if len(caption) > 85 else caption
# #     fig.suptitle(f'Attention Map — {tag}\n"{short}"',
# #                  fontsize=12, fontweight="bold",
# #                  fontfamily="DejaVu Serif", color="#1a1a1a", y=0.97)

# #     # ── ROW 0: per-part curves ────────────────────────────────────────────────
# #     for pi, part in enumerate(parts):
# #         c  = PART_COLORS[part]
# #         ax = fig.add_subplot(gs[0, pi])
# #         w  = att_r[part] * 100

# #         ax.fill_between(frames_r, w, alpha=0.22, color=c)
# #         ax.plot(frames_r, w, color=c, lw=2.0)

# #         pk = int(np.argmax(att_r[part]))
# #         ax.axvline(pk, color=c, lw=1.2, ls="--", alpha=0.6)
# #         ax.scatter([pk], [w[pk]], s=70, color=c,
# #                    zorder=5, edgecolors="white", lw=1.2)
# #         ax.text(pk + 0.5, w[pk] * 1.03, f"t={pk}",
# #                 fontsize=7, color=c, fontfamily="monospace")

# #         H = -np.sum(att_r[part] * np.log(att_r[part] + 1e-12))
# #         ax.text(0.97, 0.94, f"H={H:.2f}", transform=ax.transAxes,
# #                 fontsize=7.5, ha="right", va="top", color="#555",
# #                 bbox=dict(boxstyle="round,pad=0.25",
# #                           fc="white", ec="#ccc", lw=0.8))

# #         ax.set_title(part, fontsize=10, fontweight="bold",
# #                      color=c, fontfamily="DejaVu Serif", pad=4)
# #         ax.set_xlabel("Frame", fontsize=8, color="#555")
# #         ax.set_ylabel("Attention (%)", fontsize=8, color="#555")
# #         ax.set_xlim(0, T_real - 1); ax.set_ylim(bottom=0)
# #         ax.tick_params(labelsize=7.5)
# #         ax.spines["top"].set_visible(False)
# #         ax.spines["right"].set_visible(False)
# #         ax.set_facecolor("#ffffff")
# #         ax.grid(axis="y", alpha=0.25, lw=0.5)

# #     # ── ROW 1 LEFT: per-row normalised heatmap ────────────────────────────────
# #     ax_h = fig.add_subplot(gs[1, :3])
# #     disp = matrix_r.copy()
# #     for i in range(disp.shape[0]):
# #         mn, mx = disp[i].min(), disp[i].max()
# #         if mx > mn:
# #             disp[i] = (disp[i] - mn) / (mx - mn)

# #     im = ax_h.imshow(disp, aspect="auto", cmap=cmap_heat,
# #                      interpolation="bilinear", vmin=0, vmax=1)

# #     for ri, part in enumerate(parts):
# #         pk = int(np.argmax(matrix_r[ri]))
# #         ax_h.scatter([pk], [ri], marker="D", s=90,
# #                      color="#1a237e", zorder=6, lw=0)
# #         ax_h.text(pk + 0.7, ri - 0.28, str(pk),
# #                   fontsize=6.5, color="#1a237e", fontfamily="monospace")

# #     cbar = plt.colorbar(im, ax=ax_h, shrink=0.9, pad=0.01)
# #     cbar.set_label("Normalised Attention (per part)", fontsize=8)
# #     ax_h.set_yticks(range(4))
# #     ax_h.set_yticklabels(parts, fontsize=9.5, fontfamily="DejaVu Serif")
# #     for tick, part in zip(ax_h.get_yticklabels(), parts):
# #         tick.set_color(PART_COLORS[part])
# #         tick.set_fontweight("bold")
# #     ax_h.set_xlabel("Frame", fontsize=9, color="#444")
# #     ax_h.set_title("Attention Heatmap: Body Parts → Torso Frames",
# #                    fontsize=11, fontweight="bold",
# #                    fontfamily="DejaVu Serif", pad=6)
# #     ax_h.tick_params(labelsize=8)

# #     # ── ROW 1 RIGHT: peak + mean bar chart ────────────────────────────────────
# #     ax_b = fig.add_subplot(gs[1, 3])
# #     peaks = [att_r[p].max() * 100 for p in parts]
# #     means = [att_r[p].mean() * 100 for p in parts]
# #     stds  = [att_r[p].std()  * 100 for p in parts]
# #     colors_list = [PART_COLORS[p] for p in parts]

# #     ax_b.barh(parts, peaks, color=colors_list, alpha=0.85, label="Peak")
# #     ax_b.barh(parts, means, color=colors_list, alpha=0.35,
# #               xerr=stds,
# #               error_kw=dict(ecolor="#555", capsize=3, lw=1.2),
# #               label="Mean±Std")

# #     for i, (pv, mv) in enumerate(zip(peaks, means)):
# #         ax_b.text(pv + 0.05, i, f"pk={pv:.1f}%",
# #                   va="center", fontsize=7, color="#333")

# #     for tick, part in zip(ax_b.get_yticklabels(), parts):
# #         tick.set_color(PART_COLORS[part])
# #         tick.set_fontweight("bold")

# #     ax_b.set_xlabel("Attention (%)", fontsize=8.5, color="#444")
# #     ax_b.set_title("Peak (solid)\nMean±Std (light)",
# #                    fontsize=9, fontweight="bold",
# #                    fontfamily="DejaVu Serif", pad=4)
# #     ax_b.legend(fontsize=7, loc="lower right")
# #     ax_b.spines["top"].set_visible(False)
# #     ax_b.spines["right"].set_visible(False)
# #     ax_b.set_facecolor("#ffffff")
# #     ax_b.tick_params(labelsize=8.5)

# #     # ── ROW 2 LEFT: overlay ────────────────────────────────────────────────────
# #     ax_a = fig.add_subplot(gs[2, :2])
# #     for part in parts:
# #         w = att_r[part] * 100
# #         ax_a.plot(frames_r, w, color=PART_COLORS[part],
# #                   lw=2.0, label=part, alpha=0.9)
# #         ax_a.fill_between(frames_r, w, alpha=0.07,
# #                           color=PART_COLORS[part])
# #     ax_a.set_title("All Parts Overlaid", fontsize=10.5,
# #                    fontweight="bold", fontfamily="DejaVu Serif", pad=5)
# #     ax_a.set_xlabel("Frame", fontsize=9, color="#444")
# #     ax_a.set_ylabel("Attention (%)", fontsize=9, color="#444")
# #     ax_a.set_xlim(0, T_real - 1); ax_a.set_ylim(bottom=0)
# #     ax_a.legend(fontsize=8.5, framealpha=0.92,
# #                 edgecolor="#ddd", loc="upper right")
# #     ax_a.spines["top"].set_visible(False)
# #     ax_a.spines["right"].set_visible(False)
# #     ax_a.set_facecolor("#ffffff")
# #     ax_a.grid(axis="y", alpha=0.25, lw=0.5)
# #     ax_a.tick_params(labelsize=8)

# #     # ── ROW 2 RIGHT: cumulative ────────────────────────────────────────────────
# #     ax_c  = fig.add_subplot(gs[2, 2:])
# #     total = matrix_r.mean(axis=0); total /= total.sum()
# #     cum   = np.cumsum(total)
# #     ax_c.fill_between(frames_r, total * 100, alpha=0.28, color="#8e44ad")
# #     ax_c.plot(frames_r, total * 100, color="#8e44ad",
# #               lw=2.2, label="Total attention")
# #     ax_c2 = ax_c.twinx()
# #     ax_c2.plot(frames_r, cum * 100, color="#2c3e50",
# #                lw=1.8, ls="--", label="Cumulative (%)")

# #     for pct, col in [(50, "#27ae60"), (90, "#e67e22")]:
# #         ax_c2.axhline(pct, color=col, lw=1, ls=":", alpha=0.7)
# #         fi = int(np.searchsorted(cum, pct / 100))
# #         if fi < T_real:
# #             ax_c.axvline(fi, color=col, lw=1.2, ls=":", alpha=0.8)
# #             ax_c.text(fi + 0.5, total.max() * 80,
# #                       f"F={fi}", fontsize=7, color=col,
# #                       fontfamily="monospace")
# #         ax_c2.text(T_real - 1, pct + 1, f"{pct}%",
# #                    fontsize=7, color=col, ha="right")

# #     ax_c2.set_ylabel("Cumulative (%)", fontsize=8, color="#2c3e50")
# #     ax_c2.tick_params(labelsize=7.5, colors="#2c3e50")
# #     ax_c2.set_ylim(0, 105)
# #     l1, b1 = ax_c.get_legend_handles_labels()
# #     l2, b2 = ax_c2.get_legend_handles_labels()
# #     ax_c.legend(l1 + l2, b1 + b2, fontsize=8,
# #                 framealpha=0.9, edgecolor="#ddd", loc="upper left")
# #     ax_c.set_title("Total + Cumulative Attention", fontsize=10.5,
# #                    fontweight="bold", fontfamily="DejaVu Serif", pad=5)
# #     ax_c.set_xlabel("Frame", fontsize=9, color="#444")
# #     ax_c.set_ylabel("Total Attention (%)", fontsize=9, color="#444")
# #     ax_c.set_xlim(0, T_real - 1); ax_c.set_ylim(bottom=0)
# #     ax_c.spines["top"].set_visible(False)
# #     ax_c.set_facecolor("#ffffff")
# #     ax_c.grid(axis="y", alpha=0.25, lw=0.5)
# #     ax_c.tick_params(labelsize=8)

# #     # Border
# #     fig.add_artist(FancyBboxPatch(
# #         (0.005, 0.005), 0.990, 0.990,
# #         boxstyle="square,pad=0", linewidth=3,
# #         edgecolor="#4CAF50", facecolor="none",
# #         transform=fig.transFigure, clip_on=False))

# #     out = os.path.join(out_dir, f"attention_{tag}.png")
# #     fig.savefig(out, dpi=120, facecolor=BG, format="png")
# #     plt.close(fig)
# #     print(f"  Saved: {out}")
# #     return out

"""
Drop-in replacement for plot_attention() in step_by_step_attention.py

HOW TO USE:
  1. Open step_by_step_attention.py
  2. Find:   def plot_attention(att_weights, caption, T, out_dir, tag):
  3. Delete that entire function
  4. Paste this whole file in its place
"""

def plot_attention(att_weights, caption, T, out_dir, tag):
    import os
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.colors import LinearSegmentedColormap

    os.makedirs(out_dir, exist_ok=True)
    parts  = list(att_weights.keys())
    matrix = np.stack([att_weights[p] for p in parts], axis=0)   # (4, T)
    frames = np.arange(T)

    # ── FIXED DISPLAY LENGTH: SHOW ONLY FIRST 100 FRAMES ─────────────────────
    # Attention visualization only focuses on meaningful motion frames.
    # The remaining frames are batch padding.
    MAX_FRAMES = 100

    if T > MAX_FRAMES:
        T_real = MAX_FRAMES
    else:
        T_real = T

    print(f"  Displaying attention: {T_real}/{T} frames")

    frames_r = np.arange(T_real)
    matrix_r = matrix[:, :T_real].copy()

    # Re-normalise attention over displayed frames
    for i in range(matrix_r.shape[0]):
        s = matrix_r[i].sum()
        if s > 0:
            matrix_r[i] /= s

    att_r = {p: matrix_r[pi] for pi, p in enumerate(parts)}
    # ── SMART padding detection ────────────────────────────────────────────────
    # Only cut if there is a sudden JUMP to flat (not gradual softmax spread).
    # Compare variance in first half vs second half — if second half is
    # suspiciously much flatter AND is perfectly constant, cut there.
    # Otherwise use all T frames.
    # total_att = matrix.mean(axis=0)
    # T_real = T   # default: use everything

    # # Look for a hard plateau: a region where every value is IDENTICAL
    # # (this happens with padding zeros fed through softmax)
    # for i in range(T - 3, 2, -1):
    #     segment = total_att[i:]
    #     if segment.std() < 1e-7 and i < T * 0.95:
    #         T_real = i
    #         break

    # print(f"  Sequence length: {T_real} / {T} frames used")

    # Safety: never less than 10 frames


    # T_real = max(T_real, min(10, T))

    # frames_r = np.arange(T_real)
    # matrix_r = matrix[:, :T_real].copy()

    # # Re-normalise over real frames
    # for i in range(matrix_r.shape[0]):
    #     s = matrix_r[i].sum()
    #     if s > 0:
    #         matrix_r[i] /= s

    # att_r = {p: matrix_r[pi] for pi, p in enumerate(parts)}

    PART_COLORS = {
        "Right Arm": "#c0392b",
        "Left Arm":  "#e74c3c",
        "Right Leg": "#27ae60",
        "Left Leg":  "#2ecc71",
    }
    BG = "#fafafa"
    cmap_heat = LinearSegmentedColormap.from_list(
        "a", ["#ffffff","#fef9e7","#f9ca74",
              "#e67e22","#c0392b","#7b241c"])

    fig = plt.figure(figsize=(18, 14), facecolor=BG, dpi=120)
    gs  = gridspec.GridSpec(3, 4, figure=fig,
                            hspace=0.55, wspace=0.38,
                            left=0.07, right=0.97,
                            top=0.90, bottom=0.07)

    short = (caption[:85] + "...") if len(caption) > 85 else caption
    fig.suptitle(f'Attention Map — {tag}\n"{short}"',
                 fontsize=12, fontweight="bold",
                 fontfamily="DejaVu Serif", color="#1a1a1a", y=0.97)

    # ── ROW 0: per-part curves ────────────────────────────────────────────────
    for pi, part in enumerate(parts):
        c  = PART_COLORS[part]
        ax = fig.add_subplot(gs[0, pi])
        w  = att_r[part] * 100

        ax.fill_between(frames_r, w, alpha=0.22, color=c)
        ax.plot(frames_r, w, color=c, lw=2.0)

        pk = int(np.argmax(att_r[part]))
        ax.axvline(pk, color=c, lw=1.2, ls="--", alpha=0.6)
        ax.scatter([pk], [w[pk]], s=70, color=c,
                   zorder=5, edgecolors="white", lw=1.2)
        if T_real > 1:
            ax.text(pk + max(1, T_real * 0.01), w[pk] * 1.03,
                    f"t={pk}", fontsize=7, color=c,
                    fontfamily="monospace")

        H = -np.sum(att_r[part] * np.log(att_r[part] + 1e-12))
        ax.text(0.97, 0.94, f"H={H:.2f}", transform=ax.transAxes,
                fontsize=7.5, ha="right", va="top", color="#555",
                bbox=dict(boxstyle="round,pad=0.25",
                          fc="white", ec="#ccc", lw=0.8))

        ax.set_title(part, fontsize=10, fontweight="bold",
                     color=c, fontfamily="DejaVu Serif", pad=4)
        ax.set_xlabel("Frame", fontsize=8, color="#555")
        ax.set_ylabel("Attention (%)", fontsize=8, color="#555")
        if T_real > 1:
            ax.set_xlim(0, T_real - 1)
        ax.set_ylim(bottom=0)
        ax.tick_params(labelsize=7.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_facecolor("#ffffff")
        ax.grid(axis="y", alpha=0.25, lw=0.5)

    # ── ROW 1 LEFT: per-row normalised heatmap ────────────────────────────────
    ax_h = fig.add_subplot(gs[1, :3])
    disp = matrix_r.copy()
    for i in range(disp.shape[0]):
        mn, mx = disp[i].min(), disp[i].max()
        if mx > mn:
            disp[i] = (disp[i] - mn) / (mx - mn)
        else:
            disp[i] = np.zeros_like(disp[i])

    im = ax_h.imshow(disp, aspect="auto", cmap=cmap_heat,
                     interpolation="bilinear", vmin=0, vmax=1)

    for ri, part in enumerate(parts):
        pk = int(np.argmax(matrix_r[ri]))
        ax_h.scatter([pk], [ri], marker="D", s=90,
                     color="#1a237e", zorder=6, lw=0)
        ax_h.text(pk + max(0.5, T_real * 0.005), ri - 0.28, str(pk),
                  fontsize=6.5, color="#1a237e", fontfamily="monospace")

    cbar = plt.colorbar(im, ax=ax_h, shrink=0.9, pad=0.01)
    cbar.set_label("Normalised Attention (per part)", fontsize=8)
    ax_h.set_yticks(range(4))
    ax_h.set_yticklabels(parts, fontsize=9.5, fontfamily="DejaVu Serif")
    for tick, part in zip(ax_h.get_yticklabels(), parts):
        tick.set_color(PART_COLORS[part])
        tick.set_fontweight("bold")
    ax_h.set_xlabel("Frame", fontsize=9, color="#444")
    ax_h.set_title("Attention Heatmap: Body Parts → Torso Frames",
                   fontsize=11, fontweight="bold",
                   fontfamily="DejaVu Serif", pad=6)
    ax_h.tick_params(labelsize=8)

    # ── ROW 1 RIGHT: peak + mean bar ──────────────────────────────────────────
    ax_b = fig.add_subplot(gs[1, 3])
    peaks = [att_r[p].max() * 100 for p in parts]
    means = [att_r[p].mean() * 100 for p in parts]
    stds  = [att_r[p].std()  * 100 for p in parts]
    colors_list = [PART_COLORS[p] for p in parts]

    ax_b.barh(parts, peaks, color=colors_list, alpha=0.85, label="Peak")
    ax_b.barh(parts, means, color=colors_list, alpha=0.35,
              xerr=stds,
              error_kw=dict(ecolor="#555", capsize=3, lw=1.2),
              label="Mean±Std")

    for i, (pv, mv) in enumerate(zip(peaks, means)):
        ax_b.text(max(pv, mv) + 0.05, i,
                  f"pk={pv:.2f}%  μ={mv:.2f}%",
                  va="center", fontsize=6.5, color="#333")

    for tick, part in zip(ax_b.get_yticklabels(), parts):
        tick.set_color(PART_COLORS[part])
        tick.set_fontweight("bold")

    ax_b.set_xlabel("Attention (%)", fontsize=8.5, color="#444")
    ax_b.set_title("Peak (solid)\nMean±Std (light)",
                   fontsize=9, fontweight="bold",
                   fontfamily="DejaVu Serif", pad=4)
    ax_b.legend(fontsize=7, loc="lower right")
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    ax_b.set_facecolor("#ffffff")
    ax_b.tick_params(labelsize=8.5)

    # ── ROW 2 LEFT: overlay ────────────────────────────────────────────────────
    ax_a = fig.add_subplot(gs[2, :2])
    for part in parts:
        w = att_r[part] * 100
        ax_a.plot(frames_r, w, color=PART_COLORS[part],
                  lw=2.0, label=part, alpha=0.9)
        ax_a.fill_between(frames_r, w, alpha=0.07,
                          color=PART_COLORS[part])
    ax_a.set_title("All Parts Overlaid", fontsize=10.5,
                   fontweight="bold", fontfamily="DejaVu Serif", pad=5)
    ax_a.set_xlabel("Frame", fontsize=9, color="#444")
    ax_a.set_ylabel("Attention (%)", fontsize=9, color="#444")
    if T_real > 1:
        ax_a.set_xlim(0, T_real - 1)
    ax_a.set_ylim(bottom=0)
    ax_a.legend(fontsize=8.5, framealpha=0.92,
                edgecolor="#ddd", loc="upper right")
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)
    ax_a.set_facecolor("#ffffff")
    ax_a.grid(axis="y", alpha=0.25, lw=0.5)
    ax_a.tick_params(labelsize=8)

    # ── ROW 2 RIGHT: cumulative ────────────────────────────────────────────────
    ax_c  = fig.add_subplot(gs[2, 2:])
    total = matrix_r.mean(axis=0)
    s = total.sum()
    if s > 0:
        total = total / s
    cum = np.cumsum(total)

    ax_c.fill_between(frames_r, total * 100, alpha=0.28, color="#8e44ad")
    ax_c.plot(frames_r, total * 100, color="#8e44ad",
              lw=2.2, label="Total attention")
    ax_c2 = ax_c.twinx()
    ax_c2.plot(frames_r, cum * 100, color="#2c3e50",
               lw=1.8, ls="--", label="Cumulative (%)")

    for pct, col in [(50, "#27ae60"), (90, "#e67e22")]:
        ax_c2.axhline(pct, color=col, lw=1, ls=":", alpha=0.7)
        fi = int(np.searchsorted(cum, pct / 100))
        if 0 < fi < T_real:
            ax_c.axvline(fi, color=col, lw=1.2, ls=":", alpha=0.8)
            ax_c.text(fi + max(0.5, T_real * 0.01),
                      total.max() * 80,
                      f"F={fi}", fontsize=7, color=col,
                      fontfamily="monospace")
        if T_real > 1:
            ax_c2.text(T_real - 1, pct + 1, f"{pct}%",
                       fontsize=7, color=col, ha="right")

    ax_c2.set_ylabel("Cumulative (%)", fontsize=8, color="#2c3e50")
    ax_c2.tick_params(labelsize=7.5, colors="#2c3e50")
    ax_c2.set_ylim(0, 105)
    l1, b1 = ax_c.get_legend_handles_labels()
    l2, b2 = ax_c2.get_legend_handles_labels()
    ax_c.legend(l1 + l2, b1 + b2, fontsize=8,
                framealpha=0.9, edgecolor="#ddd", loc="upper left")
    ax_c.set_title("Total + Cumulative Attention", fontsize=10.5,
                   fontweight="bold", fontfamily="DejaVu Serif", pad=5)
    ax_c.set_xlabel("Frame", fontsize=9, color="#444")
    ax_c.set_ylabel("Total Attention (%)", fontsize=9, color="#444")
    if T_real > 1:
        ax_c.set_xlim(0, T_real - 1)
    ax_c.set_ylim(bottom=0)
    ax_c.spines["top"].set_visible(False)
    ax_c.set_facecolor("#ffffff")
    ax_c.grid(axis="y", alpha=0.25, lw=0.5)
    ax_c.tick_params(labelsize=8)

    # Border
    fig.add_artist(FancyBboxPatch(
        (0.005, 0.005), 0.990, 0.990,
        boxstyle="square,pad=0", linewidth=3,
        edgecolor="#4CAF50", facecolor="none",
        transform=fig.transFigure, clip_on=False))

    out = os.path.join(out_dir, f"attention_{tag}.png")
    fig.savefig(out, dpi=120, facecolor=BG, format="png")
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # ── Mode A: full run path ──────────────────────────────────────────────────
    parser.add_argument('--run', type=str, default=None,
        help='Full path to run directory (contains .hydra/config.yaml)')

    # ── Mode B: explicit paths (easier) ───────────────────────────────────────
    parser.add_argument('--cfg', type=str, default=None,
        help='Path to config.yaml  e.g. runs/.../.hydra/config.yaml')
    parser.add_argument('--ckpt', type=str, default=None,
        help='Path to checkpoint   e.g. runs/.../best_models/best_model_metric_all.pth')
    parser.add_argument('--npz', type=str, default=None,
        help='Path to _smpl_fit.npz (skips dataset loading entirely)')
    parser.add_argument('--caption', type=str, default='Motion sequence',
        help='Caption text (used when --npz is set)')

    # ── Shared ─────────────────────────────────────────────────────────────────
    parser.add_argument('--set',      type=str, default='test')
    parser.add_argument('--query_id', type=int, default=22)
    parser.add_argument('--out_dir',  type=str,
                        default='outputs/attention_maps')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print('ATTENTION VISUALIZATION')
    print(f"{'='*60}")
    print(f"Device: {device}")

    # ── Resolve config + ckpt paths ───────────────────────────────────────────
    if args.run:
        cfg_path  = str(Path(args.run) / '.hydra' / 'config.yaml')
        ckpt_path = str(Path(args.run) / 'best_models' /
                        'best_model_metric_all.pth')
    else:
        cfg_path  = args.cfg
        ckpt_path = args.ckpt

    # ── STEP 2: Load + patch model ─────────────────────────────────────────────
    print('\n[STEP 2] Loading and patching model...')
    model, cfg = load_model(cfg_path, ckpt_path, device)
    part_modules = patch_attention_fuse(model)

    # ── STEP 3: Get motion ─────────────────────────────────────────────────────
    if args.npz:
        print(f'\n[STEP 3] Loading motion from NPZ: {args.npz}')
        motion, T = motion_from_npz(args.npz, device)
        caption   = args.caption
        tag       = Path(args.npz).stem
    else:
        print(f'\n[STEP 3] Loading motion from dataset (query_id={args.query_id})')
        motion, T, caption = motion_from_dataset(
            cfg, args.set, args.query_id, device)
        tag = f'q{args.query_id}'

    # ── STEP 4: Extract attention ──────────────────────────────────────────────
    print('\n[STEP 4] Running forward pass...')
    att_weights = extract_attention(model, motion, part_modules, device)

    # ── STEP 5: Plot ──────────────────────────────────────────────────────────
    print('\n[STEP 5] Plotting...')
    plot_attention(att_weights, caption, T, args.out_dir, tag)

    print(f"\n{'='*60}")
    print('DONE — check:', args.out_dir)
    print(f"{'='*60}")