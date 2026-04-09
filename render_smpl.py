import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from pathlib import Path
import textwrap
import argparse
import logging
from utils.visualization import plot_3d_motion, plot_skeleton_ghost

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from models.model import MatchingModel
import evaluation
from utils.visualization import plot_3d_motion
from utils.smpl_fitting import load_smpl_model, fit_smpl_to_sequence

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_checkpoint_compatible(model, checkpoint_path):
    """
    Load only matching checkpoint weights.
    Useful when model definition changed slightly.
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model']
    model_dict = model.state_dict()

    compatible_dict = {
        k: v for k, v in state_dict.items()
        if k in model_dict and v.shape == model_dict[k].shape
    }

    model_dict.update(compatible_dict)
    model.load_state_dict(model_dict)

    log.info(f"[Checkpoint] Loaded {len(compatible_dict)}/{len(model_dict)} parameters from checkpoint.")
    return model


def build_motion_paths(dataset):
    """
    Build absolute/relative paths to new_joints motion files.
    """
    paths = []
    for i in range(len(dataset)):
        rel_path = dataset[i]['path']
        npy_path = os.path.join('dataset', 'HumanML3D', 'new_joints', rel_path) + '.npy'
        paths.append(npy_path)
    return paths


def save_smpl_params(output_dir, fit_result, basename):
    """
    Save fitted SMPL parameters for later reuse.
    """
    os.makedirs(output_dir, exist_ok=True)

    out_file = os.path.join(output_dir, f"{basename}_smpl_fit.npz")
    np.savez_compressed(
        out_file,
        joints=fit_result["joints"],
        vertices=fit_result["vertices"],
        transl=fit_result["transl"],
        global_orient=fit_result["global_orient"],
        body_pose=fit_result["body_pose"],
        betas=fit_result["betas"],
        losses=fit_result["losses"]
    )
    log.info(f"Saved SMPL fit parameters to: {out_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=str, required=True, help='Path to trained run directory')
    parser.add_argument('--set', default='test', type=str, help='Dataset split: train/val/test')
    parser.add_argument('--best_on_metric', default='all', type=str, help='Checkpoint selection metric')
    parser.add_argument('--query_ids_to_render', type=int, nargs='+', default=[0], help='Query IDs to render')
    parser.add_argument('--top_k_to_return', type=int, default=1, help='How many retrieved motions to render per query')
    parser.add_argument('--render_descriptions', action='store_true', help='Overlay text on output gif')
    parser.add_argument('--override_existing_videos', action='store_true', help='Overwrite old gifs')
    parser.add_argument('--smpl_model_dir', type=str, required=True, help='Path to SMPL model folder')
    parser.add_argument('--smpl_num_iters', type=int, default=150, help='Optimization steps per frame')
    parser.add_argument('--smpl_lr', type=float, default=0.05, help='SMPL fitting learning rate')
    parser.add_argument('--save_smpl_npz', action='store_true', help='Save fitted SMPL parameters')
    args = parser.parse_args()

    log.info(f"Rendering IDs: {args.query_ids_to_render}")

    run_path = Path(args.run)
    if not run_path.exists():
        raise FileNotFoundError(f"Run path does not exist: {run_path}")

    # --------------------------
    # Load Hydra config
    # --------------------------
    hydra_cfg = OmegaConf.load(run_path / '.hydra' / 'hydra.yaml')['hydra']
    OmegaConf.register_new_resolver("hydra", lambda x: OmegaConf.select(hydra_cfg, x))
    cfg = OmegaConf.load(run_path / '.hydra' / 'config.yaml')

    assert cfg.data.test.name == 'humanml', "This script currently supports HumanML3D only."

    # --------------------------
    # Check training finished
    # --------------------------
    last_checkpoint = run_path / 'last.pt'
    if not last_checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {last_checkpoint}")

    checkpoint = torch.load(last_checkpoint, map_location='cpu', weights_only=False)
    if checkpoint.get('epoch', 0) < cfg.optim.epochs - 1:
        log.warning("Training may be incomplete based on last checkpoint epoch.")

    # --------------------------
    # Dataloader
    # --------------------------
    batch_size = cfg.optim.batch_size
    dataset_cfg = getattr(cfg.data, args.set)
    dataloader = hydra.utils.call(dataset_cfg, batch_size=batch_size)

    # --------------------------
    # Retrieval model
    # --------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MatchingModel(cfg).to(device)
    model.eval()

    best_models_folder = run_path / 'best_models'
    metric_name = args.best_on_metric.replace('/', '-')
    ckpt_path = best_models_folder / f'best_model_metric_{metric_name}.pth'
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Best checkpoint not found: {ckpt_path}")

    log.info(f"Loading checkpoint {ckpt_path}")
    model = load_checkpoint_compatible(model, ckpt_path)

    # --------------------------
    # Load SMPL model
    # --------------------------
    if not os.path.exists(args.smpl_model_dir):
        raise FileNotFoundError(f"SMPL model folder not found: {args.smpl_model_dir}")

    smpl_model = load_smpl_model(args.smpl_model_dir, device=device, gender='NEUTRAL', batch_size=1)
    log.info("SMPL model loaded.")

    # --------------------------
    # Encode retrieval features
    # --------------------------
    motion_feats, caption_feats, motion_labels = evaluation.encode_data(model, dataloader)

    metrics, idxs, postproc_descriptions, ranks, ordered_relevances = evaluation.compute_recall(
        args.set,
        caption_feats,
        motion_feats,
        motion_labels,
        dataloader.dataset,
        return_all=True,
        top_k_to_return=args.top_k_to_return
    )

    log.info(f"Retrieval metrics: {metrics}")

    # --------------------------
    # Prepare descriptions and motion paths
    # --------------------------
    all_descriptions = [dataloader.dataset[i]['desc'] for i in range(len(dataloader.dataset))]
    paths = build_motion_paths(dataloader.dataset)

    _, q_idx = np.unique(np.asarray(all_descriptions), return_index=True)
    useful_queries_idxs = np.sort(q_idx)

    # HumanML3D 22-joint chain
    kinematic_chain_22 = [
        (0, 2), (2, 5), (5, 8), (8, 11),
        (0, 1), (1, 4), (4, 7), (7, 10),
        (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
        (9, 14), (14, 17), (17, 19), (19, 21),
        (9, 13), (13, 16), (16, 18), (18, 20)
    ]

    # --------------------------
    # Render selected queries
    # --------------------------
    for query_id_to_render in args.query_ids_to_render:
        if query_id_to_render >= len(useful_queries_idxs):
            log.warning(f"Query id {query_id_to_render} out of range. Skipping.")
            continue

        output_dir = os.path.join('outputs', 'renders_smpl', args.set, str(query_id_to_render))
        os.makedirs(output_dir, exist_ok=True)

        query_desc = all_descriptions[useful_queries_idxs[query_id_to_render]]

        with open(os.path.join(output_dir, 'desc.txt'), 'w', encoding='utf-8') as f:
            f.write(f"{query_desc}\n")
            f.write(f"query_index={useful_queries_idxs[query_id_to_render]}\n")
            f.write(f"rank_info={ranks[query_id_to_render]}\n")

        retrieved_motion_indices = idxs[query_id_to_render]
        paths_to_render = [paths[idx] for idx in retrieved_motion_indices]

        wrapped_descriptions = []
        for i in range(len(paths_to_render)):
            desc = '{}; {}; (nDCG:spice={:.2f};spacy={:.2f})'.format(
                i,
                '; '.join(postproc_descriptions[query_id_to_render][i]),
                1.0,
                ordered_relevances['spacy'][query_id_to_render, i]
            )
            wrapped_descriptions.append('\n'.join(textwrap.wrap(desc, 50)))

        output_filenames = [
            os.path.join(output_dir, f"{i}_{os.path.basename(path)}_smpl.gif")
            for i, path in enumerate(paths_to_render)
        ]

        for npy_file, output_filename, description in zip(paths_to_render, output_filenames, wrapped_descriptions):
            if os.path.exists(output_filename) and not args.override_existing_videos:
                log.info(f"Skipping existing file: {output_filename}")
                continue

            if not os.path.exists(npy_file):
                log.warning(f"Motion file not found: {npy_file}")
                continue

            log.info(f"Processing motion file: {npy_file}")
            skeleton_motion = np.load(npy_file)

            if skeleton_motion.ndim != 3 or skeleton_motion.shape[1:] != (22, 3):
                log.warning(f"Unexpected motion shape {skeleton_motion.shape} in {npy_file}, skipping.")
                continue

            log.info(f"Loaded skeleton motion shape: {skeleton_motion.shape}")

            try:
                fit_result = fit_smpl_to_sequence(
                    joint_sequence=skeleton_motion,
                    smpl_model=smpl_model,
                    device=device,
                    num_iters=args.smpl_num_iters,
                    lr=args.smpl_lr,
                    verbose=True
                )

                smpl_joints = fit_result["joints"]         # (T,24,3)
                smpl_joints_22 = smpl_joints[:, :22, :]    # keep compatible with existing renderer

                log.info(f"Fitted SMPL joints shape: {smpl_joints.shape}")
                log.info(f"Using first 22 SMPL joints for rendering: {smpl_joints_22.shape}")
                log.info(f"Average fitting loss: {fit_result['losses'].mean():.6f}")

                if args.save_smpl_npz:
                    basename = os.path.splitext(os.path.basename(npy_file))[0]
                    save_smpl_params(output_dir, fit_result, basename)

                #aption = description if args.render_descriptions else ''
                caption_clean = description if args.render_descriptions else ''
                # plot_3d_motion(
                #     output_filename,
                #     kinematic_chain_22,
                #     smpl_joints_22,
                #     title=caption,
                #     fps=20,
                #     radius=1.5,
                #     dist=2,
                #     figsize=(9, 9)
                # )
                plot_3d_motion(
                output_filename,
                kinematic_chain_22,
                smpl_joints_22,
                title=caption_clean,
                fps=20,
                radius=1.5,
                dist=2,
                figsize=(8, 7),
                elev=15,
                azim=-75,
                )
                # 2. Ghost figure — key poses for paper
                ghost_path = output_filename.replace('_smpl.gif', '_skeleton_ghost.png')
                plot_skeleton_ghost(
                ghost_path,
                kinematic_chain_22,
                smpl_joints_22,
                caption=caption_clean,
                n_keys=8,
                figsize=(16, 6),
                elev=15,           # lower angle — less top empty space
                azim=-60,          # rotate camera to face along spread direction
                spread=0.80,       # was 0.55 — spread figures wider
            )
                #log.info(f"Saved rendered SMPL gif to: {output_filename}")
                log.info(f"Saved GIF: {output_filename}")
                log.info(f"Saved ghost figure: {ghost_path}")

            except Exception as e:
                log.exception(f"Failed processing {npy_file}: {e}")

                rendered_dirs = [
                    os.path.join('outputs', 'renders_smpl', args.set, str(qid))
                    for qid in args.query_ids_to_render
                ]
                # Keep only dirs that actually exist and have a ghost PNG
                valid_dirs = []
                for d in rendered_dirs:
                    if os.path.isdir(d):
                        has_ghost = any(
                            f.endswith('_ghost.png') or f.endswith('ghost.png')
                            for f in os.listdir(d)
                        )
                        if has_ghost:
                            valid_dirs.append(d)

                if valid_dirs:
                    from make_comparison_figure import make_comparison_figure
                    comp_path = os.path.join(
                        'outputs', 'renders_smpl', args.set, 'comparison_figure.png'
                    )
                    make_comparison_figure(
                        query_dirs    = valid_dirs,
                        output_path   = comp_path,
                        highlight_idx = 0,       # first box gets green border
                        dpi           = 200,
                        divider_every = None,    # set to e.g. 3 to add red dashes every 3 boxes
                        box_width     = 2.8,
                        box_height    = 3.0,
                    )
                    log.info(f"Saved comparison figure: {comp_path}")

if __name__ == '__main__':
    main()
    # ── Add this import at the top of render_smpl.py ──────────────────────────────
# from make_comparison_figure import make_comparison_figure


# ── Add this block at the END of main(), after the query render loop ───────────

    

