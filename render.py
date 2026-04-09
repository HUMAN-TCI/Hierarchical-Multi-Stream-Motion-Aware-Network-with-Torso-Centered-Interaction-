# # import os
# # from pathlib import Path
# # import textwrap
# # import hydra
# # import numpy as np
# # import pandas as pd
# # from omegaconf import OmegaConf

# # import torch
# # import logging
# # import argparse

# # from models.model import MatchingModel
# # import evaluation
# # from utils.visualization import plot_3d_motion

# # log = logging.getLogger(__name__)
# # logging.basicConfig(level=logging.INFO)

# # def main():
# #     # --------------------------
# #     # Argument Parsing
# #     # --------------------------
# #     parser = argparse.ArgumentParser()
# #     parser.add_argument('--run', help='Path to run dir')
# #     parser.add_argument('--set', default='val', type=str, help='Set on which inference is performed.')
# #     parser.add_argument('--best_on_metric', default='all', help='Select snapshot that optimizes this metric')
# #     parser.add_argument('--override_existing_videos', action='store_true', help='Override existing videos')
# #     parser.add_argument('--render_descriptions', action='store_true', help='Render descriptions on each video')
# #     parser.add_argument('--query_ids_to_render', type=int, nargs='+', default=[0], help='IDs of queries to render')

# #     args = parser.parse_args()
# #     log.info(f"Rendering ids: {args.query_ids_to_render}")

# #     run_path = Path(args.run)
# #     if not run_path.exists():
# #         log.warning(f'This path ({run_path}) does not exist. Exiting.')
# #         exit(1)

# #     # --------------------------
# #     # Load Configuration
# #     # --------------------------
# #     hydra_cfg = OmegaConf.load(run_path / '.hydra' / 'hydra.yaml')['hydra']
# #     OmegaConf.register_new_resolver("hydra", lambda x: OmegaConf.select(hydra_cfg, x))
# #     cfg = OmegaConf.load(run_path / '.hydra' / 'config.yaml')
# #     print(OmegaConf.to_yaml(cfg))

# #     assert cfg.data.test.name == 'humanml', "Visualization only implemented for HumanML3D dataset"

# #     # --------------------------
# #     # Checkpoint Validation
# #     # --------------------------
# #     last_checkpoint = run_path / 'last.pt'
# #     if not last_checkpoint.is_file():
# #         log.warning('Checkpoint not found. Exiting...')
# #         exit(1)
# #     else:
# #         checkpoint = torch.load(last_checkpoint, map_location='cpu')
# #         if checkpoint['epoch'] < cfg.optim.epochs - 1:
# #             log.warning("Run incomplete. Exiting...")
# #             exit(1)

# #     # --------------------------
# #     # Dataset & Dataloader
# #     # --------------------------
# #     batch_size = cfg.optim.batch_size
# #     dataset_cfg = getattr(cfg.data, args.set)
# #     dataloader = hydra.utils.call(dataset_cfg, batch_size=batch_size)

# #     # --------------------------
# #     # Load Model
# #     # --------------------------
# #     model = MatchingModel(cfg)
# #     if torch.cuda.is_available():
# #         model.cuda()
# #     model.eval()

# #     best_models_folder = run_path / 'best_models'
# #     metric_name = args.best_on_metric.replace('/', '-')
# #     ckpt_path = best_models_folder / f'best_model_metric_{metric_name}.pth'
# #     log.info(f"[CKPT]: Loading {ckpt_path}")
# #     checkpoint = torch.load(ckpt_path, map_location='cpu')
# #     model.load_state_dict(checkpoint['model'], strict=True)

# #     # --------------------------
# #     # Encode & Compute Metrics
# #     # --------------------------
# #     motion_feats, caption_feats, motion_labels = evaluation.encode_data(model, dataloader)
# #     metrics, idxs, postproc_descriptions, ranks, ordered_relevances = evaluation.compute_recall(
# #         args.set, caption_feats, motion_feats, motion_labels, dataloader.dataset,
# #         return_all=True, top_k_to_return=16
# #     )
# #     log.info(metrics)

# #     # --------------------------
# #     # Rendering
# #     # --------------------------
# #     kinematic_chain = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
# #     fps = 20  # Set frame rate for GIFs

# #     for query_id_to_render in args.query_ids_to_render:
# #         all_descriptions = [dataloader.dataset[i]['desc'] for i in range(len(dataloader.dataset))]
# #         paths = [os.path.join('dataset', 'HumanML3D', 'new_joints', dataloader.dataset[i]['path']) + '.npy' for i in range(len(dataloader.dataset))]
# #         _, q_idx = np.unique(np.asarray(all_descriptions), return_index=True)
# #         useful_queries_idxs = np.sort(q_idx)

# #         output_path = os.path.join('outputs', 'renders', args.set, str(query_id_to_render))
# #         os.makedirs(output_path, exist_ok=True)

# #         # Save query description and ranks
# #         with open(os.path.join(output_path, 'desc.txt'), 'w') as f:
# #             f.write('{}\n{}'.format(
# #                 all_descriptions[useful_queries_idxs[query_id_to_render]],
# #                 ranks[query_id_to_render]
# #             ))

# #         # Select top-k paths
# #         paths_to_render = [paths[idx] for idx in idxs[query_id_to_render]]
# #         wrapped_descriptions = ['{}; {}; (nDCG:spice={:.2f};spacy={:.2f})'.format(
# #             i,
# #             '; '.join(postproc_descriptions[query_id_to_render][i]),
# #             1.0,  # Spice score manually set to 1.0
# #             ordered_relevances['spacy'][query_id_to_render, i])
# #             for i in range(len(paths_to_render))
# #         ]
# #         wrapped_descriptions = ['\n'.join(textwrap.wrap(d, 50)) for d in wrapped_descriptions]
# #         output_filenames = ['{}/{}_{}.gif'.format(output_path, i, os.path.basename(path)) for i, path in enumerate(paths_to_render)]

# #         # Render & save motion
# #         for npy_file, output_filename, description in zip(paths_to_render, output_filenames, wrapped_descriptions):
# #             data = np.load(npy_file)
# #             caption = description if args.render_descriptions else ''
# #             ani = plot_3d_motion(output_filename, kinematic_chain, data, title=caption, fps=fps, radius=4, dist=3, figsize=(5, 5))
# #             ani.save(output_filename, fps=fps)

# # if __name__ == '__main__':
# #     main()
# import os
# from pathlib import Path
# import textwrap
# import hydra
# import numpy as np
# import torch
# import logging
# import argparse
# from omegaconf import OmegaConf
# from models.model import MatchingModel
# import evaluation
# from utils.visualization import plot_3d_motion  # This function returns animation object (ani)

# log = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)


# def main():
#     # --------------------------
#     # Argument Parsing
#     # --------------------------
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--run', help='Path to run dir', required=True)
#     parser.add_argument('--set', default='val', type=str, help='Set on which inference is performed.')
#     parser.add_argument('--best_on_metric', default='all', help='Select snapshot that optimizes this metric')
#     parser.add_argument('--override_existing_videos', action='store_true', help='Override existing videos')
#     parser.add_argument('--render_descriptions', action='store_true', help='Render descriptions on each video')
#     parser.add_argument('--query_ids_to_render', type=int, nargs='+', default=[0], help='IDs of queries to render')

#     args = parser.parse_args()
#     log.info(f"Rendering ids: {args.query_ids_to_render}")

#     run_path = Path(args.run)
#     if not run_path.exists():
#         log.warning(f'This path ({run_path}) does not exist. Exiting.')
#         exit(1)

#     # --------------------------
#     # Load Configuration
#     # --------------------------
#     hydra_cfg = OmegaConf.load(run_path / '.hydra' / 'hydra.yaml')['hydra']
#     OmegaConf.register_new_resolver("hydra", lambda x: OmegaConf.select(hydra_cfg, x))
#     cfg = OmegaConf.load(run_path / '.hydra' / 'config.yaml')
#     print(OmegaConf.to_yaml(cfg))

#     assert cfg.data.test.name == 'humanml', "Visualization only implemented for HumanML3D dataset"

#     # --------------------------
#     # Checkpoint Validation
#     # --------------------------
#     last_checkpoint = run_path / 'last.pt'
#     if not last_checkpoint.is_file():
#         log.warning('Checkpoint not found. Exiting...')
#         exit(1)
#     else:
#         checkpoint = torch.load(last_checkpoint, map_location='cpu')
#         if checkpoint.get('epoch', 0) < cfg.optim.epochs - 1:
#             log.warning("Run incomplete. Exiting...")
#             exit(1)

#     # --------------------------
#     # Dataset & Dataloader
#     # --------------------------
#     batch_size = cfg.optim.batch_size
#     dataset_cfg = getattr(cfg.data, args.set)
#     dataloader = hydra.utils.call(dataset_cfg, batch_size=batch_size)

#     # --------------------------
#     # Load Model
#     # --------------------------
#     model = MatchingModel(cfg)
#     if torch.cuda.is_available():
#         model.cuda()
#     model.eval()

#     best_models_folder = run_path / 'best_models'
#     metric_name = args.best_on_metric.replace('/', '-')
#     ckpt_path = best_models_folder / f'best_model_metric_{metric_name}.pth'
#     log.info(f"[CKPT]: Loading {ckpt_path}")
#     checkpoint = torch.load(ckpt_path, map_location='cpu')
#     model.load_state_dict(checkpoint['model'], strict=True)

#     # --------------------------
#     # Encode & Compute Metrics
#     # --------------------------
#     motion_feats, caption_feats, motion_labels = evaluation.encode_data(model, dataloader)
#     metrics, idxs, postproc_descriptions, ranks, ordered_relevances = evaluation.compute_recall(
#         args.set, caption_feats, motion_feats, motion_labels, dataloader.dataset,
#         return_all=True, top_k_to_return=16
#     )
#     log.info(metrics)

#     # --------------------------
#     # Rendering
#     # --------------------------
#     kinematic_chain = [
#         (0, 2), (2, 5), (5, 8), (8, 11),
#         (0, 1), (1, 4), (4, 7), (7, 10),
#         (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
#         (9, 14), (14, 17), (17, 19), (19, 21),
#         (9, 13), (13, 16), (16, 18), (18, 20)
#     ]

#     fps = 20

#     all_descriptions = [dataloader.dataset[i]['desc'] for i in range(len(dataloader.dataset))]
#     paths = [os.path.join('dataset', 'HumanML3D', 'new_joints', dataloader.dataset[i]['path']) + '.npy' for i in range(len(dataloader.dataset))]
#     _, q_idx = np.unique(np.asarray(all_descriptions), return_index=True)
#     useful_queries_idxs = np.sort(q_idx)

#     for query_id_to_render in args.query_ids_to_render:
#         output_path = os.path.join('outputs', 'renders', args.set, str(query_id_to_render))
#         os.makedirs(output_path, exist_ok=True)

#         with open(os.path.join(output_path, 'desc.txt'), 'w') as f:
#             f.write('{}\n{}'.format(
#                 all_descriptions[useful_queries_idxs[query_id_to_render]],
#                 ranks[query_id_to_render]
#             ))

#         paths_to_render = [paths[idx] for idx in idxs[query_id_to_render]]
#         wrapped_descriptions = ['{}; {}; (nDCG:spice={:.2f};spacy={:.2f})'.format(
#             i,
#             '; '.join(postproc_descriptions[query_id_to_render][i]),
#             1.0,
#             ordered_relevances['spacy'][query_id_to_render, i])
#             for i in range(len(paths_to_render))
#         ]
#         wrapped_descriptions = ['\n'.join(textwrap.wrap(d, 50)) for d in wrapped_descriptions]
#         output_filenames = ['{}/{}_{}.gif'.format(output_path, i, os.path.basename(path)) for i, path in enumerate(paths_to_render)]

#         for npy_file, output_filename, description in zip(paths_to_render, output_filenames, wrapped_descriptions):
#             if os.path.exists(output_filename) and not args.override_existing_videos:
#                 log.info(f"Skipping existing file {output_filename}")
#                 continue

#             data = np.load(npy_file)
#             caption = description if args.render_descriptions else ''

#             ani = plot_3d_motion(
#                 None, kinematic_chain, data,
#                 title=caption, fps=fps, radius=4, dist=3, figsize=(5, 5)
#             )

#             # Explicitly save the animation
#             ani.save(output_filename, fps=fps, writer='pillow')
#             log.info(f"Rendered and saved animation to {output_filename}")


# if __name__ == '__main__':
#     main()

# Original code without modification ////////////////////////////////

# import os
# from pathlib import Path
# import textwrap
# import hydra
# import numpy as np
# import torch
# import logging
# import argparse
# # At the top of your render.py
# from omegaconf import OmegaConf
# from models.model import MatchingModel
# import evaluation
# from utils.visualization import plot_3d_motion  # Make sure this function returns the animation object

# log = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)


# def main():
#     print ("entry point 1")
#     # --------------------------
#     # Argument Parsing
#     # --------------------------
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--run', help='Path to run dir', required=True)
#     parser.add_argument('--set', default='val', type=str, help='Set on which inference is performed.')
#     parser.add_argument('--best_on_metric', default='all', help='Select snapshot that optimizes this metric')
#     parser.add_argument('--override_existing_videos', action='store_true', help='Override existing videos')
#     parser.add_argument('--render_descriptions', action='store_true', help='Render descriptions on each video')
#     parser.add_argument('--query_ids_to_render', type=int, nargs='+', default=[0], help='IDs of queries to render')

  
  
#     args = parser.parse_args()
#     log.info(f"Rendering ids: {args.query_ids_to_render}")

#     run_path = Path(args.run)
#     if not run_path.exists():
#         log.warning(f'This path ({run_path}) does not exist. Exiting.')
#         exit(1)

#     # --------------------------
#     # Load Configuration
#     # --------------------------
#     hydra_cfg = OmegaConf.load(run_path / '.hydra' / 'hydra.yaml')['hydra']
#     OmegaConf.register_new_resolver("hydra", lambda x: OmegaConf.select(hydra_cfg, x))
#     cfg = OmegaConf.load(run_path / '.hydra' / 'config.yaml')
#     # print(OmegaConf.to_yaml(cfg))

#     assert cfg.data.test.name == 'humanml', "Visualization only implemented for HumanML3D dataset"

#     # --------------------------
#     # Checkpoint Validation
#     # --------------------------
#     last_checkpoint = run_path / 'last.pt'
#     if not last_checkpoint.is_file():
#         log.warning('Checkpoint not found. Exiting...')
#         exit(1)
#     else:
#         checkpoint = torch.load(last_checkpoint, map_location='cpu')
#         if checkpoint.get('epoch', 0) < cfg.optim.epochs - 1:
#             log.warning("Run incomplete. Exiting...")
#             exit(1)

#     # --------------------------
#     # Dataset & Dataloader
#     # --------------------------
#     batch_size = cfg.optim.batch_size
#     dataset_cfg = getattr(cfg.data, args.set)
#     dataloader = hydra.utils.call(dataset_cfg, batch_size=batch_size)

#     # --------------------------
#     # Load Model
#     # --------------------------
#     model = MatchingModel(cfg)
#     if torch.cuda.is_available():
#         model.cuda()
#     model.eval()

#     best_models_folder = run_path / 'best_models'
#     metric_name = args.best_on_metric.replace('/', '-')
#     ckpt_path = best_models_folder / f'best_model_metric_{metric_name}.pth'
#     log.info(f"[CKPT]: Loading {ckpt_path}")
#     checkpoint = torch.load(ckpt_path, map_location='cpu')
#     model.load_state_dict(checkpoint['model'], strict=False)

#     # --------------------------
#     # Encode & Compute Metrics
#     # --------------------------
#     motion_feats, caption_feats, motion_labels = evaluation.encode_data(model, dataloader)
#     metrics, idxs, postproc_descriptions, ranks, ordered_relevances = evaluation.compute_recall(
#         args.set, caption_feats, motion_feats, motion_labels, dataloader.dataset,
#         return_all=True, top_k_to_return=16
#     )
#     log.info(metrics)

#     # --------------------------
#     # Rendering
#     # --------------------------
#     # Fix kinematic_chain format to a flat list of edges [(jointA, jointB), ...]
#     # Original nested lists don't represent edges correctly
#     kinematic_chain = [
#         (0, 2), (2, 5), (5, 8), (8, 11),
#         (0, 1), (1, 4), (4, 7), (7, 10),
#         (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
#         (9, 14), (14, 17), (17, 19), (19, 21),
#         (9, 13), (13, 16), (16, 18), (18, 20)
#     ]

#     fps = 20  # Set frame rate for GIFs
#     print ("entry point 2")

#     all_descriptions = [dataloader.dataset[i]['desc'] for i in range(len(dataloader.dataset))]
#     paths = [os.path.join('dataset', 'HumanML3D', 'new_joints', dataloader.dataset[i]['path']) + '.npy' for i in range(len(dataloader.dataset))]
#     _, q_idx = np.unique(np.asarray(all_descriptions), return_index=True)
#     useful_queries_idxs = np.sort(q_idx)

#     for query_id_to_render in args.query_ids_to_render:
#         output_path = os.path.join('outputs', 'renders', args.set, str(query_id_to_render))
#         os.makedirs(output_path, exist_ok=True)

#         # Save query description and ranks
#         with open(os.path.join(output_path, 'desc.txt'), 'w') as f:
#             f.write('{}\n{}\n{}'.format(
#                 all_descriptions[useful_queries_idxs[query_id_to_render]],
#                 useful_queries_idxs[query_id_to_render],
#                 ranks[query_id_to_render]
#             ))

#         # Select top-k paths
#         paths_to_render = [paths[idx] for idx in idxs[query_id_to_render]]
#         wrapped_descriptions = ['{}; {}; (nDCG:spice={:.2f};spacy={:.2f})'.format(
#             i,
#             '; '.join(postproc_descriptions[query_id_to_render][i]),
#             1.0,  # Spice score manually set to 1.0
#             ordered_relevances['spacy'][query_id_to_render, i])
#             for i in range(len(paths_to_render))
#         ]
#         wrapped_descriptions = ['\n'.join(textwrap.wrap(d, 50)) for d in wrapped_descriptions]
#         output_filenames = ['{}/{}_{}.gif'.format(output_path, i, os.path.basename(path)) for i, path in enumerate(paths_to_render)]

#         # Render & save motion
#         for npy_file, output_filename, description in zip(paths_to_render, output_filenames, wrapped_descriptions):
#             if os.path.exists(output_filename) and not args.override_existing_videos:
#                 log.info(f"Skipping existing file {output_filename}")
#                 continue

#             data = np.load(npy_file)
#             caption = description if args.render_descriptions else ''

#             ani = plot_3d_motion(
#                 output_filename, kinematic_chain, data,
#                 title=caption, fps=fps, radius=4, dist=3, figsize=(5, 5)
#             )
#             # No need to call ani.save() here if plot_3d_motion already saves it internally
#             # Just ensure plot_3d_motion returns the animation object if further processing needed

#             log.info(f"Rendered and saved animation to {output_filename}")


# if __name__ == '__main__':
#     main()


# ///////////////////////////////////  with projection  // skeleton running 

import os
from pathlib import Path
import textwrap
import hydra
import numpy as np
import torch
import logging
import argparse
from omegaconf import OmegaConf
from models.model import MatchingModel
import evaluation
from utils.visualization import plot_3d_motion  # Make sure this function returns the animation object
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_checkpoint_compatible(model, checkpoint_path):
    """
    Load checkpoint weights only for matching parameter shapes.
    Skips weights that have different shapes (like modified LSTM with torso).
    """
    # checkpoint = torch.load(checkpoint_path, map_location='cpu')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model']
    model_dict = model.state_dict()

    compatible_dict = {k: v for k, v in state_dict.items()
                       if k in model_dict and v.shape == model_dict[k].shape}

    model_dict.update(compatible_dict)
    model.load_state_dict(model_dict)
    print(f"[Checkpoint] Loaded {len(compatible_dict)}/{len(model_dict)} parameters from checkpoint.")
    return model


def main():
    print("Entry point 1")
    # --------------------------
    # Argument Parsing
    # --------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', help='Path to run dir', required=True)
    parser.add_argument('--set', default='val', type=str, help='Set on which inference is performed.')
    parser.add_argument('--best_on_metric', default='all', help='Select snapshot that optimizes this metric')
    parser.add_argument('--override_existing_videos', action='store_true', help='Override existing videos')
    parser.add_argument('--render_descriptions', action='store_true', help='Render descriptions on each video')
    parser.add_argument('--query_ids_to_render', type=int, nargs='+', default=[0], help='IDs of queries to render')
    args = parser.parse_args()
    log.info(f"Rendering ids: {args.query_ids_to_render}")

    run_path = Path(args.run)
    if not run_path.exists():
        log.warning(f'This path ({run_path}) does not exist. Exiting.')
        exit(1)

    # --------------------------
    # Load Configuration
    # --------------------------
    hydra_cfg = OmegaConf.load(run_path / '.hydra' / 'hydra.yaml')['hydra']
    OmegaConf.register_new_resolver("hydra", lambda x: OmegaConf.select(hydra_cfg, x))
    cfg = OmegaConf.load(run_path / '.hydra' / 'config.yaml')
    assert cfg.data.test.name == 'humanml', "Visualization only implemented for HumanML3D dataset"

    # --------------------------
    # Checkpoint Validation
    # --------------------------
    last_checkpoint = run_path / 'last.pt'
    if not last_checkpoint.is_file():
        log.warning('Checkpoint not found. Exiting...')
        exit(1)
    else:
        # checkpoint = torch.load(last_checkpoint, map_location='cpu')
        checkpoint = torch.load(last_checkpoint, map_location='cpu', weights_only=False)
        if checkpoint.get('epoch', 0) < cfg.optim.epochs - 1:
            log.warning("Run incomplete. Exiting...")
            exit(1)

    # --------------------------
    # Dataset & Dataloader
    # --------------------------
    batch_size = cfg.optim.batch_size
    dataset_cfg = getattr(cfg.data, args.set)
    dataloader = hydra.utils.call(dataset_cfg, batch_size=batch_size)

    # --------------------------
    # Load Model
    # --------------------------
    model = MatchingModel(cfg)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    # --------------------------
    # Safe checkpoint loading
    # --------------------------
    best_models_folder = run_path / 'best_models'
    metric_name = args.best_on_metric.replace('/', '-')
    ckpt_path = best_models_folder / f'best_model_metric_{metric_name}.pth'
    log.info(f"[CKPT]: Loading {ckpt_path}")
    model = load_checkpoint_compatible(model, ckpt_path)

    # --------------------------
    # Encode & Compute Metrics
    # --------------------------
    motion_feats, caption_feats, motion_labels = evaluation.encode_data(model, dataloader)
    metrics, idxs, postproc_descriptions, ranks, ordered_relevances = evaluation.compute_recall(
        args.set, caption_feats, motion_feats, motion_labels, dataloader.dataset,
        return_all=True, top_k_to_return=1
    )
    log.info(metrics)

    # --------------------------
    # Rendering
    # --------------------------
    kinematic_chain = [
        (0, 2), (2, 5), (5, 8), (8, 11),
        (0, 1), (1, 4), (4, 7), (7, 10),
        (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
        (9, 14), (14, 17), (17, 19), (19, 21),
        (9, 13), (13, 16), (16, 18), (18, 20)
    ]

    fps = 20
    print("Entry point 2")

    all_descriptions = [dataloader.dataset[i]['desc'] for i in range(len(dataloader.dataset))]
    paths = [os.path.join('dataset', 'HumanML3D', 'new_joints', dataloader.dataset[i]['path']) + '.npy'
             for i in range(len(dataloader.dataset))]
    _, q_idx = np.unique(np.asarray(all_descriptions), return_index=True)
    useful_queries_idxs = np.sort(q_idx)

    for query_id_to_render in args.query_ids_to_render:
        output_path = os.path.join('outputs', 'renders', args.set, str(query_id_to_render))
        os.makedirs(output_path, exist_ok=True)

        # Save query description and ranks
        with open(os.path.join(output_path, 'desc.txt'), 'w') as f:
            f.write('{}\n{}\n{}'.format(
                all_descriptions[useful_queries_idxs[query_id_to_render]],
                useful_queries_idxs[query_id_to_render],
                ranks[query_id_to_render]
            ))

        # Select top-k paths
        paths_to_render = [paths[idx] for idx in idxs[query_id_to_render]]
        wrapped_descriptions = ['{}; {}; (nDCG:spice={:.2f};spacy={:.2f})'.format(
            i,
            '; '.join(postproc_descriptions[query_id_to_render][i]),
            1.0,  # Spice score manually set to 1.0
            ordered_relevances['spacy'][query_id_to_render, i])
            for i in range(len(paths_to_render))
        ]
        wrapped_descriptions = ['\n'.join(textwrap.wrap(d, 50)) for d in wrapped_descriptions]
        output_filenames = ['{}/{}_{}.gif'.format(output_path, i, os.path.basename(path))
                            for i, path in enumerate(paths_to_render)]

        # Render & save motion
        for npy_file, output_filename, description in zip(paths_to_render, output_filenames, wrapped_descriptions):
            if os.path.exists(output_filename) and not args.override_existing_videos:
                log.info(f"Skipping existing file {output_filename}")
                continue

            data = np.load(npy_file)
            caption = description if args.render_descriptions else ''

            ani = plot_3d_motion(
                output_filename, kinematic_chain, data,
                title=caption, fps=fps, radius=1.5, dist=2, figsize=(9, 9)
            )

            log.info(f"Rendered and saved animation to {output_filename}")


if __name__ == '__main__':
    main()



# # SMPL
# import os
# from pathlib import Path
# import textwrap
# import hydra
# import numpy as np
# import torch
# import logging
# import argparse
# from omegaconf import OmegaConf
# from models.model import MatchingModel
# import evaluation
# from utils.visualization import plot_3d_motion
# from smplx import SMPL
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# # -----------------------
# # Logging
# # -----------------------
# log = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)

# # -----------------------
# # Helper: Load checkpoint safely
# # -----------------------
# def load_checkpoint_compatible(model, checkpoint_path):
#     checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
#     state_dict = checkpoint['model']
#     model_dict = model.state_dict()
#     compatible_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
#     model_dict.update(compatible_dict)
#     model.load_state_dict(model_dict)
#     print(f"[Checkpoint] Loaded {len(compatible_dict)}/{len(model_dict)} parameters from checkpoint.")
#     return model

# # -----------------------
# # Helper: Skeleton -> SMPL joints
# # -----------------------
# def skeleton_to_smpl_joints(skel_seq, smpl_model, device):
#     """
#     Converts skeleton sequence (T, J, 3) to SMPL joints (T, 24, 3)
#     Uses only root translation for visualization
#     """
#     smpl_joints = []
#     betas = torch.zeros((1, 10), dtype=torch.float32, device=device)  # default shape

#     for t in range(skel_seq.shape[0]):
#         transl = torch.tensor(skel_seq[t, 0, :], dtype=torch.float32, device=device).unsqueeze(0)
#         global_orient = torch.zeros((1, 3), dtype=torch.float32, device=device)
#         body_pose = torch.zeros((1, 69), dtype=torch.float32, device=device)

#         output = smpl_model(
#             betas=betas,
#             body_pose=body_pose,
#             global_orient=global_orient,
#             transl=transl
#         )
#         joints = output.joints[0].detach().cpu().numpy()  # (24, 3)
#         smpl_joints.append(joints)

#     return np.array(smpl_joints)  # (T, 24, 3)

# # -----------------------
# # Main
# # -----------------------
# def main():
#     # --------------------------
#     # Argument Parsing
#     # --------------------------
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--run', required=True, help='Path to run directory')
#     parser.add_argument('--set', default='val', type=str, help='Dataset split: val/test')
#     parser.add_argument('--best_on_metric', default='all', help='Metric for selecting best checkpoint')
#     parser.add_argument('--query_ids_to_render', type=int, nargs='+', default=[0], help='Query IDs to render')
#     parser.add_argument('--render_descriptions', action='store_true', help='Render descriptions on video')
#     parser.add_argument('--override_existing_videos', action='store_true', help='Override existing files')
#     args = parser.parse_args()
#     log.info(f"Rendering IDs: {args.query_ids_to_render}")

#     run_path = Path(args.run)
#     if not run_path.exists():
#         log.error(f"Run path does not exist: {run_path}")
#         exit(1)

#     # --------------------------
#     # Load Hydra Config
#     # --------------------------
#     hydra_cfg = OmegaConf.load(run_path / '.hydra' / 'hydra.yaml')['hydra']
#     OmegaConf.register_new_resolver("hydra", lambda x: OmegaConf.select(hydra_cfg, x))
#     cfg = OmegaConf.load(run_path / '.hydra' / 'config.yaml')
#     assert cfg.data.test.name == 'humanml', "Only HumanML3D dataset supported"

#     # --------------------------
#     # Dataset & Dataloader
#     # --------------------------
#     batch_size = cfg.optim.batch_size
#     dataset_cfg = getattr(cfg.data, args.set)
#     dataloader = hydra.utils.call(dataset_cfg, batch_size=batch_size)

#     # --------------------------
#     # Load Retrieval Model
#     # --------------------------
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = MatchingModel(cfg).to(device)
#     model.eval()

#     # --------------------------
#     # Load Best Checkpoint
#     # --------------------------
#     best_models_folder = run_path / 'best_models'
#     metric_name = args.best_on_metric.replace('/', '-')
#     ckpt_path = best_models_folder / f'best_model_metric_{metric_name}.pth'
#     if not ckpt_path.is_file():
#         log.error(f"Checkpoint not found: {ckpt_path}")
#         exit(1)
#     log.info(f"Loading checkpoint {ckpt_path}")
#     model = load_checkpoint_compatible(model, ckpt_path)

#     # --------------------------
#     # Load SMPL Model
#     # --------------------------
#     smpl_model_path = r"C:\text-to-motion-retrieval_Exp\smpl_models\smpl\SMPL"
#     if not os.path.exists(smpl_model_path):
#         log.error(f"SMPL path does not exist: {smpl_model_path}")
#         exit(1)
#     smpl = SMPL(model_path=smpl_model_path, gender='neutral', batch_size=1).to(device)
#     log.info("SMPL model loaded.")

#     # --------------------------
#     # Encode & Retrieve Motions
#     # --------------------------
#     motion_feats, caption_feats, motion_labels = evaluation.encode_data(model, dataloader)
#     metrics, idxs, postproc_descriptions, ranks, ordered_relevances = evaluation.compute_recall(
#         args.set,
#         caption_feats,
#         motion_feats,
#         motion_labels,
#         dataloader.dataset,
#         return_all=True,
#         top_k_to_return=1
#     )
#     log.info(f"Metrics: {metrics}")

#     # --------------------------
#     # Rendering Setup
#     # --------------------------
#     kinematic_chain = [
#         (0, 2), (2, 5), (5, 8), (8, 11),
#         (0, 1), (1, 4), (4, 7), (7, 10),
#         (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
#         (9, 14), (14, 17), (17, 19), (19, 21),
#         (9, 13), (13, 16), (16, 18), (18, 20)
#     ]
#     fps = 20

#     all_descriptions = [dataloader.dataset[i]['desc'] for i in range(len(dataloader.dataset))]
#     paths = [os.path.join('dataset', 'HumanML3D', 'new_joints', dataloader.dataset[i]['path']) + '.npy' 
#              for i in range(len(dataloader.dataset))]
#     _, q_idx = np.unique(np.asarray(all_descriptions), return_index=True)
#     useful_queries_idxs = np.sort(q_idx)

#     # --------------------------
#     # Render Each Query
#     # --------------------------
#     for query_id_to_render in args.query_ids_to_render:
#         output_path = os.path.join('outputs', 'renders', args.set, str(query_id_to_render))
#         os.makedirs(output_path, exist_ok=True)

#         # Save description & ranks
#         with open(os.path.join(output_path, 'desc.txt'), 'w') as f:
#             f.write('{}\n{}\n{}'.format(
#                 all_descriptions[useful_queries_idxs[query_id_to_render]],
#                 useful_queries_idxs[query_id_to_render],
#                 ranks[query_id_to_render]
#             ))

#         # Top-k retrieval paths
#         paths_to_render = [paths[idx] for idx in idxs[query_id_to_render]]

#         for i, npy_file in enumerate(paths_to_render):
#             output_filename = os.path.join(output_path, f"{i}_{os.path.basename(npy_file)}.gif")
#             smpl_npy_filename = os.path.join(output_path, f"{i}_{os.path.basename(npy_file)}_smpl.npy")

#             if os.path.exists(output_filename) and not args.override_existing_videos:
#                 log.info(f"Skipping existing file {output_filename}")
#                 continue

#             # Load skeleton motion
#             skel_seq = np.load(npy_file)  # (T, J, 3)
#             smpl_joints = skeleton_to_smpl_joints(skel_seq, smpl, device)

#             # Save SMPL joints
#             np.save(smpl_npy_filename, smpl_joints)
#             log.info(f"Saved SMPL joints to {smpl_npy_filename}")

#             # Render animation
#             caption = ''
#             if args.render_descriptions:
#                 desc_texts = '; '.join(postproc_descriptions[query_id_to_render][i])
#                 caption = '\n'.join(textwrap.wrap(desc_texts, 50))

#             ani = plot_3d_motion(
#                 output_filename, kinematic_chain, smpl_joints,
#                 title=caption, fps=fps, radius=1.5, dist=2, figsize=(9, 9)
#             )

#             log.info(f"Rendered animation saved to {output_filename}")


# if __name__ == '__main__':
#     main()