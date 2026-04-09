import os
from pathlib import Path
import tqdm
import hydra
from hydra.core.hydra_config import HydraConfig
import yaml
import traceback
import torch
import logging
from torch.utils.tensorboard import SummaryWriter
import torch
from omegaconf import DictConfig

torch.serialization.add_safe_globals([DictConfig])
 
import argparse
import utils
# from data import *

from models.model import MatchingModel

from shutil import copyfile
from ast import literal_eval
import evaluation
from utils.checkpoint import CheckpointManager

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path='configs', config_name='config')
def main(cfg):
    try:
        from omegaconf import OmegaConf; print(OmegaConf.to_yaml(cfg))
        log.info(f"Run path: {Path.cwd()}")
        run_dir = HydraConfig.get().runtime.output_dir

        last_checkpoint = Path(run_dir) / 'last.pt'
        if last_checkpoint.is_file():
            # checkpoint = torch.load(last_checkpoint, map_location='cpu')
            checkpoint = torch.load(last_checkpoint, map_location='cpu', weights_only=False)

            checkpoint_epoch = checkpoint['epoch']
            if checkpoint_epoch >= cfg.optim.epochs - 1:
                log.info("This run has already been entirely executed. Exiting....")
                return None

        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu)
        # tb_logger = SummaryWriter(run_dir)
        log_dir = os.path.join("runs", "exp1")
        os.makedirs(log_dir, exist_ok=True)

        tb_logger = SummaryWriter(log_dir=log_dir)

        # torch seed
        torch.manual_seed(cfg.optim.seed)

        batch_size = cfg.optim.batch_size
        print("hhhhhhhhhhhhhhhhhhh", batch_size)
        # Load datasets and create dataloaders
        train_dataloader = hydra.utils.call(cfg.data.train, batch_size=batch_size)
        val_dataloader = hydra.utils.call(cfg.data.val, batch_size=batch_size)

        # Construct the model
        model = MatchingModel(cfg)
        # model.double()

        # Construct the optimizer and scheduler
        optimizer = hydra.utils.instantiate(cfg.optim.optimizer, model.parameters())
        scheduler = hydra.utils.instantiate(cfg.optim.lr_scheduler, optimizer)

        # # optionally resume from a checkpoint
        start_epoch = 0
        best_metrics = {}
        if cfg.resume:
            filename = 'last.pth'
            assert os.path.isfile(filename), 'Cannot find checkpoint for resuming.'

            log.info("=> loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename, map_location='cpu')
            # checkpoint = torch.load(filename, map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint['model'], strict=False)
            if torch.cuda.is_available():
                model.cuda()

            if cfg.optim.resume:
                log.info("=> loading also optim state from '{}'".format(filename))
                start_epoch = checkpoint['epoch']
                best_metrics = checkpoint['best_metrics']
                # best_rsum = checkpoint['best_rsum']
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])
                # Eiters is used to show logs as the continuation of another
                # training
                # model.Eiters = checkpoint['Eiters']
            log.info("=> loaded checkpoint '{}' (epoch {})"
                    .format(cfg.resume, start_epoch))

        model.train()

        # checkpoint manager
        ckpt_dir = Path(run_dir) / Path('best_models')
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_manager = CheckpointManager(ckpt_dir, current_best=best_metrics)

        # Train loop
        mean_loss = 0
        best_rsum = 0
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        for epoch in tqdm.trange(start_epoch, cfg.optim.epochs):
            progress_bar = tqdm.tqdm(train_dataloader)
            progress_bar.set_description('Train')
            for it, batch in enumerate(progress_bar):
                global_iteration = epoch * len(train_dataloader) + it

                # forward the model
                optimizer.zero_grad()

                # prepare the inputs
                caption, motion, motion_len = batch['desc'], batch['motion'], batch['motion_len']
                # motion = motion.to(device)
                # pose, trajectory, start_trajectory = X
                # # pose_gt, trajectory_gt, start_trajectory_gt = Y

                # x = torch.cat((trajectory, pose), dim=-1).to(device)
                # # y = torch.cat((trajectory_gt, pose_gt), dim=-1).to(device)

                # if isinstance(s2v, torch.Tensor):
                #     s2v = s2v.to(device)

                # # Transform before the model
                # x = pre.transform(x)
                # # y = pre.transform(y)
                # x = x[..., :-4]
                # # y = y[..., :-4]

                loss, monitors = model(motion, motion_len, caption, epoch=epoch)
                loss.backward()

                torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
                mean_loss += loss.item()

                if global_iteration % cfg.optim.log_every == 0:
                    mean_loss /= cfg.optim.log_every
                    progress_bar.set_postfix(dict(loss='{:.2}'.format(mean_loss)))
                    mean_loss = 0

                tb_logger.add_scalar("Training/Epoch", epoch, global_iteration)
                tb_logger.add_scalar("Training/Loss", loss.item(), global_iteration)
                tb_logger.add_scalar("Training/Learning_Rate", optimizer.param_groups[0]['lr'], global_iteration)
                if monitors is not None and len(monitors) > 0:
                    # tb_logger.add_scalars("Training/Monitor Values", monitors, global_iteration)
                    tb_logger.add_scalars("train_metrics", monitors, global_iteration)

                if global_iteration % cfg.optim.val_freq == 0:
                    # validate
                    metrics = validate(val_dataloader, model)
                    for m, v in metrics.items():
                        tb_logger.add_scalar("Validation/{}".format(m), v, global_iteration)
                    # progress_bar.set_postfix(dict(r1='{:.2}'.format(metrics['r1']), r5='{:.2}'.format(metrics['r5']), meanr='{:.2}'.format(metrics['meanr'])))
                    log.info(metrics)

                    # save only if best on some metric (via CheckpointManager)
                    best_metrics = ckpt_manager.save({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch,
                        'metrics': metrics
                    }, metrics, epoch)

                    # save model
                    log.info('Saving model...')
                    checkpoint = {
                        'cfg': cfg,
                        'epoch': epoch,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'best_metrics': best_metrics}
                    torch.save(checkpoint, Path(run_dir) / 'last.pt')

            scheduler.step()

    except Exception as error:
        log.error(f"Training ended due to Runtime Error: {error}. Exiting....")
        traceback.print_exc()
        exit(1)

    log.info("Training ended. Exiting....")



def validate(val_dataloader, model):
    model.eval()

    motion_feats, caption_feats, motion_labels = evaluation.encode_data(model, val_dataloader)
    metrics = evaluation.compute_recall('val', caption_feats, motion_feats, motion_labels, val_dataloader.dataset)

    model.train()
    return metrics

if __name__ == '__main__':
    main()





# # # # ///////////////////////////////

# # # # import os
# # # # import torch
# # # # import torch.nn as nn
# # # # import torch.optim as optim
# # # # from torch.utils.data import DataLoader
# # # # from omegaconf import OmegaConf
# # # # from omegaconf.dictconfig import DictConfig
# # # # from omegaconf.base import ContainerMetadata
# # # # import typing
# # # # import hydra
# # # # from hydra.utils import instantiate

# # # # # ====== Helper: Safe checkpoint load ======
# # # # def load_checkpoint_safe(path, device):
# # # #     checkpoint = None
# # # #     if not os.path.isfile(path):
# # # #         return None
# # # #     try:
# # # #         # Try safe load first
# # # #         with torch.serialization.safe_globals([DictConfig, ContainerMetadata, typing.Any]):
# # # #             checkpoint = torch.load(path, map_location=device)
# # # #         print(f"[INFO] Loaded checkpoint safely from {path}")
# # # #     except Exception as e_safe:
# # # #         print(f"[WARNING] Safe loading failed: {e_safe}")
# # # #         print("[INFO] Trying unsafe load with weights_only=False...")
# # # #         checkpoint = torch.load(path, map_location=device, weights_only=False)
# # # #         print(f"[INFO] Loaded checkpoint in unsafe mode from {path}")
# # # #     return checkpoint

# # # # # ====== Training loop ======
# # # # def train(model, optimizer, train_loader, val_loader, loss_fn, epochs, device, checkpoint_path=None, resume=False):
# # # #     start_epoch = 0
# # # #     if resume and checkpoint_path:
# # # #         checkpoint = load_checkpoint_safe(checkpoint_path, device)
# # # #         if checkpoint:
# # # #             model.load_state_dict(checkpoint['model_state_dict'])
# # # #             optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# # # #             start_epoch = checkpoint.get('epoch', 0)
# # # #             print(f"[INFO] Resuming training from epoch {start_epoch}")

# # # #     model.to(device)
# # # #     for epoch in range(start_epoch, epochs):
# # # #         model.train()
# # # #         for i, batch in enumerate(train_loader):
# # # #             optimizer.zero_grad()
# # # #             x, y = batch  # Adjust depending on your dataset
# # # #             x, y = x.to(device), y.to(device)
# # # #             pred = model(x)
# # # #             loss = loss_fn(pred, y)
# # # #             loss.backward()
# # # #             optimizer.step()
# # # #             if i % 10 == 0:
# # # #                 print(f"Epoch [{epoch}/{epochs}] Step [{i}/{len(train_loader)}] Loss: {loss.item():.4f}")

# # # #         # Optionally: validation step
# # # #         if val_loader is not None:
# # # #             model.eval()
# # # #             val_loss = 0.0
# # # #             with torch.no_grad():
# # # #                 for x_val, y_val in val_loader:
# # # #                     x_val, y_val = x_val.to(device), y_val.to(device)
# # # #                     pred_val = model(x_val)
# # # #                     val_loss += loss_fn(pred_val, y_val).item()
# # # #             val_loss /= len(val_loader)
# # # #             print(f"Epoch [{epoch}/{epochs}] Validation Loss: {val_loss:.4f}")

# # # #         # Save checkpoint
# # # #         checkpoint_dict = {
# # # #             'epoch': epoch + 1,
# # # #             'model_state_dict': model.state_dict(),
# # # #             'optimizer_state_dict': optimizer.state_dict()
# # # #         }
# # # #         torch.save(checkpoint_dict, f"{checkpoint_path}_epoch{epoch+1}.pt")

# # # # # ====== Hydra main entry ======
# # # # @hydra.main(config_path="configs", config_name="config")
# # # # def main(cfg: DictConfig):
# # # #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # # #     print(f"[INFO] Using device: {device}")

# # # #     # Instantiate models, optimizer, datasets
# # # #     model = instantiate(cfg.motion_model.module)
# # # #     text_model = instantiate(cfg.text_model.module)  # if used in pipeline
# # # #     optimizer = instantiate(cfg.optim.optimizer, params=model.parameters())
# # # #     loss_fn = instantiate(cfg.optim.loss)

# # # #     train_loader = instantiate(cfg.data.train)
# # # #     val_loader = instantiate(cfg.data.val)

# # # #     checkpoint_path = os.path.join(os.getcwd(), "last.pt")
# # # #     resume = cfg.get("resume", False)

# # # #     train(model, optimizer, train_loader, val_loader, loss_fn,
# # # #           epochs=cfg.optim.epochs, device=device,
# # # #           checkpoint_path=checkpoint_path, resume=resume)

# # # # if __name__ == "__main__":
# # # #     main()

# # # import os
# # # import sys
# # # import torch
# # # import hydra
# # # from omegaconf import DictConfig, OmegaConf
# # # from torch.utils.data import DataLoader
# # # from transformers import BertModel
# # # from data_loaders.get_data import get_dataset_loader  # Adjust your import path
# # # from models.motions import UpperLowerGRU
# # # from models.texts import BERTSentenceEncoderLSTM
# # # from models.losses import InfoNCELoss

# # # # GPU/CPU device
# # # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # # print(f"Using device: {device}")

# # # # Utility function to get absolute dataset path
# # # def get_abs_dataset_path(filename):
# # #     base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "dataset"))
# # #     return os.path.join(base_dir, filename)

# # # @hydra.main(config_path="configs", config_name="config", version_base=None)
# # # def main(cfg: DictConfig):

# # #     print("Hydra config:")
# # #     print(OmegaConf.to_yaml(cfg))

# # #     # --- Initialize Motion Model ---
# # #     motion_model_cfg = cfg.motion_model.module
# # #     motion_model = UpperLowerGRU(
# # #         h1=motion_model_cfg.h1,
# # #         h2=motion_model_cfg.h2,
# # #         h3=motion_model_cfg.h3
# # #     ).to(device)
# # #     print("Motion model initialized.")

# # #     # --- Initialize Text Model (BERT+LSTM) ---
# # #     text_model_cfg = cfg.text_model.module
# # #     bert_model = BertModel.from_pretrained("bert-large-cased")
# # #     text_model = BERTSentenceEncoderLSTM(
# # #         bert_model=bert_model,
# # #         hidden_size=text_model_cfg.hidden_size,
# # #         lstm_layers=text_model_cfg.lstm_layers
# # #     ).to(device)
# # #     print("Text model initialized.")

# # #     # --- Load Dataset ---
# # #     abs_dataset_path = get_abs_dataset_path("kit_opt.txt")
# # #     if not os.path.exists(abs_dataset_path):
# # #         raise FileNotFoundError(f"Dataset file not found: {abs_dataset_path}")

# # #     train_loader = get_dataset_loader(
# # #         name="kit",
# # #         num_frames=cfg.data.train.num_frames,
# # #         split="train",
# # #         hml_mode="train",
# # #         max_violation_after=cfg.data.train.max_violation_after,
# # #         batch_size=cfg.optim.batch_size
# # #     )

# # #     val_loader = get_dataset_loader(
# # #         name="kit",
# # #         num_frames=cfg.data.val.num_frames,
# # #         split="val",
# # #         hml_mode="eval",
# # #         max_violation_after=cfg.data.val.max_violation_after,
# # #         batch_size=cfg.optim.batch_size
# # #     )

# # #     print("Datasets loaded.")

# # #     # --- Optimizer and Scheduler ---
# # #     optimizer_cfg = cfg.optim
# # #     optimizer = torch.optim.Adam(
# # #         list(motion_model.parameters()) + list(text_model.parameters()),
# # #         lr=optimizer_cfg.optimizer.lr
# # #     )
# # #     scheduler = torch.optim.lr_scheduler.MultiStepLR(
# # #         optimizer, milestones=optimizer_cfg.lr_scheduler.milestones, gamma=optimizer_cfg.lr_scheduler.gamma
# # #     )

# # #     # --- Loss ---
# # #     criterion = InfoNCELoss().to(device)
# # #     print("Optimizer, scheduler, and loss initialized.")

# # #     # --- Resume checkpoint if exists ---
# # #     if cfg.resume:
# # #         checkpoint_path = os.path.join(cfg.hydra.run.dir, "last.pt")
# # #         if os.path.exists(checkpoint_path):
# # #             checkpoint = torch.load(checkpoint_path, map_location=device)
# # #             motion_model.load_state_dict(checkpoint["motion_model_state"])
# # #             text_model.load_state_dict(checkpoint["text_model_state"])
# # #             optimizer.load_state_dict(checkpoint["optimizer_state"])
# # #             print(f"Resumed from checkpoint: {checkpoint_path}")
# # #         else:
# # #             print(f"No checkpoint found at {checkpoint_path}, starting fresh.")

# # #     # --- Training loop ---
# # #     epochs = optimizer_cfg.epochs
# # #     for epoch in range(epochs):
# # #         motion_model.train()
# # #         text_model.train()

# # #         for batch_idx, batch in enumerate(train_loader):
# # #             # Move batch to device
# # #             for key in batch:
# # #                 if torch.is_tensor(batch[key]):
# # #                     batch[key] = batch[key].to(device)

# # #             optimizer.zero_grad()

# # #             # Forward pass
# # #             motion_feats = motion_model(batch["motion"])
# # #             text_feats = text_model(batch["text_input_ids"], batch["text_attention_mask"])

# # #             # Compute loss
# # #             loss = criterion(motion_feats, text_feats)
# # #             loss.backward()
# # #             optimizer.step()

# # #             if batch_idx % optimizer_cfg.log_every == 0:
# # #                 print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

# # #         scheduler.step()
# # #         print(f"Epoch [{epoch+1}/{epochs}] completed.")

# # #     # --- Save checkpoint ---
# # #     save_path = os.path.join(cfg.hydra.run.dir, "last.pt")
# # #     torch.save({
# # #         "motion_model_state": motion_model.state_dict(),
# # #         "text_model_state": text_model.state_dict(),
# # #         "optimizer_state": optimizer.state_dict(),
# # #     }, save_path)
# # #     print(f"Checkpoint saved at: {save_path}")


# # # if __name__ == "__main__":
# # #     main()
# # # train.py
# # import os
# # import sys
# # import random
# # import torch
# # import hydra
# # from omegaconf import DictConfig, OmegaConf
# # import numpy as np

# # # -----------------------------
# # # Ensure project root is in sys.path
# # # -----------------------------
# # project_root = os.path.dirname(os.path.abspath(__file__))
# # if project_root not in sys.path:
# #     sys.path.append(project_root)

# # # -----------------------------
# # # Set random seeds for reproducibility
# # # -----------------------------
# # def set_seed(seed: int = 42):
# #     random.seed(seed)
# #     np.random.seed(seed)
# #     torch.manual_seed(seed)
# #     torch.cuda.manual_seed_all(seed)
# #     torch.backends.cudnn.deterministic = True
# #     torch.backends.cudnn.benchmark = False

# # # -----------------------------
# # # Main training function
# # # -----------------------------
# # @hydra.main(config_path="configs", config_name="config", version_base=None)
# # def main(cfg: DictConfig):

# #     # Print Hydra config
# #     print("Hydra config:")
# #     print(OmegaConf.to_yaml(cfg))

# #     # -----------------------------
# #     # Device
# #     # -----------------------------
# #     device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() and cfg.gpu >= 0 else "cpu")
# #     print(f"Using device: {device}")

# #     # -----------------------------
# #     # Set seeds
# #     # -----------------------------
# #     set_seed(cfg.optim.seed)

# #     # -----------------------------
# #     # Import models dynamically
# #     # -----------------------------
# #     from hydra.utils import instantiate

# #     motion_model = instantiate(cfg.motion_model.module).to(device)
# #     print("Motion model initialized.")

# #     text_model = instantiate(cfg.text_model.module).to(device)
# #     print("Text model initialized.")

# #     # -----------------------------
# #     # Load datasets
# #     # -----------------------------
# #     train_loader = instantiate(cfg.data.train)
# #     val_loader = instantiate(cfg.data.val)
# #     test_loader = instantiate(cfg.data.test)
# #     print("Data loaders initialized.")

# #     # -----------------------------
# #     # Optimizer and scheduler
# #     # -----------------------------
# #     optimizer = instantiate(cfg.optim.optimizer, params=list(motion_model.parameters()) + list(text_model.parameters()))
# #     scheduler = instantiate(cfg.optim.lr_scheduler, optimizer=optimizer)

# #     # -----------------------------
# #     # Loss function
# #     # -----------------------------
# #     loss_fn = instantiate(cfg.optim.loss)

# #     # -----------------------------
# #     # Training loop
# #     # -----------------------------
# #     for epoch in range(cfg.optim.epochs):
# #         motion_model.train()
# #         text_model.train()
# #         running_loss = 0.0

# #         for i, batch in enumerate(train_loader):
# #             optimizer.zero_grad()

# #             motion_input = batch['motion'].to(device)
# #             text_input = batch['text'].to(device)

# #             motion_feat = motion_model(motion_input)
# #             text_feat = text_model(text_input)

# #             loss = loss_fn(motion_feat, text_feat)
# #             loss.backward()
# #             optimizer.step()

# #             running_loss += loss.item()
# #             if (i + 1) % cfg.optim.log_every == 0:
# #                 print(f"[Epoch {epoch+1}/{cfg.optim.epochs}] Step {i+1} Loss: {running_loss / cfg.optim.log_every:.4f}")
# #                 running_loss = 0.0

# #         scheduler.step()

# #         # Validation
# #         if (epoch + 1) % cfg.optim.val_freq == 0:
# #             motion_model.eval()
# #             text_model.eval()
# #             val_loss = 0.0
# #             with torch.no_grad():
# #                 for batch in val_loader:
# #                     motion_input = batch['motion'].to(device)
# #                     text_input = batch['text'].to(device)

# #                     motion_feat = motion_model(motion_input)
# #                     text_feat = text_model(text_input)

# #                     val_loss += loss_fn(motion_feat, text_feat).item()
# #             val_loss /= len(val_loader)
# #             print(f"Validation Loss after epoch {epoch+1}: {val_loss:.4f}")

# #     print("Training complete.")


# # if __name__ == "__main__":
# #     main()
# # import os
# # import sys
# # import random
# # import torch
# # import hydra
# # from omegaconf import DictConfig, OmegaConf
# # import numpy as np
# # import traceback
# # import logging
# # from pathlib import Path
# # from torch.utils.tensorboard import SummaryWriter
# # from transformers import BertTokenizer
# # import os
# # from pathlib import Path
# # import tqdm
# # import hydra
# # from hydra.core.hydra_config import HydraConfig
# # import yaml
# # import traceback
# # import torch
# # import logging
# # from torch.utils.tensorboard import SummaryWriter

# # import argparse
# # import utils
# # # from data import *

# # from models.model import MatchingModel

# # from shutil import copyfile
# # from ast import literal_eval
# # import evaluation
# # from utils.checkpoint import CheckpointManager


# # # Ensure project root is in sys.path
# # project_root = os.path.dirname(os.path.abspath(__file__))
# # if project_root not in sys.path:
# #     sys.path.append(project_root)

# # log = logging.getLogger(__name__)

# # # -----------------------------
# # # Set random seeds for reproducibility
# # # -----------------------------
# # def set_seed(seed: int = 42):
# #     random.seed(seed)
# #     np.random.seed(seed)
# #     torch.manual_seed(seed)
# #     torch.cuda.manual_seed_all(seed)
# #     torch.backends.cudnn.deterministic = True
# #     torch.backends.cudnn.benchmark = False

# # # -----------------------------
# # # Main training function
# # # -----------------------------
# # @hydra.main(config_path="configs", config_name="config", version_base=None)
# # def main(cfg: DictConfig):
# #     try:
# #         print("Hydra config:")
# #         print(OmegaConf.to_yaml(cfg))
# #         log.info(f"Run path: {Path.cwd()}")

# #         # -----------------------------
# #         # Device setup
# #         # -----------------------------
# #         device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() and cfg.gpu >= 0 else "cpu")
# #         print(f"Using device: {device}")

# #         # -----------------------------
# #         # Set seeds
# #         # -----------------------------
# #         set_seed(cfg.optim.seed)

# #         # -----------------------------
# #         # Tensorboard logger
# #         # -----------------------------
# #         run_dir = os.getcwd()  # Hydra changes cwd
# #         tb_logger = SummaryWriter(run_dir)

# #         # -----------------------------
# #         # Import and initialize models
# #         # -----------------------------
# #         from hydra.utils import instantiate

# #         motion_model = instantiate(cfg.motion_model.module).to(device)
# #         print("Motion model initialized.")

# #         text_model = instantiate(cfg.text_model.module).to(device)
# #         print("Text model initialized.")

# #         # -----------------------------
# #         # Load datasets
# #         # -----------------------------
# #         train_loader = instantiate(cfg.data.train)
# #         val_loader = instantiate(cfg.data.val)
# #         test_loader = instantiate(cfg.data.test)
# #         print("Data loaders initialized.")

# #         # -----------------------------
# #         # Optimizer and scheduler
# #         # -----------------------------
# #         optimizer = instantiate(cfg.optim.optimizer, params=list(motion_model.parameters()) + list(text_model.parameters()))
# #         scheduler = instantiate(cfg.optim.lr_scheduler, optimizer=optimizer)

# #         # -----------------------------
# #         # Loss function
# #         # -----------------------------
# #         loss_fn = instantiate(cfg.optim.loss)

# #         # -----------------------------
# #         # Resume from checkpoint if needed
# #         # -----------------------------
# #         start_epoch = 0
# #         if cfg.resume:
# #             last_checkpoint = Path(run_dir) / "last.pt"
# #             if last_checkpoint.is_file():
# #                 print(f"Resuming from checkpoint: {last_checkpoint}")
# #                 # Safely allow DictConfig unpickling
# #                 from omegaconf import DictConfig
# #                 with torch.serialization.safe_globals([DictConfig]):
# #                     checkpoint = torch.load(last_checkpoint, map_location="cpu")
# #                 motion_model.load_state_dict(checkpoint['model'], strict=False)
# #                 optimizer.load_state_dict(checkpoint['optimizer'])
# #                 scheduler.load_state_dict(checkpoint['scheduler'])
# #                 start_epoch = checkpoint['epoch']
# #                 print(f"Resumed from epoch {start_epoch}")

# #         # -----------------------------
# #         # Training loop
# #         # -----------------------------
# #         for epoch in range(start_epoch, cfg.optim.epochs):
# #             motion_model.train()
# #             text_model.train()
# #             running_loss = 0.0

# #             for i, batch in enumerate(train_loader):
# #                 optimizer.zero_grad()

# #                 motion_input = batch['motion'].to(device)
# #                 text_input = batch['text'].to(device)
# #                 text_list = [torch.tensor(x, dtype=torch.long) for x in batch['desc']]
# #                 text_input = pad_sequence(text_list, batch_first=True).to(device)
# #                 motion_feat = motion_model(motion_input)
# #                 text_feat = text_model(text_input)

# #                 loss = loss_fn(motion_feat, text_feat)
# #                 loss.backward()

# #                 torch.nn.utils.clip_grad.clip_grad_norm_(list(motion_model.parameters()) + list(text_model.parameters()), 2.0)
# #                 optimizer.step()

# #                 running_loss += loss.item()
# #                 if (i + 1) % cfg.optim.log_every == 0:
# #                     print(f"[Epoch {epoch+1}/{cfg.optim.epochs}] Step {i+1} Loss: {running_loss / cfg.optim.log_every:.4f}")
# #                     running_loss = 0.0

# #             scheduler.step()

# #             # -----------------------------
# #             # Validation
# #             # -----------------------------
# #             if (epoch + 1) % cfg.optim.val_freq == 0:
# #                 motion_model.eval()
# #                 text_model.eval()
# #                 val_loss = 0.0
# #                 with torch.no_grad():
# #                     for batch in val_loader:
# #                         motion_input = batch['motion'].to(device)
# #                         text_input = batch['text'].to(device)

# #                         motion_feat = motion_model(motion_input)
# #                         text_feat = text_model(text_input)

# #                         val_loss += loss_fn(motion_feat, text_feat).item()
# #                 val_loss /= len(val_loader)
# #                 print(f"Validation Loss after epoch {epoch+1}: {val_loss:.4f}")

# #                 # Save checkpoint
# #                 checkpoint = {
# #                     'epoch': epoch,
# #                     'model': motion_model.state_dict(),
# #                     'optimizer': optimizer.state_dict(),
# #                     'scheduler': scheduler.state_dict(),
# #                 }
# #                 torch.save(checkpoint, Path(run_dir) / "last.pt")

# #         print("Training complete.")

# #     except Exception as error:
# #         log.error(f"Training ended due to Runtime Error: {error}")
# #         traceback.print_exc()
# #         exit(1)

# # if __name__ == "__main__":
# #     main()
# # import os
# # import sys
# # import random
# # import torch
# # import hydra
# # from omegaconf import DictConfig, OmegaConf
# # import numpy as np
# # import traceback
# # import logging
# # from pathlib import Path
# # from torch.utils.tensorboard import SummaryWriter
# # from transformers import BertTokenizer
# # from torch.nn.utils.rnn import pad_sequence

# # # Ensure project root is in sys.path
# # project_root = os.path.dirname(os.path.abspath(__file__))
# # if project_root not in sys.path:
# #     sys.path.append(project_root)

# # log = logging.getLogger(__name__)

# # # -----------------------------
# # # Set random seeds for reproducibility
# # # -----------------------------
# # def set_seed(seed: int = 42):
# #     random.seed(seed)
# #     np.random.seed(seed)
# #     torch.manual_seed(seed)
# #     torch.cuda.manual_seed_all(seed)
# #     torch.backends.cudnn.deterministic = True
# #     torch.backends.cudnn.benchmark = False

# # # -----------------------------
# # # Main training function
# # # -----------------------------
# # @hydra.main(config_path="configs", config_name="config", version_base=None)
# # def main(cfg: DictConfig):
# #     try:
# #         print("Hydra config:")
# #         print(OmegaConf.to_yaml(cfg))
# #         log.info(f"Run path: {Path.cwd()}")

# #         # -----------------------------
# #         # Device setup
# #         # -----------------------------
# #         device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() and cfg.gpu >= 0 else "cpu")
# #         print(f"Using device: {device}")

# #         # -----------------------------
# #         # Set seeds
# #         # -----------------------------
# #         set_seed(cfg.optim.seed)

# #         # -----------------------------
# #         # Tensorboard logger
# #         # -----------------------------
# #         run_dir = os.getcwd()  # Hydra changes cwd
# #         tb_logger = SummaryWriter(run_dir)

# #         # -----------------------------
# #         # Initialize tokenizer
# #         # -----------------------------
# #         tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# #         # -----------------------------
# #         # Import and initialize models
# #         # -----------------------------
# #         from hydra.utils import instantiate

# #         motion_model = instantiate(cfg.motion_model.module).to(device)
# #         print("Motion model initialized.")

# #         text_model = instantiate(cfg.text_model.module).to(device)
# #         print("Text model initialized.")

# #         # -----------------------------
# #         # Load datasets
# #         # -----------------------------
# #         train_loader = instantiate(cfg.data.train)
# #         val_loader = instantiate(cfg.data.val)
# #         test_loader = instantiate(cfg.data.test)
# #         print("Data loaders initialized.")

# #         # -----------------------------
# #         # Optimizer and scheduler
# #         # -----------------------------
# #         optimizer = instantiate(cfg.optim.optimizer, params=list(motion_model.parameters()) + list(text_model.parameters()))
# #         scheduler = instantiate(cfg.optim.lr_scheduler, optimizer=optimizer)

# #         # -----------------------------
# #         # Loss function
# #         # -----------------------------
# #         loss_fn = instantiate(cfg.optim.loss)

# #         # -----------------------------
# #         # Resume from checkpoint if needed
# #         # -----------------------------
# #         start_epoch = 0
# #         if cfg.resume:
# #             last_checkpoint = Path(run_dir) / "last.pt"
# #             if last_checkpoint.is_file():
# #                 print(f"Resuming from checkpoint: {last_checkpoint}")
# #                 from omegaconf import DictConfig
# #                 with torch.serialization.safe_globals([DictConfig]):
# #                     checkpoint = torch.load(last_checkpoint, map_location="cpu")
# #                 motion_model.load_state_dict(checkpoint['model'], strict=False)
# #                 optimizer.load_state_dict(checkpoint['optimizer'])
# #                 scheduler.load_state_dict(checkpoint['scheduler'])
# #                 start_epoch = checkpoint['epoch']
# #                 print(f"Resumed from epoch {start_epoch}")

# #         # -----------------------------
# #         # Training loop
# #         # -----------------------------
# #         for epoch in range(start_epoch, cfg.optim.epochs):
# #             motion_model.train()
# #             text_model.train()
# #             running_loss = 0.0

# #             for i, batch in enumerate(train_loader):
# #                 optimizer.zero_grad()

# #                 # Move motion to device
# #                 motion_input = batch['motion'].to(device)

# #                 # Tokenize text batch
# #                 text_strings = batch['desc']  # list of strings
# #                 encoded = tokenizer(
# #                     text_strings,
# #                     padding=True,
# #                     truncation=True,
# #                     max_length=cfg.data.num_frames,  # or suitable max length
# #                     return_tensors='pt'
# #                 )
# #                 input_ids = encoded['input_ids'].to(device)
# #                 attention_mask = encoded['attention_mask'].to(device)

# #                 # Forward pass
# #                 motion_feat = motion_model(motion_input)
# #                 text_feat = text_model(input_ids=input_ids, attention_mask=attention_mask)

# #                 loss = loss_fn(motion_feat, text_feat)
# #                 loss.backward()

# #                 torch.nn.utils.clip_grad.clip_grad_norm_(list(motion_model.parameters()) + list(text_model.parameters()), 2.0)
# #                 optimizer.step()

# #                 running_loss += loss.item()
# #                 if (i + 1) % cfg.optim.log_every == 0:
# #                     print(f"[Epoch {epoch+1}/{cfg.optim.epochs}] Step {i+1} Loss: {running_loss / cfg.optim.log_every:.4f}")
# #                     running_loss = 0.0

# #                 # Tensorboard logging
# #                 global_step = epoch * len(train_loader) + i
# #                 tb_logger.add_scalar("Training/Loss", loss.item(), global_step)
# #                 tb_logger.add_scalar("Training/LR", optimizer.param_groups[0]['lr'], global_step)

# #             scheduler.step()

# #             # -----------------------------
# #             # Validation
# #             # -----------------------------
# #             if (epoch + 1) % cfg.optim.val_freq == 0:
# #                 motion_model.eval()
# #                 text_model.eval()
# #                 val_loss = 0.0
# #                 with torch.no_grad():
# #                     for batch in val_loader:
# #                         motion_input = batch['motion'].to(device)

# #                         text_strings = batch['desc']
# #                         encoded = tokenizer(
# #                             text_strings,
# #                             padding=True,
# #                             truncation=True,
# #                             max_length=cfg.data.num_frames,
# #                             return_tensors='pt'
# #                         )
# #                         input_ids = encoded['input_ids'].to(device)
# #                         attention_mask = encoded['attention_mask'].to(device)

# #                         motion_feat = motion_model(motion_input)
# #                         text_feat = text_model(input_ids=input_ids, attention_mask=attention_mask)

# #                         val_loss += loss_fn(motion_feat, text_feat).item()
# #                 val_loss /= len(val_loader)
# #                 print(f"Validation Loss after epoch {epoch+1}: {val_loss:.4f}")

# #                 # Save checkpoint
# #                 checkpoint = {
# #                     'epoch': epoch,
# #                     'model': motion_model.state_dict(),
# #                     'optimizer': optimizer.state_dict(),
# #                     'scheduler': scheduler.state_dict(),
# #                 }
# #                 torch.save(checkpoint, Path(run_dir) / "last.pt")

# #         print("Training complete.")

# #     except Exception as error:
# #         log.error(f"Training ended due to Runtime Error: {error}")
# #         traceback.print_exc()
# #         exit(1)


# # if __name__ == "__main__":
# #     main()
# # import os
# # import sys
# # import random
# # import torch
# # import hydra
# # from omegaconf import DictConfig, OmegaConf
# # import numpy as np
# # import traceback
# # import logging
# # from pathlib import Path
# # from torch.utils.tensorboard import SummaryWriter
# # from torch.nn.utils.rnn import pad_sequence
# # from transformers import BertTokenizer
# # from hydra.utils import instantiate

# # # -----------------------------
# # # Ensure project root is in sys.path
# # # -----------------------------
# # project_root = os.path.dirname(os.path.abspath(__file__))
# # if project_root not in sys.path:
# #     sys.path.append(project_root)

# # log = logging.getLogger(__name__)

# # # -----------------------------
# # # Set random seeds for reproducibility
# # # -----------------------------
# # def set_seed(seed: int = 42):
# #     random.seed(seed)
# #     np.random.seed(seed)
# #     torch.manual_seed(seed)
# #     torch.cuda.manual_seed_all(seed)
# #     torch.backends.cudnn.deterministic = True
# #     torch.backends.cudnn.benchmark = False

# # # -----------------------------
# # # Tokenizer (shared for text model)
# # # -----------------------------
# # tokenizer = BertTokenizer.from_pretrained("bert-large-cased")

# # # -----------------------------
# # # Main training function
# # # -----------------------------
# # @hydra.main(config_path="configs", config_name="config", version_base=None)
# # def main(cfg: DictConfig):
# #     try:
# #         print("Hydra config:")
# #         print(OmegaConf.to_yaml(cfg))
# #         log.info(f"Run path: {Path.cwd()}")

# #         # -----------------------------
# #         # Device setup
# #         # -----------------------------
# #         device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() and cfg.gpu >= 0 else "cpu")
# #         print(f"Using device: {device}")

# #         # -----------------------------
# #         # Set seeds
# #         # -----------------------------
# #         set_seed(cfg.optim.seed)

# #         # -----------------------------
# #         # Tensorboard logger
# #         # -----------------------------
# #         run_dir = Path.cwd()  # Hydra changes cwd
# #         tb_logger = SummaryWriter(run_dir)

# #         # -----------------------------
# #         # Instantiate models
# #         # -----------------------------
# #         motion_model = instantiate(cfg.motion_model.module).to(device)
# #         print("Motion model initialized.")

# #         text_model = instantiate(cfg.text_model.module).to(device)
# #         print("Text model initialized.")

# #         # -----------------------------
# #         # Load datasets
# #         # -----------------------------
# #         train_loader = instantiate(cfg.data.train)
# #         val_loader = instantiate(cfg.data.val)
# #         test_loader = instantiate(cfg.data.test)
# #         print("Data loaders initialized.")

# #         # -----------------------------
# #         # Optimizer, scheduler, loss
# #         # -----------------------------
# #         optimizer = instantiate(cfg.optim.optimizer, params=list(motion_model.parameters()) + list(text_model.parameters()))
# #         scheduler = instantiate(cfg.optim.lr_scheduler, optimizer=optimizer)
# #         loss_fn = instantiate(cfg.optim.loss)

# #         # -----------------------------
# #         # Resume from checkpoint
# #         # -----------------------------
# #         start_epoch = 0
# #         if cfg.resume:
# #             last_checkpoint = run_dir / "last.pt"
# #             if last_checkpoint.is_file():
# #                 print(f"Resuming from checkpoint: {last_checkpoint}")
# #                 checkpoint = torch.load(last_checkpoint, map_location=device)
# #                 motion_model.load_state_dict(checkpoint['model'], strict=False)
# #                 optimizer.load_state_dict(checkpoint['optimizer'])
# #                 scheduler.load_state_dict(checkpoint['scheduler'])
# #                 start_epoch = checkpoint['epoch']
# #                 print(f"Resumed from epoch {start_epoch}")

# #         # -----------------------------
# #         # Training loop
# #         # -----------------------------
# #         for epoch in range(start_epoch, cfg.optim.epochs):
# #             motion_model.train()
# #             text_model.train()
# #             running_loss = 0.0

# #             for i, batch in enumerate(train_loader):
# #                 optimizer.zero_grad()

# #                 # Motion input
# #                 motion_input = batch['motion'].to(device)

# #                 # Text input: convert list of strings to token IDs
# #                 text_list = [tokenizer.encode(x, add_special_tokens=True, truncation=True,
# #                                               max_length=cfg.data.train.num_frames) for x in batch['desc']]
# #                 text_tensor = pad_sequence([torch.tensor(x, dtype=torch.long) for x in text_list],
# #                                            batch_first=True).to(device)

# #                 # Forward pass
# #                 motion_feat = motion_model(motion_input)
# #                 text_feat = text_model(text_tensor)

# #                 # Loss
# #                 loss = loss_fn(motion_feat, text_feat)
# #                 loss.backward()
# #                 torch.nn.utils.clip_grad.clip_grad_norm_(list(motion_model.parameters()) + list(text_model.parameters()), 2.0)
# #                 optimizer.step()

# #                 running_loss += loss.item()
# #                 if (i + 1) % cfg.optim.log_every == 0:
# #                     avg_loss = running_loss / cfg.optim.log_every
# #                     print(f"[Epoch {epoch+1}/{cfg.optim.epochs}] Step {i+1} Loss: {avg_loss:.4f}")
# #                     tb_logger.add_scalar("train/loss", avg_loss, epoch * len(train_loader) + i)
# #                     running_loss = 0.0

# #             scheduler.step()

# #             # -----------------------------
# #             # Validation
# #             # -----------------------------
# #             if (epoch + 1) % cfg.optim.val_freq == 0:
# #                 motion_model.eval()
# #                 text_model.eval()
# #                 val_loss = 0.0
# #                 with torch.no_grad():
# #                     for batch in val_loader:
# #                         motion_input = batch['motion'].to(device)
# #                         text_list = [tokenizer.encode(x, add_special_tokens=True, truncation=True,
# #                                                       max_length=cfg.data.val.num_frames) for x in batch['desc']]
# #                         text_tensor = pad_sequence([torch.tensor(x, dtype=torch.long) for x in text_list],
# #                                                    batch_first=True).to(device)

# #                         motion_feat = motion_model(motion_input)
# #                         text_feat = text_model(text_tensor)

# #                         val_loss += loss_fn(motion_feat, text_feat).item()
# #                 val_loss /= len(val_loader)
# #                 print(f"Validation Loss after epoch {epoch+1}: {val_loss:.4f}")
# #                 tb_logger.add_scalar("val/loss", val_loss, epoch)

# #                 # Save checkpoint
# #                 checkpoint = {
# #                     'epoch': epoch,
# #                     'model': motion_model.state_dict(),
# #                     'optimizer': optimizer.state_dict(),
# #                     'scheduler': scheduler.state_dict(),
# #                 }
# #                 torch.save(checkpoint, run_dir / "last.pt")

# #         print("Training complete.")

# #     except Exception as error:
# #         log.error(f"Training ended due to Runtime Error: {error}")
# #         traceback.print_exc()
# #         exit(1)


# # if __name__ == "__main__":
# #     main()
# import os
# import sys
# import random
# import torch
# import hydra
# from omegaconf import DictConfig, OmegaConf
# import numpy as np
# import traceback
# import logging
# from pathlib import Path
# from torch.utils.tensorboard import SummaryWriter
# from torch.nn.utils.rnn import pad_sequence
# from transformers import BertTokenizer

# # Ensure project root is in sys.path
# project_root = os.path.dirname(os.path.abspath(__file__))
# if project_root not in sys.path:
#     sys.path.append(project_root)

# log = logging.getLogger(__name__)

# # -----------------------------
# # Set random seeds for reproducibility
# # -----------------------------
# def set_seed(seed: int = 42):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# # -----------------------------
# # Main training function
# # -----------------------------
# @hydra.main(config_path="configs", config_name="config", version_base=None)
# def main(cfg: DictConfig):
#     try:
#         print("Hydra config:")
#         print(OmegaConf.to_yaml(cfg))
#         log.info(f"Run path: {Path.cwd()}")

#         # -----------------------------
#         # Device setup
#         # -----------------------------
#         device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() and cfg.gpu >= 0 else "cpu")
#         print(f"Using device: {device}")

#         # -----------------------------
#         # Set seeds
#         # -----------------------------
#         set_seed(cfg.optim.seed)

#         # -----------------------------
#         # Tensorboard logger
#         # -----------------------------
#         run_dir = os.getcwd()  # Hydra changes cwd
#         tb_logger = SummaryWriter(run_dir)

#         # -----------------------------
#         # Import and initialize models
#         # -----------------------------
#         from hydra.utils import instantiate

#         motion_model = instantiate(cfg.motion_model.module).to(device)
#         print("Motion model initialized.")

#         text_model = instantiate(cfg.text_model.module).to(device)
#         print("Text model initialized.")

#         # -----------------------------
#         # Tokenizer for text
#         # -----------------------------
#         tokenizer = BertTokenizer.from_pretrained("bert-large-cased")

#         # -----------------------------
#         # Load datasets
#         # -----------------------------
#         train_loader = instantiate(cfg.data.train)
#         val_loader = instantiate(cfg.data.val)
#         test_loader = instantiate(cfg.data.test)
#         print("Data loaders initialized.")

#         # -----------------------------
#         # Optimizer and scheduler
#         # -----------------------------
#         optimizer = instantiate(
#             cfg.optim.optimizer, 
#             params=list(motion_model.parameters()) + list(text_model.parameters())
#         )
#         scheduler = instantiate(cfg.optim.lr_scheduler, optimizer=optimizer)

#         # -----------------------------
#         # Loss function
#         # -----------------------------
#         loss_fn = instantiate(cfg.optim.loss)

#         # -----------------------------
#         # Resume from checkpoint if needed
#         # -----------------------------
#         start_epoch = 0
#         if cfg.resume:
#             last_checkpoint = Path(run_dir) / "last.pt"
#             if last_checkpoint.is_file():
#                 print(f"Resuming from checkpoint: {last_checkpoint}")
#                 checkpoint = torch.load(last_checkpoint, map_location="cpu")
#                 motion_model.load_state_dict(checkpoint['model'], strict=False)
#                 optimizer.load_state_dict(checkpoint['optimizer'])
#                 scheduler.load_state_dict(checkpoint['scheduler'])
#                 start_epoch = checkpoint['epoch']
#                 print(f"Resumed from epoch {start_epoch}")

#         # -----------------------------
#         # Training loop
#         # -----------------------------
#         for epoch in range(start_epoch, cfg.optim.epochs):
#             motion_model.train()
#             text_model.train()
#             running_loss = 0.0

#             for i, batch in enumerate(train_loader):
#                 optimizer.zero_grad()

#                 # --- Motion input ---
#                 motion_input = batch['motion'].to(device)  # [B, T, ...]
#                 lengths = batch.get('lengths', torch.tensor([motion_input.shape[1]]*motion_input.shape[0]))
#                 lengths = lengths.to(device, dtype=torch.long)

#                 # --- Text input ---
#                 desc_list = batch['desc']  # List of strings
#                 encoded = tokenizer(desc_list, return_tensors='pt', padding=True, truncation=True, max_length=cfg.data.train.num_frames)
#                 text_input = encoded['input_ids'].to(device)
#                 attention_mask = encoded['attention_mask'].to(device)

#                 # --- Forward pass ---
#                 motion_feat = motion_model(motion_input, lengths)  # pass lengths for RNN/GRU
#                 text_feat = text_model(text_input, attention_mask)

#                 # --- Loss and backward ---
#                 loss = loss_fn(motion_feat, text_feat)
#                 loss.backward()
#                 torch.nn.utils.clip_grad.clip_grad_norm_(
#                     list(motion_model.parameters()) + list(text_model.parameters()), 2.0
#                 )
#                 optimizer.step()

#                 running_loss += loss.item()
#                 if (i + 1) % cfg.optim.log_every == 0:
#                     print(f"[Epoch {epoch+1}/{cfg.optim.epochs}] Step {i+1} Loss: {running_loss / cfg.optim.log_every:.4f}")
#                     running_loss = 0.0

#             scheduler.step()

#             # -----------------------------
#             # Validation
#             # -----------------------------
#             if (epoch + 1) % cfg.optim.val_freq == 0:
#                 motion_model.eval()
#                 text_model.eval()
#                 val_loss = 0.0
#                 with torch.no_grad():
#                     for batch in val_loader:
#                         motion_input = batch['motion'].to(device)
#                         lengths = batch.get('lengths', torch.tensor([motion_input.shape[1]]*motion_input.shape[0]))
#                         lengths = lengths.to(device, dtype=torch.long)

#                         desc_list = batch['desc']
#                         encoded = tokenizer(desc_list, return_tensors='pt', padding=True, truncation=True, max_length=cfg.data.val.num_frames)
#                         text_input = encoded['input_ids'].to(device)
#                         attention_mask = encoded['attention_mask'].to(device)

#                         motion_feat = motion_model(motion_input, lengths)
#                         text_feat = text_model(text_input, attention_mask)

#                         val_loss += loss_fn(motion_feat, text_feat).item()
#                 val_loss /= len(val_loader)
#                 print(f"Validation Loss after epoch {epoch+1}: {val_loss:.4f}")

#                 # Save checkpoint
#                 checkpoint = {
#                     'epoch': epoch,
#                     'model': motion_model.state_dict(),
#                     'optimizer': optimizer.state_dict(),
#                     'scheduler': scheduler.state_dict(),
#                 }
#                 torch.save(checkpoint, Path(run_dir) / "last.pt")

#         print("Training complete.")

#     except Exception as error:
#         log.error(f"Training ended due to Runtime Error: {error}")
#         traceback.print_exc()
#         exit(1)

# if __name__ == "__main__":
#     main()
