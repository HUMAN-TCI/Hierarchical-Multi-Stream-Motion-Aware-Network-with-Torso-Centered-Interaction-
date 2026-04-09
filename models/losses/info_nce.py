# import torch
# from torch import nn as nn
# from torch.nn import functional as F
# import numpy as np

# class InfoNCELoss(nn.Module):
#     def __init__(self, **kwargs):
#         super().__init__()
#         self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

#     def forward(self, comp_data, epoch, return_similarity_mat=False, **kwargs):
#         im, s = comp_data['motion_emb'], comp_data['text_emb']

#         # cosine similarity as logits
#         logit_scale = self.logit_scale.exp()
#         logits_per_image = logit_scale * im @ s.t()
#         logits_per_text = logits_per_image.t()

#         # compute bidirectional CE loss
#         num_logits = logits_per_image.shape[0]
#         labels = torch.arange(num_logits, device=logits_per_image.device, dtype=torch.long)
#         loss = (
#             F.cross_entropy(logits_per_image, labels) +
#             F.cross_entropy(logits_per_text, labels)
#             ) / 2

#         if return_similarity_mat:
#             return loss, logits_per_image
#         else:
#             monitors = {}
#             return loss, monitors

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07, max_logit_scale=100, margin=0.0, **kwargs):
        super().__init__()

        # Learnable temperature (CLIP-style)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))

        # Safety clamp (prevents exploding logits)
        self.max_logit_scale = max_logit_scale

        # Optional margin (helps R@1 slightly)
        self.margin = margin

    def forward(self, comp_data, epoch=None, return_similarity_mat=False, **kwargs):
        """
        comp_data should contain:
            motion_emb: (B, D)
            text_emb:   (B, D)
        """

        # -------------------------------
        # 1. Get embeddings
        # -------------------------------
        im = comp_data['motion_emb']   # motion embeddings
        s  = comp_data['text_emb']     # text embeddings

        # -------------------------------
        # 2. Normalize (CRITICAL)
        # -------------------------------
        im = F.normalize(im, dim=-1)
        s  = F.normalize(s, dim=-1)

        # -------------------------------
        # 3. Compute similarity (cosine)
        # -------------------------------
        logit_scale = self.logit_scale.exp().clamp(max=self.max_logit_scale)

        logits_per_image = logit_scale * im @ s.t()
        logits_per_text  = logits_per_image.t()

        # Optional margin (small boost for R@1)
        if self.margin > 0:
            logits_per_image = logits_per_image - self.margin
            logits_per_text  = logits_per_text - self.margin

        # -------------------------------
        # 4. Labels (diagonal = correct match)
        # -------------------------------
        batch_size = logits_per_image.shape[0]
        labels = torch.arange(batch_size, device=logits_per_image.device)

        # -------------------------------
        # 5. Bidirectional InfoNCE loss
        # -------------------------------
        loss_i2t = F.cross_entropy(logits_per_image, labels)
        loss_t2i = F.cross_entropy(logits_per_text, labels)

        loss = (loss_i2t + loss_t2i) / 2

        # -------------------------------
        # 6. Optional monitoring
        # -------------------------------
        monitors = {
            "loss_i2t": loss_i2t.item(),
            "loss_t2i": loss_t2i.item(),
            "logit_scale": logit_scale.item()
        }

        if return_similarity_mat:
            return loss, logits_per_image
        else:
            return loss, monitors