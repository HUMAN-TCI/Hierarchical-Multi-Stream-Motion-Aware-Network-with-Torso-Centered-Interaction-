# ////////////////////////////////////  Original code //////////////////////////

# import torch
# import torch.nn as nn

# from .skeleton import skeleton_parts_ids

# class UpperLowerGRU(nn.Module):
#     def __init__(self, h1, h2, h3, num_layers=1, data_rep='cont_6d', dataset='kit'):
#         super(UpperLowerGRU, self).__init__()

#         self.skel_parts_ids = skeleton_parts_ids[dataset]

#         f = 6 if data_rep == 'cont_6d' else 6+3
#         self.layer1_rarm_enc = nn.Linear(f * len(self.skel_parts_ids['right_arm']), h1)
#         self.layer1_larm_enc = nn.Linear(f * len(self.skel_parts_ids['left_arm']), h1)
#         self.layer1_rleg_enc = nn.Linear(f * len(self.skel_parts_ids['right_leg']), h1)
#         self.layer1_lleg_enc = nn.Linear(f * len(self.skel_parts_ids['left_leg']), h1)
#         self.layer1_torso_enc = nn.Linear(f * len(self.skel_parts_ids['mid_body']), h1)
#         self.layer2_rarm_enc = nn.Linear(2*h1, h2)
#         self.layer2_larm_enc = nn.Linear(2*h1, h2)
#         self.layer2_rleg_enc = nn.Linear(2*h1, h2)
#         self.layer2_lleg_enc = nn.Linear(2*h1, h2)
#         self.batchnorm_up = nn.BatchNorm1d(2*h2)
#         self.batchnorm_lo = nn.BatchNorm1d(2*h2)
#         self.layer3_arm = nn.GRU(
#             2*h2, h3, num_layers=num_layers, batch_first=True)
#         self.layer3_leg = nn.GRU(
#             2*h2, h3, num_layers=num_layers, batch_first=True)
#         self.h3 = h3

#     def get_output_dim(self):
#         return self.h3 * 2

#     def forward(self, P_in, lengths):
#         # poseinput is of shape [b,t,num_joints,dim]

#         # P_in, h = self.rnn(pose_input)
#         b, t = P_in.shape[:2]
#         right_arm = P_in[..., self.skel_parts_ids['right_arm'], :].view(b, t, -1)
#         left_arm = P_in[..., self.skel_parts_ids['left_arm'], :].view(b, t, -1)
#         right_leg = P_in[..., self.skel_parts_ids['right_leg'], :].view(b, t, -1)
#         left_leg = P_in[..., self.skel_parts_ids['left_leg'], :].view(b, t, -1)
#         mid_body = P_in[..., self.skel_parts_ids['mid_body'], :].view(b, t, -1)

#         right_arm_layer1 = self.layer1_rarm_enc(right_arm)
#         left_arm_layer1 = self.layer1_larm_enc(left_arm)
#         mid_body_layer1 = self.layer1_torso_enc(mid_body)

#         right_arm_layer2 = self.layer2_rarm_enc(
#             torch.cat((right_arm_layer1, mid_body_layer1), dim=-1))
#         left_arm_layer2 = self.layer2_larm_enc(
#             torch.cat((left_arm_layer1, mid_body_layer1), dim=-1))

#         upperbody = torch.cat((right_arm_layer2, left_arm_layer2), dim=-1)
#         upperbody_bn = self.batchnorm_up(
#             upperbody.permute(0, 2, 1)).permute(0, 2, 1)

#         # handle padded sequence with correct lengths
#         upperbody_seq = torch.nn.utils.rnn.pack_padded_sequence(upperbody_bn.view(
#             upperbody.shape[0], upperbody.shape[1], -1), lengths, batch_first=True, enforce_sorted=False)
#         _, h = self.layer3_arm(upperbody_seq)
#         z_p_upper = h.squeeze(0)

#         right_leg_layer1 = self.layer1_rleg_enc(right_leg)
#         left_leg_layer1 = self.layer1_lleg_enc(left_leg)

#         right_leg_layer2 = self.layer2_rleg_enc(
#             torch.cat((right_leg_layer1, mid_body_layer1), dim=-1))
#         left_leg_layer2 = self.layer2_lleg_enc(
#             torch.cat((left_leg_layer1, mid_body_layer1), dim=-1))

#         lower_body = torch.cat((right_leg_layer2, left_leg_layer2), dim=-1)
#         lower_body_bn = self.batchnorm_lo(
#             lower_body.permute(0, 2, 1)).permute(0, 2, 1)

#         # handle padded sequence with correct lengths
#         lowerbody_seq = torch.nn.utils.rnn.pack_padded_sequence(lower_body_bn.view(
#             lower_body.shape[0], lower_body.shape[1], -1), lengths, batch_first=True, enforce_sorted=False)
#         _, h = self.layer3_leg(lowerbody_seq)
#         z_p_lower = h.squeeze(0)

#         motion_emb = torch.cat((z_p_upper, z_p_lower), dim=-1)
#         return motion_emb

# ///////////////////////////// no projection for full training 3 layers ///////////////////




# import torch
# import torch.nn as nn
# from .skeleton import skeleton_parts_ids

# class UpperLowerTorsoGRU(nn.Module):
#     def __init__(self, h1, h2, h3, num_layers=1, data_rep='cont_6d', dataset='kit'):
#         super(UpperLowerTorsoGRU, self).__init__()

#         self.skel_parts_ids = skeleton_parts_ids[dataset]
#         f = 6 if data_rep == 'cont_6d' else 6 + 3

#         # -----------------------------
#         # 1️⃣  Layer 1 encoders
#         # -----------------------------
#         # Upper (arms)
#         self.layer1_rarm_enc = nn.Linear(f * len(self.skel_parts_ids['right_arm']), h1)
#         self.layer1_larm_enc = nn.Linear(f * len(self.skel_parts_ids['left_arm']), h1)

#         # Lower (legs)
#         self.layer1_rleg_enc = nn.Linear(f * len(self.skel_parts_ids['right_leg']), h1)
#         self.layer1_lleg_enc = nn.Linear(f * len(self.skel_parts_ids['left_leg']), h1)

#         # Torso (spine, hips, mid_body)
#         self.layer1_torso_enc = nn.Linear(f * len(self.skel_parts_ids['mid_body']), h1)

#         # -----------------------------
#         # 2️⃣  Layer 2 encoders (combine limb+torso)
#         # -----------------------------
#         self.layer2_arm_enc = nn.Linear(2 * h1 + h1, h2)   # arms + torso
#         self.layer2_leg_enc = nn.Linear(2 * h1 + h1, h2)   # legs + torso
#         self.layer2_torso_enc = nn.Linear(h1, h2)          # torso standalone

#         # -----------------------------
#         # 3️⃣  Normalization and GRUs
#         # -----------------------------
#         self.batchnorm_up = nn.BatchNorm1d(h2)
#         self.batchnorm_lo = nn.BatchNorm1d(h2)
#         self.batchnorm_torso = nn.BatchNorm1d(h2)

#         self.layer3_arm = nn.GRU(h2, h3, num_layers=num_layers, batch_first=True)
#         self.layer3_leg = nn.GRU(h2, h3, num_layers=num_layers, batch_first=True)
#         self.layer3_torso = nn.GRU(h2, h3, num_layers=num_layers, batch_first=True)

#         self.h3 = h3

#     def get_output_dim(self):
#         return self.h3 * 3  # upper + lower + torso

#     def forward(self, P_in, lengths):
#         b, t = P_in.shape[:2]
#         # -----------------------------
#         # Split into parts
#         # -----------------------------
#         right_arm = P_in[..., self.skel_parts_ids['right_arm'], :].view(b, t, -1)
#         left_arm = P_in[..., self.skel_parts_ids['left_arm'], :].view(b, t, -1)
#         right_leg = P_in[..., self.skel_parts_ids['right_leg'], :].view(b, t, -1)
#         left_leg = P_in[..., self.skel_parts_ids['left_leg'], :].view(b, t, -1)
#         mid_body = P_in[..., self.skel_parts_ids['mid_body'], :].view(b, t, -1)

#         # -----------------------------
#         # Layer 1
#         # -----------------------------
#         rarm_1 = self.layer1_rarm_enc(right_arm)
#         larm_1 = self.layer1_larm_enc(left_arm)
#         rleg_1 = self.layer1_rleg_enc(right_leg)
#         lleg_1 = self.layer1_lleg_enc(left_leg)
#         torso_1 = self.layer1_torso_enc(mid_body)

#         # -----------------------------
#         # Layer 2 - combine limbs with torso
#         # -----------------------------
#         arm_features = torch.cat((rarm_1, larm_1, torso_1), dim=-1)
#         leg_features = torch.cat((rleg_1, lleg_1, torso_1), dim=-1)

#         arm_2 = self.layer2_arm_enc(arm_features)
#         leg_2 = self.layer2_leg_enc(leg_features)
#         torso_2 = self.layer2_torso_enc(torso_1)

#         # -----------------------------
#         # Normalization
#         # -----------------------------
#         arm_bn = self.batchnorm_up(arm_2.permute(0, 2, 1)).permute(0, 2, 1)
#         leg_bn = self.batchnorm_lo(leg_2.permute(0, 2, 1)).permute(0, 2, 1)
#         torso_bn = self.batchnorm_torso(torso_2.permute(0, 2, 1)).permute(0, 2, 1)

#         # -----------------------------
#         # GRU for each part (handle padding)
#         # -----------------------------
#         arm_seq = nn.utils.rnn.pack_padded_sequence(arm_bn, lengths, batch_first=True, enforce_sorted=False)
#         leg_seq = nn.utils.rnn.pack_padded_sequence(leg_bn, lengths, batch_first=True, enforce_sorted=False)
#         torso_seq = nn.utils.rnn.pack_padded_sequence(torso_bn, lengths, batch_first=True, enforce_sorted=False)

#         _, h_arm = self.layer3_arm(arm_seq)
#         _, h_leg = self.layer3_leg(leg_seq)
#         _, h_torso = self.layer3_torso(torso_seq)

#         z_upper = h_arm.squeeze(0)
#         z_lower = h_leg.squeeze(0)
#         z_torso = h_torso.squeeze(0)

#         # -----------------------------
#         # Fuse all parts
#         # -----------------------------
#         motion_emb = torch.cat((z_upper, z_lower, z_torso), dim=-1)
#         return motion_emb




# //////////////////////////////////////3rd variant of torso with projction layer /////////////////////////////////////////////

# import torch
# import torch.nn as nn
# from .skeleton import skeleton_parts_ids

# class UpperLowerGRU(nn.Module):
#     """
#     Original class name restored so the whole pipeline,
#     pretrained checkpoints, and imports keep working.
#     Now includes torso processing but preserves all naming.
#     """

#     def __init__(self, h1, h2, h3, num_layers=1, data_rep='cont_6d', dataset='kit'):
#         super(UpperLowerGRU, self).__init__()

#         self.skel_parts_ids = skeleton_parts_ids[dataset]
#         f = 6 if data_rep == 'cont_6d' else 6 + 3

#         # -----------------------------
#         # 1) Layer 1 encoders
#         # -----------------------------
#         self.layer1_rarm_enc = nn.Linear(f * len(self.skel_parts_ids['right_arm']), h1)
#         self.layer1_larm_enc = nn.Linear(f * len(self.skel_parts_ids['left_arm']), h1)

#         self.layer1_rleg_enc = nn.Linear(f * len(self.skel_parts_ids['right_leg']), h1)
#         self.layer1_lleg_enc = nn.Linear(f * len(self.skel_parts_ids['left_leg']), h1)

#         self.layer1_torso_enc = nn.Linear(f * len(self.skel_parts_ids['mid_body']), h1)

#         # -----------------------------
#         # 2) Layer 2 encoders
#         # -----------------------------
#         self.layer2_rarm_enc = nn.Linear(2 * h1, h2)
#         self.layer2_larm_enc = nn.Linear(2 * h1, h2)

#         self.layer2_rleg_enc = nn.Linear(2 * h1, h2)
#         self.layer2_lleg_enc = nn.Linear(2 * h1, h2)
# # /////////////// check again 
#         self.layer2_torso_enc = nn.Linear(h1, h2)

#         # -----------------------------
#         # 3) BatchNorm
#         # -----------------------------
#         self.batchnorm_up = nn.BatchNorm1d(2 * h2)
#         self.batchnorm_lo = nn.BatchNorm1d(2 * h2)
#         self.batchnorm_torso = nn.BatchNorm1d(h2)

#         # -----------------------------
#         # 4) GRUs
#         # -----------------------------
#         self.layer3_arm = nn.GRU(2 * h2, h3, num_layers=num_layers, batch_first=True)
#         self.layer3_leg = nn.GRU(2 * h2, h3, num_layers=num_layers, batch_first=True)
#         self.layer3_torso = nn.GRU(h2, h3, num_layers=num_layers, batch_first=True)

#         self.h3 = h3

#         # ----------------------------------------------------
#         # *** NEW FIX ***
#         # Projection for (z_upper + z_lower + z_torso)
#         # concat_dim = 3 * h3 → project to 512
#         # ----------------------------------------------------
#         self.torso_merge = nn.Linear(3 * h3, 512)

#     def get_output_dim(self):
#         # The output dim is now fixed to 512 due to torso_merge
#         return 512

#     def forward(self, P_in, lengths):
#         b, t = P_in.shape[:2]

#         # -----------------------------
#         # Extract body parts
#         # -----------------------------
#         right_arm = P_in[..., self.skel_parts_ids['right_arm'], :].view(b, t, -1)
#         left_arm  = P_in[..., self.skel_parts_ids['left_arm'], :].view(b, t, -1)
#         right_leg = P_in[..., self.skel_parts_ids['right_leg'], :].view(b, t, -1)
#         left_leg  = P_in[..., self.skel_parts_ids['left_leg'], :].view(b, t, -1)
#         mid_body  = P_in[..., self.skel_parts_ids['mid_body'], :].view(b, t, -1)

#         # -----------------------------
#         # Layer 1 encodings
#         # -----------------------------
#         rarm_1 = self.layer1_rarm_enc(right_arm)
#         larm_1 = self.layer1_larm_enc(left_arm)
#         rleg_1 = self.layer1_rleg_enc(right_leg)
#         lleg_1 = self.layer1_lleg_enc(left_leg)
#         torso_1 = self.layer1_torso_enc(mid_body)

#         # -----------------------------
#         # Layer 2 upper body (arms + torso)
#         # -----------------------------
#         upper_r = self.layer2_rarm_enc(torch.cat((rarm_1, torso_1), dim=-1))
#         upper_l = self.layer2_larm_enc(torch.cat((larm_1, torso_1), dim=-1))

#         upper = torch.cat((upper_r, upper_l), dim=-1)
#         upper_bn = self.batchnorm_up(upper.permute(0, 2, 1)).permute(0, 2, 1)

#         upper_seq = nn.utils.rnn.pack_padded_sequence(
#             upper_bn, lengths, batch_first=True, enforce_sorted=False
#         )
#         _, h_upper = self.layer3_arm(upper_seq)
#         z_upper = h_upper.squeeze(0)

#         # -----------------------------
#         # Layer 2 lower body (legs + torso)
#         # -----------------------------
#         lower_r = self.layer2_rleg_enc(torch.cat((rleg_1, torso_1), dim=-1))
#         lower_l = self.layer2_lleg_enc(torch.cat((lleg_1, torso_1), dim=-1))

#         lower = torch.cat((lower_r, lower_l), dim=-1)
#         lower_bn = self.batchnorm_lo(lower.permute(0, 2, 1)).permute(0, 2, 1)

#         lower_seq = nn.utils.rnn.pack_padded_sequence(
#             lower_bn, lengths, batch_first=True, enforce_sorted=False
#         )
#         _, h_lower = self.layer3_leg(lower_seq)
#         z_lower = h_lower.squeeze(0)

#         # -----------------------------
#         # Torso branch
#         # -----------------------------
#         torso_2 = self.layer2_torso_enc(torso_1)
#         torso_bn = self.batchnorm_torso(torso_2.permute(0, 2, 1)).permute(0, 2, 1)

#         torso_seq = nn.utils.rnn.pack_padded_sequence(
#             torso_bn, lengths, batch_first=True, enforce_sorted=False
#         )
#         _, h_torso = self.layer3_torso(torso_seq)
#         z_torso = h_torso.squeeze(0)

#         # -----------------------------
#         # CONCAT = (3 * h3) dimension
#         # -----------------------------
#         concat_emb = torch.cat((z_upper, z_lower, z_torso), dim=-1)

#         # -----------------------------
#         # NEW: REDUCE → 512 dims
#         # -----------------------------
#         merged_emb = self.torso_merge(concat_emb)

#         return merged_emb

# /////////////// variant 4 with attension instead of concat ////////////////////

import torch
import torch.nn as nn
from .skeleton import skeleton_parts_ids


class AttentionFuse(nn.Module):
    """
    Simple cross-attention to fuse (part, torso) features.
    Input:  part_feat  [B,T,h1]
            torso_feat [B,T,h1]
    Output: fused_feat [B,T,h2]
    """
    def __init__(self, h1, h2):
        super().__init__()
        self.query = nn.Linear(h1, h1)
        self.key   = nn.Linear(h1, h1)
        self.value = nn.Linear(h1, h1)

        self.out = nn.Linear(h1, h2)

    def forward(self, part_feat, torso_feat):
        # Q = arm/leg, K/V = torso
        Q = self.query(part_feat)         # [B,T,h1]
        K = self.key(torso_feat)          # [B,T,h1]
        V = self.value(torso_feat)        # [B,T,h1]

        att = torch.softmax((Q * K).sum(-1, keepdim=True), dim=1)   # [B,T,1]
        fused = att * V + part_feat        # residual connection

        return self.out(fused)             # → h2 dim


class UpperLowerGRU(nn.Module):

    def __init__(self, h1, h2, h3, num_layers=1, data_rep='cont_6d', dataset='kit'):
        super().__init__()

        self.skel_parts_ids = skeleton_parts_ids[dataset]
        f = 6 if data_rep == 'cont_6d' else 9

        # ---------- Layer 1 ----------
        self.layer1_rarm_enc = nn.Linear(f * len(self.skel_parts_ids['right_arm']), h1)
        self.layer1_larm_enc = nn.Linear(f * len(self.skel_parts_ids['left_arm']), h1)

        self.layer1_rleg_enc = nn.Linear(f * len(self.skel_parts_ids['right_leg']), h1)
        self.layer1_lleg_enc = nn.Linear(f * len(self.skel_parts_ids['left_leg']), h1)

        self.layer1_torso_enc = nn.Linear(f * len(self.skel_parts_ids['mid_body']), h1)

        # ---------- Layer 2 (Attention Instead of Concatenation) ----------
        self.att_rarm = AttentionFuse(h1, h2)
        self.att_larm = AttentionFuse(h1, h2)

        self.att_rleg = AttentionFuse(h1, h2)
        self.att_lleg = AttentionFuse(h1, h2)

        # Torso independently → no concat, no attention
        self.layer2_torso_enc = nn.Linear(h1, h2)

        # ---------- BatchNorm ----------
        self.batchnorm_up = nn.BatchNorm1d(2 * h2)
        self.batchnorm_lo = nn.BatchNorm1d(2 * h2)
        self.batchnorm_torso = nn.BatchNorm1d(h2)

        # ---------- GRUs ----------
        self.layer3_arm = nn.GRU(2 * h2, h3, num_layers=num_layers, batch_first=True)
        self.layer3_leg = nn.GRU(2 * h2, h3, num_layers=num_layers, batch_first=True)
        self.layer3_torso = nn.GRU(h2, h3, num_layers=num_layers, batch_first=True)

        # combine (upper+lower+torso)
        self.torso_merge = nn.Linear(3 * h3, 512)
        self.h3 = h3

    
    def get_output_dim(self):
        return 512


    def forward(self, P_in, lengths):
        b, t = P_in.shape[:2]

        # ---------- Extract body parts ----------
        right_arm = P_in[..., self.skel_parts_ids['right_arm'], :].view(b, t, -1)
        left_arm  = P_in[..., self.skel_parts_ids['left_arm'], :].view(b, t, -1)
        right_leg = P_in[..., self.skel_parts_ids['right_leg'], :].view(b, t, -1)
        left_leg  = P_in[..., self.skel_parts_ids['left_leg'], :].view(b, t, -1)
        mid_body  = P_in[..., self.skel_parts_ids['mid_body'], :].view(b, t, -1)

        # ---------- Layer 1 ----------
        rarm_1 = self.layer1_rarm_enc(right_arm)
        larm_1 = self.layer1_larm_enc(left_arm)
        rleg_1 = self.layer1_rleg_enc(right_leg)
        lleg_1 = self.layer1_lleg_enc(left_leg)
        torso_1 = self.layer1_torso_enc(mid_body)

        # ---------- Layer 2 with attention ----------
        upper_r = self.att_rarm(rarm_1, torso_1)
        upper_l = self.att_larm(larm_1, torso_1)
        lower_r = self.att_rleg(rleg_1, torso_1)
        lower_l = self.att_lleg(lleg_1, torso_1)

        # ---------- Upper GRU ----------
        upper = torch.cat((upper_r, upper_l), dim=-1)
        upper_bn = self.batchnorm_up(upper.permute(0, 2, 1)).permute(0, 2, 1)

        up_seq = nn.utils.rnn.pack_padded_sequence(
            upper_bn, lengths, batch_first=True, enforce_sorted=False
        )
        _, h_upper = self.layer3_arm(up_seq)
        z_upper = h_upper.squeeze(0)

        # ---------- Lower GRU ----------
        lower = torch.cat((lower_r, lower_l), dim=-1)
        lower_bn = self.batchnorm_lo(lower.permute(0, 2, 1)).permute(0, 2, 1)

        lo_seq = nn.utils.rnn.pack_padded_sequence(
            lower_bn, lengths, batch_first=True, enforce_sorted=False
        )
        _, h_lower = self.layer3_leg(lo_seq)
        z_lower = h_lower.squeeze(0)

        # ---------- Torso GRU ----------
        torso_2 = self.layer2_torso_enc(torso_1)
        torso_bn = self.batchnorm_torso(torso_2.permute(0, 2, 1)).permute(0, 2, 1)

        to_seq = nn.utils.rnn.pack_padded_sequence(
            torso_bn, lengths, batch_first=True, enforce_sorted=False
        )
        _, h_torso = self.layer3_torso(to_seq)
        z_torso = h_torso.squeeze(0)

        # ---------- Merge ----------
        concat = torch.cat((z_upper, z_lower, z_torso), dim=-1)
        return self.torso_merge(concat)


# =================================================================================================


# import torch
# import torch.nn as nn
# from torch.nn.utils.rnn import pack_padded_sequence
# from .skeleton import skeleton_parts_ids


# # ============================================================
# # Multi-head cross-attention for body-part ↔ context fusion
# # ============================================================
# class AttentionFuse(nn.Module):
#     """
#     Multi-head cross-attention:
#         Query  = part (arm / leg)
#         Key    = context (torso or global)
#         Value  = context
#     """

#     def __init__(self, dim, num_heads=4):
#         super().__init__()
#         self.attn = nn.MultiheadAttention(
#             embed_dim=dim,
#             num_heads=num_heads,
#             batch_first=True
#         )
#         self.norm = nn.LayerNorm(dim)

#     def forward(self, part_feat, context_feat):
#         """
#         part_feat    : [B, T, D]
#         context_feat : [B, T, D]
#         """
#         attn_out, _ = self.attn(
#             query=part_feat,
#             key=context_feat,
#             value=context_feat
#         )
#         return self.norm(part_feat + attn_out)   # residual


# # ============================================================
# # Upper–Lower–Torso GRU Motion Encoder
# # ============================================================
# class UpperLowerGRU(nn.Module):

#     def __init__(
#         self,
#         h1,
#         h2,
#         h3,
#         num_layers=1,
#         data_rep="cont_6d_plus_rifke",
#         dataset="t2m"
#     ):
#         super().__init__()

#         # --------------------------------------------------
#         # Handle data representation correctly
#         # --------------------------------------------------
#         if data_rep in ["xyz", "cont_6d_plus_rifke"]:
#             self.f = 9
#         elif data_rep == "cont_6d":
#             self.f = 6
#         else:
#             raise ValueError(f"Unknown data_rep: {data_rep}")

#         self.parts = skeleton_parts_ids[dataset]

#         # --------------------------------------------------
#         # Layer 1: Per-part linear encoding
#         # --------------------------------------------------
#         def enc(part):
#             return nn.Linear(self.f * len(self.parts[part]), h1)

#         self.rarm_enc = enc("right_arm")
#         self.larm_enc = enc("left_arm")
#         self.rleg_enc = enc("right_leg")
#         self.lleg_enc = enc("left_leg")
#         self.torso_enc = enc("mid_body")

#         # --------------------------------------------------
#         # Layer 2: Attention-based fusion
#         # --------------------------------------------------
#         self.att_rarm = AttentionFuse(h1)
#         self.att_larm = AttentionFuse(h1)
#         self.att_rleg = AttentionFuse(h1)
#         self.att_lleg = AttentionFuse(h1)

#         self.proj = nn.Linear(h1, h2)

#         # --------------------------------------------------
#         # BatchNorm
#         # --------------------------------------------------
#         self.bn_upper = nn.BatchNorm1d(2 * h2)
#         self.bn_lower = nn.BatchNorm1d(2 * h2)
#         self.bn_torso = nn.BatchNorm1d(h2)

#         # --------------------------------------------------
#         # GRUs
#         # --------------------------------------------------
#         self.gru_upper = nn.GRU(2 * h2, h3, num_layers, batch_first=True)
#         self.gru_lower = nn.GRU(2 * h2, h3, num_layers, batch_first=True)
#         self.gru_torso = nn.GRU(h2, h3, num_layers, batch_first=True)

#         # --------------------------------------------------
#         # Final embedding
#         # --------------------------------------------------
#         self.out = nn.Linear(3 * h3, 512)

#     # --------------------------------------------------
#     def get_output_dim(self):
#         return 512

#     # --------------------------------------------------
#     def forward(self, motion, lengths):
#         """
#         motion : [B, T, J, f]
#         lengths: [B]
#         """
#         B, T = motion.shape[:2]

#         # -------------------------------
#         # Extract joints
#         # -------------------------------
#         def get(part):
#             return motion[:, :, self.parts[part], :].reshape(B, T, -1)

#         rarm = self.rarm_enc(get("right_arm"))
#         larm = self.larm_enc(get("left_arm"))
#         rleg = self.rleg_enc(get("right_leg"))
#         lleg = self.lleg_enc(get("left_leg"))
#         torso = self.torso_enc(get("mid_body"))

#         # -------------------------------
#         # Attention fusion
#         # -------------------------------
#         rarm = self.proj(self.att_rarm(rarm, torso))
#         larm = self.proj(self.att_larm(larm, torso))
#         rleg = self.proj(self.att_rleg(rleg, torso))
#         lleg = self.proj(self.att_lleg(lleg, torso))
#         torso = self.proj(torso)

#         # -------------------------------
#         # Upper body GRU
#         # -------------------------------
#         upper = torch.cat([rarm, larm], dim=-1)
#         upper = self.bn_upper(upper.permute(0, 2, 1)).permute(0, 2, 1)
#         upper = pack_padded_sequence(upper, lengths, batch_first=True, enforce_sorted=False)
#         _, h_upper = self.gru_upper(upper)
#         z_upper = h_upper[-1]

#         # -------------------------------
#         # Lower body GRU
#         # -------------------------------
#         lower = torch.cat([rleg, lleg], dim=-1)
#         lower = self.bn_lower(lower.permute(0, 2, 1)).permute(0, 2, 1)
#         lower = pack_padded_sequence(lower, lengths, batch_first=True, enforce_sorted=False)
#         _, h_lower = self.gru_lower(lower)
#         z_lower = h_lower[-1]

#         # -------------------------------
#         # Torso GRU
#         # -------------------------------
#         torso = self.bn_torso(torso.permute(0, 2, 1)).permute(0, 2, 1)
#         torso = pack_padded_sequence(torso, lengths, batch_first=True, enforce_sorted=False)
#         _, h_torso = self.gru_torso(torso)
#         z_torso = h_torso[-1]

#         # -------------------------------
#         # Final embedding
#         # -------------------------------
#         return self.out(torch.cat([z_upper, z_lower, z_torso], dim=-1))



# # /////////////////////////// transformer //////////////////


# # import torch
# # import torch.nn as nn
# # from torch.nn.utils.rnn import pad_sequence
# # from .skeleton import skeleton_parts_ids


# # # ============================================================
# # # Multi-head cross-attention
# # # ============================================================
# # class AttentionFuse(nn.Module):
# #     def __init__(self, dim, num_heads=4):
# #         super().__init__()
# #         self.attn = nn.MultiheadAttention(
# #             embed_dim=dim,
# #             num_heads=num_heads,
# #             batch_first=True
# #         )
# #         self.norm = nn.LayerNorm(dim)

# #     def forward(self, part_feat, context_feat):
# #         attn_out, _ = self.attn(
# #             query=part_feat,
# #             key=context_feat,
# #             value=context_feat
# #         )
# #         return self.norm(part_feat + attn_out)


# # # ============================================================
# # # Transformer encoder block
# # # ============================================================
# # class TemporalTransformer(nn.Module):
# #     def __init__(self, dim, depth=2, num_heads=4, dropout=0.1):
# #         super().__init__()

# #         encoder_layer = nn.TransformerEncoderLayer(
# #             d_model=dim,
# #             nhead=num_heads,
# #             dim_feedforward=4 * dim,
# #             dropout=dropout,
# #             batch_first=True,
# #             norm_first=True
# #         )
# #         self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

# #     def forward(self, x, lengths):
# #         """
# #         x: [B, T, D]
# #         lengths: [B]
# #         """
# #         B, T, _ = x.shape
# #         device = x.device

# #         # padding mask (True = ignore)
# #         mask = torch.arange(T, device=device).expand(B, T) >= lengths.unsqueeze(1)

# #         x = self.encoder(x, src_key_padding_mask=mask)

# #         # temporal pooling (masked mean)
# #         mask = (~mask).unsqueeze(-1)
# #         x = (x * mask).sum(dim=1) / mask.sum(dim=1)

# #         return x


# # # ============================================================
# # # Upper–Lower–Torso Transformer Motion Encoder
# # # ============================================================
# # class UpperLowerGRU(nn.Module):
# #     """
# #     NOTE:
# #     Class name kept as UpperLowerGRU
# #     so Hydra + checkpoints DO NOT BREAK.
# #     """

# #     def __init__(
# #         self,
# #         h1,
# #         h2,
# #         h3,
# #         num_layers=2,
# #         data_rep="cont_6d_plus_rifke",
# #         dataset="t2m",
# #         num_heads=4,
# #         dropout=0.1
# #     ):
# #         super().__init__()

# #         # --------------------------------------------------
# #         # Data representation
# #         # --------------------------------------------------
# #         if data_rep in ["xyz", "cont_6d_plus_rifke"]:
# #             self.f = 9
# #         elif data_rep == "cont_6d":
# #             self.f = 6
# #         else:
# #             raise ValueError(f"Unknown data_rep: {data_rep}")

# #         self.parts = skeleton_parts_ids[dataset]

# #         # --------------------------------------------------
# #         # Per-part encoders
# #         # --------------------------------------------------
# #         def enc(part):
# #             return nn.Linear(self.f * len(self.parts[part]), h1)

# #         self.rarm_enc = enc("right_arm")
# #         self.larm_enc = enc("left_arm")
# #         self.rleg_enc = enc("right_leg")
# #         self.lleg_enc = enc("left_leg")
# #         self.torso_enc = enc("mid_body")

# #         # --------------------------------------------------
# #         # Cross-attention
# #         # --------------------------------------------------
# #         self.att_rarm = AttentionFuse(h1, num_heads)
# #         self.att_larm = AttentionFuse(h1, num_heads)
# #         self.att_rleg = AttentionFuse(h1, num_heads)
# #         self.att_lleg = AttentionFuse(h1, num_heads)

# #         self.proj = nn.Linear(h1, h2)

# #         # --------------------------------------------------
# #         # Transformers
# #         # --------------------------------------------------
# #         self.upper_tf = TemporalTransformer(h2 * 2, num_layers, num_heads, dropout)
# #         self.lower_tf = TemporalTransformer(h2 * 2, num_layers, num_heads, dropout)
# #         self.torso_tf = TemporalTransformer(h2, num_layers, num_heads, dropout)

# #         # --------------------------------------------------
# #         # Output
# #         # --------------------------------------------------
# #         self.out = nn.Linear(3 * h2, 512)

# #     # --------------------------------------------------
# #     def get_output_dim(self):
# #         return 512

# #     # --------------------------------------------------
# #     def forward(self, motion, lengths):
# #         """
# #         motion: [B, T, J, f]
# #         lengths: [B]
# #         """
# #         B, T = motion.shape[:2]

# #         def get(part):
# #             return motion[:, :, self.parts[part], :].reshape(B, T, -1)

# #         # -------------------------------
# #         # Encode joints
# #         # -------------------------------
# #         rarm = self.rarm_enc(get("right_arm"))
# #         larm = self.larm_enc(get("left_arm"))
# #         rleg = self.rleg_enc(get("right_leg"))
# #         lleg = self.lleg_enc(get("left_leg"))
# #         torso = self.torso_enc(get("mid_body"))

# #         # -------------------------------
# #         # Attention fusion
# #         # -------------------------------
# #         rarm = self.proj(self.att_rarm(rarm, torso))
# #         larm = self.proj(self.att_larm(larm, torso))
# #         rleg = self.proj(self.att_rleg(rleg, torso))
# #         lleg = self.proj(self.att_lleg(lleg, torso))
# #         torso = self.proj(torso)

# #         # -------------------------------
# #         # Transformers
# #         # -------------------------------
# #         upper = torch.cat([rarm, larm], dim=-1)
# #         lower = torch.cat([rleg, lleg], dim=-1)

# #         z_upper = self.upper_tf(upper, lengths)
# #         z_lower = self.lower_tf(lower, lengths)
# #         z_torso = self.torso_tf(torso, lengths)

# #         # -------------------------------
# #         # Final embedding
# #         # -------------------------------
# #         return self.out(torch.cat([z_upper, z_lower, z_torso], dim=-1))
