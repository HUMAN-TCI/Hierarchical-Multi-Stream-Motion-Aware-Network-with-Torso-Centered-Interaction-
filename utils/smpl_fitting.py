import numpy as np
import torch
from smplx import SMPL


def load_smpl_model(smpl_model_dir, device, gender='NEUTRAL', batch_size=1):
    """
    Load SMPL model from folder containing SMPL_NEUTRAL.pkl, etc.

    Example smpl_model_dir:
        C:\\text-to-motion-retrieval_Exp\\smpl_models\\smpl\\SMPL
    """
    smpl = SMPL(
        model_path=smpl_model_dir,
        gender=gender.lower(),
        batch_size=batch_size
    ).to(device)
    return smpl


def get_humanml_to_smpl_joint_map():
    """
    Approximate joint mapping from HumanML3D 22-joint format
    to SMPL 24-joint output.

    HumanML3D 22 joints often follow:
        0 pelvis
        1 left_hip
        2 right_hip
        3 spine1
        4 left_knee
        5 right_knee
        6 spine2
        7 left_ankle
        8 right_ankle
        9 spine3
        10 left_foot
        11 right_foot
        12 neck
        13 left_collar
        14 right_collar
        15 head
        16 left_shoulder
        17 right_shoulder
        18 left_elbow
        19 right_elbow
        20 left_wrist
        21 right_wrist

    SMPL 24 joints commonly:
        0 pelvis
        1 left_hip
        2 right_hip
        3 spine1
        4 left_knee
        5 right_knee
        6 spine2
        7 left_ankle
        8 right_ankle
        9 spine3
        10 left_foot
        11 right_foot
        12 neck
        13 left_collar
        14 right_collar
        15 head
        16 left_shoulder
        17 right_shoulder
        18 left_elbow
        19 right_elbow
        20 left_wrist
        21 right_wrist
        22 left_hand
        23 right_hand

    We fit only the overlapping 22 joints.
    """
    humanml_idx = list(range(22))
    smpl_idx = list(range(22))
    return humanml_idx, smpl_idx


def fit_smpl_to_single_frame(
    target_joints,
    smpl_model,
    device,
    num_iters=150,
    lr=0.05,
    pose_reg_weight=0.001,
    betas_reg_weight=0.0005,
    init_transl_from_root=True
):
    """
    Fit SMPL to one frame of target joints.

    Args:
        target_joints: numpy array of shape (22, 3)
        smpl_model: loaded SMPL model
        device: torch device
    Returns:
        result dict with:
            joints_24: (24,3)
            vertices: (6890,3)
            transl: (3,)
            global_orient: (3,)
            body_pose: (69,)
            betas: (10,)
            loss: float
    """
    assert target_joints.shape == (22, 3), f"Expected (22,3), got {target_joints.shape}"

    target = torch.tensor(target_joints, dtype=torch.float32, device=device).unsqueeze(0)  # (1,22,3)

    # Parameters to optimize
    if init_transl_from_root:
        init_root = target[:, 0, :].clone()
    else:
        init_root = torch.zeros((1, 3), dtype=torch.float32, device=device)

    transl = torch.nn.Parameter(init_root)
    global_orient = torch.nn.Parameter(torch.zeros((1, 3), dtype=torch.float32, device=device))
    body_pose = torch.nn.Parameter(torch.zeros((1, 69), dtype=torch.float32, device=device))
    betas = torch.nn.Parameter(torch.zeros((1, 10), dtype=torch.float32, device=device))

    optimizer = torch.optim.Adam([transl, global_orient, body_pose, betas], lr=lr)

    humanml_idx, smpl_idx = get_humanml_to_smpl_joint_map()

    best_loss = float("inf")
    best_output = None

    for _ in range(num_iters):
        optimizer.zero_grad()

        output = smpl_model(
            betas=betas,
            body_pose=body_pose,
            global_orient=global_orient,
            transl=transl
        )

        smpl_joints = output.joints[:, smpl_idx, :]   # (1,22,3)

        joint_loss = ((smpl_joints - target[:, humanml_idx, :]) ** 2).mean()
        pose_reg = (body_pose ** 2).mean()
        betas_reg = (betas ** 2).mean()

        loss = joint_loss + pose_reg_weight * pose_reg + betas_reg_weight * betas_reg
        loss.backward()
        optimizer.step()

        current_loss = loss.item()
        if current_loss < best_loss:
            best_loss = current_loss
            best_output = {
                "joints_24": output.joints[0].detach().cpu().numpy(),
                "vertices": output.vertices[0].detach().cpu().numpy(),
                "transl": transl[0].detach().cpu().numpy(),
                "global_orient": global_orient[0].detach().cpu().numpy(),
                "body_pose": body_pose[0].detach().cpu().numpy(),
                "betas": betas[0].detach().cpu().numpy(),
                "loss": best_loss
            }

    return best_output


def fit_smpl_to_sequence(
    joint_sequence,
    smpl_model,
    device,
    num_iters=150,
    lr=0.05,
    pose_reg_weight=0.001,
    betas_reg_weight=0.0005,
    verbose=True
):
    """
    Fit SMPL frame-by-frame to a motion sequence.

    Args:
        joint_sequence: numpy array (T,22,3)

    Returns:
        result dict with:
            joints: (T,24,3)
            vertices: (T,6890,3)
            transl: (T,3)
            global_orient: (T,3)
            body_pose: (T,69)
            betas: (T,10)
            losses: (T,)
    """
    assert joint_sequence.ndim == 3 and joint_sequence.shape[1:] == (22, 3), \
        f"Expected (T,22,3), got {joint_sequence.shape}"

    T = joint_sequence.shape[0]

    all_joints = []
    all_vertices = []
    all_transl = []
    all_global_orient = []
    all_body_pose = []
    all_betas = []
    all_losses = []

    for t in range(T):
        if verbose and (t % 10 == 0 or t == T - 1):
            print(f"[SMPL Fitting] Frame {t+1}/{T}")

        fit = fit_smpl_to_single_frame(
            target_joints=joint_sequence[t],
            smpl_model=smpl_model,
            device=device,
            num_iters=num_iters,
            lr=lr,
            pose_reg_weight=pose_reg_weight,
            betas_reg_weight=betas_reg_weight
        )

        all_joints.append(fit["joints_24"])
        all_vertices.append(fit["vertices"])
        all_transl.append(fit["transl"])
        all_global_orient.append(fit["global_orient"])
        all_body_pose.append(fit["body_pose"])
        all_betas.append(fit["betas"])
        all_losses.append(fit["loss"])

    return {
        "joints": np.stack(all_joints, axis=0),              # (T,24,3)
        "vertices": np.stack(all_vertices, axis=0),          # (T,6890,3)
        "transl": np.stack(all_transl, axis=0),              # (T,3)
        "global_orient": np.stack(all_global_orient, axis=0),# (T,3)
        "body_pose": np.stack(all_body_pose, axis=0),        # (T,69)
        "betas": np.stack(all_betas, axis=0),                # (T,10)
        "losses": np.array(all_losses)                       # (T,)
    }


# # utils/smpl_utils.py
# import torch
# import pickle
# import os

# class SMPLWrapper:
#     def __init__(self, model_path, device='cpu'):
#         with open(model_path, 'rb') as f:
#             self.smpl_data = pickle.load(f, encoding='latin1')
#         self.device = device
#         self.faces = self.smpl_data['f']
#         self.v_template = torch.tensor(self.smpl_data['v_template'], dtype=torch.float32, device=device)

#     def forward(self, joint_positions):
#         """
#         joint_positions: (T, J, 3)
#         Returns: vertices (T, 6890, 3), faces (6890, 3)
#         """
#         T = joint_positions.shape[0]
#         verts = self.v_template.unsqueeze(0).repeat(T, 1, 1)
#         # Here we just add root translation from joint_positions[:, 0] to SMPL verts
#         root_pos = joint_positions[:, 0:1, :]  # assuming root is joint 0
#         verts = verts + root_pos
#         return verts, self.faces