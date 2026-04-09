# test_smpl.py
import os
import sys
import torch

# --- Ensure SMPL module is accessible ---
# Add the smpl folder to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from smpl import SMPL  # your SMPL class
except ModuleNotFoundError:
    print("Could not find SMPL module. Make sure smpl.py is in the same folder.")
    raise

# Optional: for visualization
try:
    import trimesh
    visualize = True
except ModuleNotFoundError:
    print("trimesh not installed. Skipping visualization.")
    visualize = False

# --- Check SMPL folder exists ---
smpl_folder = os.path.join(current_dir, 'models')  # adjust if your SMPL weights are elsewhere
if not os.path.exists(smpl_folder):
    print("SMPL folder does not exist at:", smpl_folder)
else:
    print("Checking SMPL folder exists:", os.path.exists(smpl_folder))

# --- Initialize SMPL ---
smpl_model = SMPL(model_path=smpl_folder)
print("SMPL loaded!")

# --- Forward pass to get vertices and faces ---
# Dummy batch of zeros for pose and shape
pose = torch.zeros((1, smpl_model.pose_params_dim))  # depends on your SMPL version
beta = torch.zeros((1, smpl_model.shape_params_dim))  # usually 10

output = smpl_model(pose, beta)
vertices = output.vertices  # [1, 6890, 3]
faces = smpl_model.faces   # [13776, 3] or loaded from model

print("Vertices shape:", vertices.shape)
print("Faces shape:", faces.shape)

# --- Optional visualization ---
if visualize:
    mesh = trimesh.Trimesh(vertices[0].detach().cpu().numpy(), faces)
    mesh.show()