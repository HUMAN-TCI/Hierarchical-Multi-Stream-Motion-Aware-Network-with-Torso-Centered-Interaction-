# # # import os
# # # import numpy as np
# # # import pyrender
# # # import trimesh
# # # import imageio
# # # from tqdm import tqdm

# # # def render_smpl_mesh(npz_file, save_dir, gif_name="smpl_render.gif"):
# # #     """
# # #     Render SMPL mesh from .npz fitted file and save as a GIF.
# # #     Args:
# # #         npz_file: path to .npz SMPL fitted file
# # #         save_dir: directory to save GIF
# # #         gif_name: output GIF filename
# # #     """
# # #     os.makedirs(save_dir, exist_ok=True)

# # #     # Load fitted SMPL parameters
# # #     data = np.load(npz_file)
# # #     if "verts" in data:
# # #         smpl_vertices = data["verts"]  # (T, 6890, 3)
# # #     else:
# # #         raise ValueError("No 'verts' key found in .npz file")

# # #     # Load SMPL faces
# # #     smpl_faces_path = "./smpl_models/smpl/SMPL_NEUTRAL.pkl"  # adjust path if needed
# # #     import pickle, torch
# # #     with open(smpl_faces_path, 'rb') as f:
# # #         smpl_model = pickle.load(f, encoding='latin1')
# # #         smpl_faces = smpl_model['f']  # (13776, 3)

# # #     frames = []

# # #     # Create scene for rendering
# # #     for verts in tqdm(smpl_vertices, desc="Rendering frames"):
# # #         mesh = trimesh.Trimesh(vertices=verts, faces=smpl_faces, process=False)
# # #         mesh.visual.vertex_colors = [200, 200, 250, 255]  # light blue mesh

# # #         scene = pyrender.Scene(bg_color=[255, 255, 255, 0], ambient_light=[0.3, 0.3, 0.3])
# # #         mesh_node = pyrender.Mesh.from_trimesh(mesh)
# # #         scene.add(mesh_node)

# # #         # Set camera
# # #         camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
# # #         scene.add(camera, pose=np.array([
# # #             [1.0, 0.0, 0.0, 0.0],
# # #             [0.0, 1.0, 0.0, -1.0],
# # #             [0.0, 0.0, 1.0, 2.0],
# # #             [0.0, 0.0, 0.0, 1.0]
# # #         ]))

# # #         # Lighting
# # #         light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
# # #         scene.add(light, pose=np.eye(4))

# # #         # Render
# # #         r = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480)
# # #         color, _ = r.render(scene)
# # #         frames.append(color)
# # #         r.delete()

# # #     # Save as GIF
# # #     gif_path = os.path.join(save_dir, gif_name)
# # #     imageio.mimsave(gif_path, frames, fps=25)
# # #     print(f"Saved rendered SMPL GIF to: {gif_path}")

# # # if __name__ == "__main__":
# # #     import argparse
# # #     parser = argparse.ArgumentParser()
# # #     parser.add_argument("--npz_file", type=str, required=True, help="Path to SMPL fitted .npz file")
# # #     parser.add_argument("--save_dir", type=str, default="./outputs/renders_smpl_mesh", help="Directory to save GIF")
# # #     parser.add_argument("--gif_name", type=str, default="smpl_render.gif", help="GIF filename")
# # #     args = parser.parse_args()

# # #     render_smpl_mesh(args.npz_file, args.save_dir, args.gif_name)

# # import os
# # import numpy as np
# # import pyrender
# # import trimesh
# # import imageio
# # from tqdm import tqdm
# # import torch
# # from smplx import SMPL  # make sure smplx is installed: pip install smplx

# # def render_smpl_mesh_from_params(npz_file, save_dir, gif_name="smpl_render.gif"):
# #     """
# #     Render SMPL mesh from .npz file with SMPL parameters (pose, betas, trans)
# #     and save as a GIF.
# #     """
# #     os.makedirs(save_dir, exist_ok=True)

# #     # Load SMPL parameters
# #     data = np.load(npz_file)
# #     print(f"Loaded keys from .npz: {data.files}")
# #     global_orient = torch.tensor(data['global_orient'], dtype=torch.float32)  # (T, 3)
# #     body_pose = torch.tensor(data['body_pose'], dtype=torch.float32)          # (T, 69)
# #     pose = torch.cat([global_orient, body_pose], dim=-1)                       # (T, 72)
# #     betas = torch.tensor(data['betas'], dtype=torch.float32)                   # (T, 10)
# #     trans = torch.tensor(data['transl'], dtype=torch.float32)                  # (T, 3)

# #     # Load SMPL model
# #     smpl_model = SMPL(model_path="./smpl_models/smpl\SMPL", gender='NEUTRAL', batch_size=1)
# #     # C:\text-to-motion-retrieval_Exp\smpl_models\smpl\SMPL\SMPL_NEUTRAL.pkl

# #     frames = []

# #     for i in tqdm(range(pose.shape[0]), desc="Rendering frames"):
# #         output = smpl_model(
# #             global_orient=pose[i, :3].unsqueeze(0),
# #             body_pose=pose[i, 3:].unsqueeze(0),
# #             betas=betas[i].unsqueeze(0),
# #             transl=trans[i].unsqueeze(0)
# #         )

# #         # verts = output.vertices.detach().cpu().numpy().squeeze()  # (6890, 3)
# #         # faces = smpl_model.faces.astype(np.int32)

# #         # mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
# #         # mesh.visual.vertex_colors = [200, 200, 250, 255]  # light blue mesh

# #         # # Create scene
# #         # scene = pyrender.Scene(bg_color=[255, 255, 255, 0], ambient_light=[0.3, 0.3, 0.3])
# #         # mesh_node = pyrender.Mesh.from_trimesh(mesh)
# #         # scene.add(mesh_node)

# #         # # Camera
# #         # camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
# #         # scene.add(camera, pose=np.array([
# #         #     [1.0, 0.0, 0.0, 0.0],
# #         #     [0.0, 1.0, 0.0, -1.0],
# #         #     [0.0, 0.0, 1.0, 2.0],
# #         #     [0.0, 0.0, 0.0, 1.0]
# #         # ]))
# #         verts_centered = verts - verts.mean(axis=0)
# #         verts_centered *= 1.0  # optional scaling

# #         mesh = trimesh.Trimesh(vertices=verts_centered, faces=smpl_faces, process=False)
# #         mesh.visual.vertex_colors = [200, 200, 250, 255]

# #         scene = pyrender.Scene(bg_color=[255,255,255,0], ambient_light=[0.3,0.3,0.3])
# #         mesh_node = pyrender.Mesh.from_trimesh(mesh)
# #         scene.add(mesh_node)

# #         # Camera
# #         camera_pose = np.array([
# #               [1.0, 0.0, 0.0, 0.0],
# #               [0.0, 1.0, 0.0, 1.0],
# #               [0.0, 0.0, 1.0, 3.0],
# #               [0.0, 0.0, 0.0, 1.0]
# #         ])
# #         camera = pyrender.PerspectiveCamera(yfov=np.pi/3.0)
# #         scene.add(camera, pose=camera_pose)

# #         # Light
# #         light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
# #         scene.add(light, pose=np.eye(4))

# #         # Render
# #         r = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480)
# #         color, _ = r.render(scene)
# #         frames.append(color)
# #         r.delete()

# #     # Save GIF
# #     gif_path = os.path.join(save_dir, gif_name)
# #     imageio.mimsave(gif_path, frames, fps=25)
# #     print(f"Saved rendered SMPL GIF to: {gif_path}")


# # if __name__ == "__main__":
# #     import argparse
# #     parser = argparse.ArgumentParser()
# #     parser.add_argument("--npz_file", type=str, required=True, help="Path to SMPL .npz file with pose, betas, trans")
# #     parser.add_argument("--save_dir", type=str, default="./outputs/renders_smpl_mesh", help="Directory to save GIF")
# #     parser.add_argument("--gif_name", type=str, default="smpl_render.gif", help="GIF filename")
# #     args = parser.parse_args()

# #     render_smpl_mesh_from_params(args.npz_file, args.save_dir, args.gif_name)

# import os
# import numpy as np
# import pyrender
# import trimesh
# import imageio
# from tqdm import tqdm

# def render_smpl_mesh(npz_file, save_dir, gif_name="smpl_render.gif"):
#     """
#     Render SMPL mesh from .npz fitted file and save as a GIF.
#     Args:
#         npz_file: path to .npz SMPL fitted file
#         save_dir: directory to save GIF
#         gif_name: output GIF filename
#     """
#     os.makedirs(save_dir, exist_ok=True)

#     # Load fitted SMPL vertices
#     data = np.load(npz_file)
#     print(f"Loaded keys from .npz: {list(data.keys())}")

#     if "vertices" in data:
#         smpl_vertices = data["vertices"]  # (T, 6890, 3)
#     else:
#         raise ValueError("No 'vertices' key found in .npz file")

#     # Load SMPL faces from SMPL_NEUTRAL.pkl
#     smpl_faces_path = "./smpl_models/smpl/SMPL\SMPL_NEUTRAL.pkl"
#     import pickle
#     with open(smpl_faces_path, 'rb') as f:
#         smpl_model = pickle.load(f, encoding='latin1')
#         smpl_faces = smpl_model['f']  # (13776, 3)

#     frames = []

#     for verts in tqdm(smpl_vertices, desc="Rendering frames"):
#         # Center the mesh
#         verts_centered = verts - verts.mean(axis=0)
#         # Optional: scale down if mesh is too big
#         verts_centered *= 1.0

#         mesh = trimesh.Trimesh(vertices=verts_centered, faces=smpl_faces, process=False)
#         mesh.visual.vertex_colors = [200, 200, 250, 255]  # light blue

#         # Create scene
#         scene = pyrender.Scene(bg_color=[255, 255, 255, 0], ambient_light=[0.3,0.3,0.3])
#         mesh_node = pyrender.Mesh.from_trimesh(mesh)
#         scene.add(mesh_node)

#         # Camera placement (adjusted to see full body)
#         camera_pose = np.array([
#             [1.0, 0.0, 0.0, 0.0],
#             [0.0, 1.0, 0.0, 1.0],
#             [0.0, 0.0, 1.0, 3.0],  # back from origin
#             [0.0, 0.0, 0.0, 1.0]
#         ])
#         camera = pyrender.PerspectiveCamera(yfov=np.pi/3.0)
#         scene.add(camera, pose=camera_pose)

#         # Lighting
#         light = pyrender.DirectionalLight(color=[1.0,1.0,1.0], intensity=2.0)
#         scene.add(light, pose=np.eye(4))

#         # Render
#         r = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480)
#         color, _ = r.render(scene)
#         frames.append(color)
#         r.delete()

#     # Save GIF
#     gif_path = os.path.join(save_dir, gif_name)
#     imageio.mimsave(gif_path, frames, fps=25)
#     print(f"Saved rendered SMPL GIF to: {gif_path}")


# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--npz_file", type=str, required=True, help="Path to SMPL fitted .npz file")
#     parser.add_argument("--save_dir", type=str, default="./outputs/renders_smpl_mesh", help="Directory to save GIF")
#     parser.add_argument("--gif_name", type=str, default="smpl_render.gif", help="GIF filename")
#     args = parser.parse_args()

#     render_smpl_mesh(args.npz_file, args.save_dir, args.gif_name)


# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ grey body mesh style runnable code $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

# import os
# import numpy as np
# import pyrender
# import trimesh
# import imageio
# from tqdm import tqdm

# def render_smpl_mesh(npz_file, save_dir, gif_name="smpl_render.gif"):
#     """
#     Render SMPL mesh from .npz fitted file and save as a GIF.
#     Camera distance adjusts automatically based on mesh size.
#     """
#     os.makedirs(save_dir, exist_ok=True)

#     # Load fitted SMPL vertices
#     data = np.load(npz_file)
#     print(f"Loaded keys from .npz: {list(data.keys())}")

#     if "vertices" in data:
#         smpl_vertices = data["vertices"]  # (T, 6890, 3)
#     else:
#         raise ValueError("No 'vertices' key found in .npz file")

#     # Load SMPL faces from SMPL_NEUTRAL.pkl
#     smpl_faces_path = "./smpl_models/smpl/SMPL/SMPL_NEUTRAL.pkl"
#     import pickle
#     with open(smpl_faces_path, 'rb') as f:
#         smpl_model = pickle.load(f, encoding='latin1')
#         smpl_faces = smpl_model['f']  # (13776, 3)

#     frames = []

#     # Precompute mesh bounds to determine camera distance
#     all_verts = np.vstack(smpl_vertices)
#     max_dim = np.max(all_verts, axis=0) - np.min(all_verts, axis=0)
#     body_radius = np.linalg.norm(max_dim) / 2.0
#     print(f"Computed body radius for camera distance: {body_radius:.3f}")

#     for verts in tqdm(smpl_vertices, desc="Rendering frames"):
#         # Center the mesh
#         verts_centered = verts - verts.mean(axis=0)

#         mesh = trimesh.Trimesh(vertices=verts_centered, faces=smpl_faces, process=False)
#         mesh.visual.vertex_colors = [200, 200, 250, 255]  # light blue

#         # Create scene
#         scene = pyrender.Scene(bg_color=[255, 255, 255, 0], ambient_light=[0.3,0.3,0.3])
#         mesh_node = pyrender.Mesh.from_trimesh(mesh)
#         scene.add(mesh_node)

#         # Camera placement: distance scales with body radius
#         cam_distance = body_radius * 2.5  # factor to ensure full body fits
#         camera_pose = np.array([
#             [1.0, 0.0, 0.0, 0.0],
#             [0.0, 1.0, 0.0, 0.0],
#             [0.0, 0.0, 1.0, cam_distance],
#             [0.0, 0.0, 0.0, 1.0]
#         ])
#         camera = pyrender.PerspectiveCamera(yfov=np.pi/3.0)
#         scene.add(camera, pose=camera_pose)

#         # Lighting
#         light = pyrender.DirectionalLight(color=[1.0,1.0,1.0], intensity=2.0)
#         scene.add(light, pose=np.eye(4))

#         # Render
#         r = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480)
#         color, _ = r.render(scene)
#         frames.append(color)
#         r.delete()

#     # Save GIF
#     gif_path = os.path.join(save_dir, gif_name)
#     imageio.mimsave(gif_path, frames, fps=25)
#     print(f"Saved rendered SMPL GIF to: {gif_path}")


# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--npz_file", type=str, required=True, help="Path to SMPL fitted .npz file")
#     parser.add_argument("--save_dir", type=str, default="./outputs/renders_smpl_mesh", help="Directory to save GIF")
#     parser.add_argument("--gif_name", type=str, default="smpl_render.gif", help="GIF filename")
#     args = parser.parse_args()

#     render_smpl_mesh(args.npz_file, args.save_dir, args.gif_name)

# 44444444444444444444444444 color upper lower and torso ///////////////////

# import os
# import numpy as np
# import pyrender
# import trimesh
# import imageio
# from tqdm import tqdm

# def render_smpl_mesh_colored(npz_file, save_dir, gif_name="smpl_render_colored.gif"):
#     """
#     Render SMPL mesh from .npz with different colors for torso, upper body, lower body.
#     """
#     os.makedirs(save_dir, exist_ok=True)

#     # Load fitted SMPL vertices
#     data = np.load(npz_file)
#     print(f"Loaded keys from .npz: {list(data.keys())}")

#     if "vertices" in data:
#         smpl_vertices = data["vertices"]  # (T, 6890, 3)
#     else:
#         raise ValueError("No 'vertices' key found in .npz file")

#     # Load SMPL faces
#     smpl_faces_path = "./smpl_models/smpl/SMPL/SMPL_NEUTRAL.pkl"
#     import pickle
#     with open(smpl_faces_path, 'rb') as f:
#         smpl_model = pickle.load(f, encoding='latin1')
#         smpl_faces = smpl_model['f']  # (13776, 3)

#     # Define colors
#     torso_color = [255, 200, 200, 255]       # light red
#     upper_body_color = [200, 255, 200, 255]  # light green
#     lower_body_color = [200, 200, 255, 255]  # light blue

#     # Define vertex groups based on SMPL layout
#     torso_idx = np.arange(0, 2000)         # example: torso vertices
#     upper_body_idx = np.arange(2000, 4000) # arms + shoulders
#     lower_body_idx = np.arange(4000, 6890) # legs

#     frames = []

#     # Compute body radius for camera
#     all_verts = np.vstack(smpl_vertices)
#     max_dim = np.max(all_verts, axis=0) - np.min(all_verts, axis=0)
#     body_radius = np.linalg.norm(max_dim) / 2.0

#     for verts in tqdm(smpl_vertices, desc="Rendering frames"):
#         verts_centered = verts - verts.mean(axis=0)

#         # Create colors for each vertex
#         vertex_colors = np.zeros((verts.shape[0], 4), dtype=np.uint8)
#         vertex_colors[torso_idx] = torso_color
#         vertex_colors[upper_body_idx] = upper_body_color
#         vertex_colors[lower_body_idx] = lower_body_color

#         mesh = trimesh.Trimesh(vertices=verts_centered, faces=smpl_faces, vertex_colors=vertex_colors, process=False)

#         scene = pyrender.Scene(bg_color=[255, 255, 255, 0], ambient_light=[0.3,0.3,0.3])
#         scene.add(pyrender.Mesh.from_trimesh(mesh))

#         cam_distance = body_radius * 2.5
#         camera_pose = np.array([
#             [1, 0, 0, 0],
#             [0, 1, 0, 0],
#             [0, 0, 1, cam_distance],
#             [0, 0, 0, 1]
#         ])
#         camera = pyrender.PerspectiveCamera(yfov=np.pi/3.0)
#         scene.add(camera, pose=camera_pose)

#         light = pyrender.DirectionalLight(color=[1,1,1], intensity=2.0)
#         scene.add(light, pose=np.eye(4))

#         r = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480)
#         color, _ = r.render(scene)
#         frames.append(color)
#         r.delete()

#     gif_path = os.path.join(save_dir, gif_name)
#     imageio.mimsave(gif_path, frames, fps=25)
#     print(f"Saved colored SMPL GIF to: {gif_path}")

# import os
# import numpy as np
# import pyrender
# import trimesh
# import imageio
# from tqdm import tqdm

# def render_smpl_mesh_colored(npz_file, save_dir, gif_name="smpl_render.gif"):
#     """
#     Render SMPL mesh from .npz fitted file with colored body parts and save as a GIF.
    
#     Args:
#         npz_file: path to .npz SMPL fitted file
#         save_dir: directory to save GIF
#         gif_name: output GIF filename
#     """
#     os.makedirs(save_dir, exist_ok=True)

#     # Load fitted SMPL parameters
#     data = np.load(npz_file)
#     print("Loaded keys from .npz:", list(data.keys()))

#     if "vertices" not in data:
#         raise ValueError("No 'vertices' key found in .npz file")
#     smpl_vertices = data["vertices"]  # (T, 6890, 3)
#     print(f"Total frames to render: {len(smpl_vertices)}")
#     print("First frame vertices shape:", smpl_vertices[0].shape)

#     # Load SMPL faces
#     smpl_faces_path = "./smpl_models/smpl/SMPL/SMPL_NEUTRAL.pkl"  # adjust path if needed
#     import pickle
#     with open(smpl_faces_path, 'rb') as f:
#         smpl_model = pickle.load(f, encoding='latin1')
#         smpl_faces = smpl_model['f']  # (13776, 3)

#     # Define vertex ranges for body parts (example indices)
#     # You may need to adjust according to your SMPL model vertex mapping
#     torso_idx = np.arange(0, 3000)          # roughly torso vertices
#     upper_body_idx = np.arange(3000, 5000)  # arms
#     lower_body_idx = np.arange(5000, 6890)  # legs

#     frames = []

#     for frame_idx, verts in enumerate(tqdm(smpl_vertices, desc="Rendering frames")):
#         mesh = trimesh.Trimesh(vertices=verts, faces=smpl_faces, process=False)

#         # Assign colors per body part
#         colors = np.zeros((verts.shape[0], 4), dtype=np.uint8)
#         colors[torso_idx] = [255, 0, 0, 255]       # Red torso
#         colors[upper_body_idx] = [0, 255, 0, 255]  # Green arms
#         colors[lower_body_idx] = [0, 0, 255, 255]  # Blue legs
#         mesh.visual.vertex_colors = colors

#         # Create scene
#         scene = pyrender.Scene(bg_color=[255, 255, 255, 0], ambient_light=[0.3, 0.3, 0.3])
#         mesh_node = pyrender.Mesh.from_trimesh(mesh)
#         scene.add(mesh_node)

#         # Camera
#         camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
#         camera_pose = np.array([
#             [1.0, 0.0, 0.0, 0.0],
#             [0.0, 1.0, 0.0, -1.0],
#             [0.0, 0.0, 1.0, 2.0],
#             [0.0, 0.0, 0.0, 1.0]
#         ])
#         scene.add(camera, pose=camera_pose)

#         # Light
#         light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
#         scene.add(light, pose=np.eye(4))

#         # Render frame
#         r = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480)
#         color, _ = r.render(scene)
#         frames.append(color)
#         r.delete()

#     # Save GIF
#     gif_path = os.path.join(save_dir, gif_name)
#     imageio.mimsave(gif_path, frames, fps=25)
#     print(f"Saved rendered SMPL GIF to: {gif_path}")
#     print("Rendering complete!")

# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--npz_file", type=str, required=True, help="Path to SMPL fitted .npz file")
#     parser.add_argument("--save_dir", type=str, default="./outputs/renders_smpl_mesh", help="Directory to save GIF")
#     parser.add_argument("--gif_name", type=str, default="smpl_render.gif", help="GIF filename")
#     args = parser.parse_args()

#     render_smpl_mesh_colored(args.npz_file, args.save_dir, args.gif_name)

# import os
# import numpy as np
# import pyrender
# import trimesh
# import imageio
# from tqdm import tqdm

# def render_smpl_mesh_centered(npz_file, save_dir, gif_name="smpl_render.gif"):
#     """
#     Render SMPL mesh from .npz fitted file with colored body parts and centered camera.
    
#     Args:
#         npz_file: path to .npz SMPL fitted file
#         save_dir: directory to save GIF
#         gif_name: output GIF filename
#     """
#     os.makedirs(save_dir, exist_ok=True)

#     # Load fitted SMPL parameters
#     data = np.load(npz_file)
#     print("Loaded keys from .npz:", list(data.keys()))

#     if "vertices" not in data:
#         raise ValueError("No 'vertices' key found in .npz file")
#     smpl_vertices = data["vertices"]  # (T, 6890, 3)
#     print(f"Total frames to render: {len(smpl_vertices)}")
#     print("First frame vertices shape:", smpl_vertices[0].shape)

#     # Load SMPL faces
#     smpl_faces_path = "./smpl_models/smpl/SMPL/SMPL_NEUTRAL.pkl"  # adjust path if needed
#     import pickle
#     with open(smpl_faces_path, 'rb') as f:
#         smpl_model = pickle.load(f, encoding='latin1')
#         smpl_faces = smpl_model['f']  # (13776, 3)

#     # Define vertex ranges for body parts (example indices)
#     torso_idx = np.arange(0, 3000)          # torso
#     upper_body_idx = np.arange(3000, 5000)  # arms
#     lower_body_idx = np.arange(5000, 6890)  # legs

#     frames = []

#     for frame_idx, verts in enumerate(tqdm(smpl_vertices, desc="Rendering frames")):
#         mesh = trimesh.Trimesh(vertices=verts, faces=smpl_faces, process=False)

#         # Assign colors per body part
#         colors = np.zeros((verts.shape[0], 4), dtype=np.uint8)
#         colors[torso_idx] = [255, 0, 0, 255]       # Red torso
#         colors[upper_body_idx] = [0, 255, 0, 255]  # Green arms
#         colors[lower_body_idx] = [0, 0, 255, 255]  # Blue legs
#         mesh.visual.vertex_colors = colors

#         # Compute center of the body
#         center = verts.mean(axis=0)
#         scale = max(verts.max(axis=0) - verts.min(axis=0))  # approximate body size

#         # Create scene
#         scene = pyrender.Scene(bg_color=[255, 255, 255, 0], ambient_light=[0.3, 0.3, 0.3])
#         mesh_node = pyrender.Mesh.from_trimesh(mesh)
#         scene.add(mesh_node)

#         # Camera positioned along z-axis looking at body center
#         camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
#         camera_distance = scale * 1.5
#         camera_pose = np.array([
#             [1.0, 0.0, 0.0, center[0]],
#             [0.0, 1.0, 0.0, center[1]],
#             [0.0, 0.0, 1.0, center[2] + camera_distance],
#             [0.0, 0.0, 0.0, 1.0]
#         ])
#         scene.add(camera, pose=camera_pose)

#         # Light
#         light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
#         scene.add(light, pose=np.eye(4))

#         # Render frame
#         r = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480)
#         color, _ = r.render(scene)
#         frames.append(color)
#         r.delete()

#     # Save GIF
#     gif_path = os.path.join(save_dir, gif_name)
#     imageio.mimsave(gif_path, frames, fps=25)
#     print(f"Saved rendered SMPL GIF to: {gif_path}")
#     print("Rendering complete!")

# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--npz_file", type=str, required=True, help="Path to SMPL fitted .npz file")
#     parser.add_argument("--save_dir", type=str, default="./outputs/renders_smpl_mesh", help="Directory to save GIF")
#     parser.add_argument("--gif_name", type=str, default="smpl_render.gif", help="GIF filename")
#     args = parser.parse_args()

#     render_smpl_mesh_centered(args.npz_file, args.save_dir, args.gif_name)


# import os
# import numpy as np
# import pyrender
# import trimesh
# import imageio
# from tqdm import tqdm

# def render_smpl_mesh(npz_file, save_dir, gif_name="smpl_render.gif", part_map_file="smpl_vertex_parts.npy"):
#     """
#     Render SMPL mesh with colored body parts and save as GIF.
#     Args:
#         npz_file: path to SMPL fitted .npz file
#         save_dir: directory to save GIF
#         gif_name: output GIF filename
#         part_map_file: .npy file containing vertex-to-body-part mapping
#     """
#     os.makedirs(save_dir, exist_ok=True)

#     # Load fitted SMPL vertices
#     data = np.load(npz_file)
#     if "vertices" in data:
#         smpl_vertices = data["vertices"]  # (T, 6890, 3)
#     else:
#         raise ValueError("No 'vertices' key found in .npz file")

#     # Load SMPL faces
#     smpl_faces_path = "./smpl_models/smpl/SMPL/SMPL_NEUTRAL.pkl"
#     import pickle
#     if not os.path.exists(smpl_faces_path):
#         raise FileNotFoundError(f"SMPL faces file not found: {smpl_faces_path}")
#     with open(smpl_faces_path, 'rb') as f:
#         smpl_model = pickle.load(f, encoding='latin1')
#         smpl_faces = smpl_model['f']  # (13776, 3)

#     # Load body part segmentation map
#     if not os.path.exists(part_map_file):
#         raise FileNotFoundError(f"SMPL vertex-to-part map not found: {part_map_file}")
#     vertex_parts = np.load(part_map_file)  # shape (6890,)

#     # Define colors for body parts
#     color_map = {
#         0: [255, 0, 0, 255],    # torso = red
#         1: [0, 255, 0, 255],    # left arm = green
#         2: [0, 255, 0, 255],    # right arm = green
#         3: [0, 0, 255, 255],    # left leg = blue
#         4: [0, 0, 255, 255],    # right leg = blue
#         5: [255, 255, 0, 255],  # head = yellow
#     }

#     frames = []

#     for i, verts in enumerate(tqdm(smpl_vertices, desc="Rendering frames")):
#         # Center mesh
#         verts_centered = verts - verts.mean(axis=0)

#         # Optional: use transl if exists
#         if 'transl' in data:
#             verts_centered = verts_centered - data['transl'][i]

#         # Create mesh and assign colors
#         mesh = trimesh.Trimesh(vertices=verts_centered, faces=smpl_faces, process=False)
#         colors = np.array([color_map.get(part, [200,200,250,255]) for part in vertex_parts], dtype=np.uint8)
#         mesh.visual.vertex_colors = colors

#         # Create scene
#         scene = pyrender.Scene(bg_color=[255,255,255,0], ambient_light=[0.3,0.3,0.3])
#         mesh_node = pyrender.Mesh.from_trimesh(mesh)
#         scene.add(mesh_node)

#         # Camera (adjusted to show full body)
#         camera = pyrender.PerspectiveCamera(yfov=np.pi/3.0)
#         scene.add(camera, pose=np.array([
#             [1.0, 0.0, 0.0, 0.0],
#             [0.0, 1.0, 0.0, -0.2],  # move camera slightly down
#             [0.0, 0.0, 1.0, 2.5],   # move camera back to see full body
#             [0.0, 0.0, 0.0, 1.0]
#         ]))

#         # Lighting
#         light = pyrender.DirectionalLight(color=[1.0,1.0,1.0], intensity=2.0)
#         scene.add(light, pose=np.eye(4))

#         # Render
#         r = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480)
#         color, _ = r.render(scene)
#         frames.append(color)
#         r.delete()

#     # Save GIF
#     gif_path = os.path.join(save_dir, gif_name)
#     imageio.mimsave(gif_path, frames, fps=25)
#     print(f"Saved rendered SMPL GIF to: {gif_path}")

# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--npz_file", type=str, required=True)
#     parser.add_argument("--save_dir", type=str, default="./outputs/renders_smpl_mesh")
#     parser.add_argument("--gif_name", type=str, default="smpl_render.gif")
#     parser.add_argument("--part_map_file", type=str, default="smpl_vertex_parts.npy")
#     args = parser.parse_args()

#     render_smpl_mesh(args.npz_file, args.save_dir, args.gif_name, args.part_map_file)

# import numpy as np

# vertex_parts = np.zeros(6890, dtype=np.int32)

# # Original part ranges (your current code)
# vertex_parts[0:2000] = 0   # torso
# vertex_parts[2000:2700] = 1  # left arm
# vertex_parts[2700:3400] = 2  # right arm
# vertex_parts[3400:4800] = 3  # left leg
# vertex_parts[4800:6200] = 4  # right leg
# vertex_parts[6200:6890] = 5  # head

# # Merge into 3 groups: 0=torso+head, 1=arms, 2=legs
# merged_parts = np.zeros_like(vertex_parts)

# # torso + head -> 0
# merged_parts[np.isin(vertex_parts, [0, 5])] = 0

# # left arm + right arm -> 1
# merged_parts[np.isin(vertex_parts, [1, 2])] = 1

# # left leg + right leg -> 2
# merged_parts[np.isin(vertex_parts, [3, 4])] = 2
# # Assign colors for the merged groups

# # Save to file
# np.save("smpl_vertex_parts.npy", vertex_parts)
# print("Saved smpl_vertex_parts.npy")
# import os
# import numpy as np
# import pyrender
# import trimesh
# import imageio
# from tqdm import tqdm

# def render_smpl_mesh(npz_file, save_dir, gif_name="smpl_render.gif", part_map_file="smpl_vertex_parts.npy"):
#     """
#     Render SMPL mesh with distinct colored body parts, fully centered, and save as GIF.
#     Args:
#         npz_file: path to SMPL fitted .npz file
#         save_dir: directory to save GIF
#         gif_name: output GIF filename
#         part_map_file: .npy file containing vertex-to-body-part mapping
#     """
#     os.makedirs(save_dir, exist_ok=True)

#     # Load fitted SMPL vertices
#     data = np.load(npz_file)
#     if "vertices" in data:
#         smpl_vertices = data["vertices"]  # (T, 6890, 3)
#     else:
#         raise ValueError("No 'vertices' key found in .npz file")

#     # Load SMPL faces
#     smpl_faces_path = "./smpl_models/smpl/SMPL/SMPL_NEUTRAL.pkl"
#     import pickle
#     if not os.path.exists(smpl_faces_path):
#         raise FileNotFoundError(f"SMPL faces file not found: {smpl_faces_path}")
#     with open(smpl_faces_path, 'rb') as f:
#         smpl_model = pickle.load(f, encoding='latin1')
#         smpl_faces = smpl_model['f']  # (13776, 3)

#     # Load body part segmentation map
#     if not os.path.exists(part_map_file):
#         raise FileNotFoundError(f"SMPL vertex-to-part map not found: {part_map_file}")
#     vertex_parts = np.load(part_map_file)  # shape (6890,)

#     # Define colors for each body part
#     # color_map = {
#     #     0: [0, 0, 255, 255],     # torso = red
#     #     1: [0, 0, 255, 255],     # left arm = green
#     #     2: [0, 0, 255, 255],     # right arm = dark green
#     #     3: [0, 0, 255, 255],     # left leg = blue
#     #     4: [0, 0, 255, 255],     # right leg = dark blue
#     #     5: [0, 0, 255, 255],   # head = yellow
#     #     6: [255, 0, 255, 255],   # left hand
#     #     7: [255, 0, 255, 255],   # right hand
#     #     8: [255, 0, 255, 255],   # left foot
#     #     9: [255, 0, 255, 255],    # right foot
#     # }
#     color_map = {
#     0: [205, 205, 205, 255],   # torso + head
#     1: [120, 160, 235, 255],   # arms + hands
#     2: [120, 205, 160, 255],   # legs + feet
# }

#     frames = []

#     # Compute global center and scale to fit camera
#     all_verts = np.concatenate(smpl_vertices, axis=0)
#     center = all_verts.mean(axis=0)
#     scale = np.max(np.linalg.norm(all_verts - center, axis=1))  # for distance

#     for i, verts in enumerate(tqdm(smpl_vertices, desc="Rendering frames")):
#         # Center and scale mesh
#         verts_centered = (verts - center)

#         # Create mesh and assign vertex colors
#         mesh = trimesh.Trimesh(vertices=verts_centered, faces=smpl_faces, process=False)
#         colors = np.array([color_map.get(part, [200, 200, 250, 255]) for part in vertex_parts], dtype=np.uint8)
#         mesh.visual.vertex_colors = colors

#         # Create scene
#         scene = pyrender.Scene(bg_color=[255, 255, 255, 0], ambient_light=[0.4, 0.4, 0.4])
#         mesh_node = pyrender.Mesh.from_trimesh(mesh)
#         scene.add(mesh_node)

#         # Camera: place back and above for full body view
#         camera = pyrender.PerspectiveCamera(yfov=np.pi/3.0)
#         cam_distance = scale * 2.5
#         camera_pose = np.array([
#             [1.0, 0.0, 0.0, 0.0],
#             [0.0, 1.0, 0.0, -0.2],        # slight downward tilt
#             [0.0, 0.0, 1.0, cam_distance],# back far enough
#             [0.0, 0.0, 0.0, 1.0]
#         ])
#         scene.add(camera, pose=camera_pose)

#         # Lighting
#         light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
#         scene.add(light, pose=np.eye(4))

#         # Render
#         r = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480)
#         color, _ = r.render(scene)
#         frames.append(color)
#         r.delete()

#     # Save GIF
#     gif_path = os.path.join(save_dir, gif_name)
#     imageio.mimsave(gif_path, frames, fps=25)
#     print(f"Saved rendered SMPL GIF to: {gif_path}")


# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--npz_file", type=str, required=True)
#     parser.add_argument("--save_dir", type=str, default="./outputs/renders_smpl_mesh")
#     parser.add_argument("--gif_name", type=str, default="smpl_render.gif")
#     parser.add_argument("--part_map_file", type=str, default="smpl_vertex_parts.npy")
#     args = parser.parse_args()

#     render_smpl_mesh(args.npz_file, args.save_dir, args.gif_name, args.part_map_file)

# import os
# import numpy as np
# import pyrender
# import trimesh
# import imageio
# from tqdm import tqdm

# # ============================
# # Vertex part mapping
# # ============================
# vertex_parts = np.zeros(6890, dtype=np.int32)

# # Original part ranges (your current code)
# vertex_parts[0:2000] = 0   # torso
# vertex_parts[2000:2700] = 1  # left arm
# vertex_parts[2700:3400] = 2  # right arm
# vertex_parts[3400:4800] = 3  # left leg
# vertex_parts[4800:6200] = 4  # right leg
# vertex_parts[6200:6890] = 5  # head

# # Merge into 3 groups: 0=torso+head, 1=arms+hands, 2=legs+feet
# merged_parts = np.zeros_like(vertex_parts)
# merged_parts[np.isin(vertex_parts, [0, 5])] = 0   # torso + head  <<< CHANGED
# merged_parts[np.isin(vertex_parts, [1, 2])] = 1   # arms + hands  <<< CHANGED
# merged_parts[np.isin(vertex_parts, [3, 4])] = 2   # legs + feet   <<< CHANGED

# # Save merged mapping
# np.save("smpl_vertex_parts.npy", vertex_parts)
# print("Saved smpl_vertex_parts.npy")

# # ============================
# # Rendering function
# # ============================
# def render_smpl_mesh(npz_file, save_dir, gif_name="smpl_render.gif", part_map_file="smpl_vertex_parts.npy"):
#     """
#     Render SMPL mesh with distinct colored body parts, fully centered, and save as GIF.
#     """
#     os.makedirs(save_dir, exist_ok=True)

#     # Load fitted SMPL vertices
#     data = np.load(npz_file)
#     if "vertices" in data:
#         smpl_vertices = data["vertices"]  # (T, 6890, 3)
#     else:
#         raise ValueError("No 'vertices' key found in .npz file")

#     # Load SMPL faces
#     smpl_faces_path = "./smpl_models/smpl/SMPL/SMPL_NEUTRAL.pkl"
#     import pickle
#     if not os.path.exists(smpl_faces_path):
#         raise FileNotFoundError(f"SMPL faces file not found: {smpl_faces_path}")
#     with open(smpl_faces_path, 'rb') as f:
#         smpl_model = pickle.load(f, encoding='latin1')
#         smpl_faces = smpl_model['f']  # (13776, 3)

#     # Load vertex-to-part map
#     if not os.path.exists(part_map_file):
#         raise FileNotFoundError(f"SMPL vertex-to-part map not found: {part_map_file}")
#     vertex_parts = np.load(part_map_file)

#     # ============================
#     # New: Assign colors using merged parts
#     # ============================
#     color_map = {
#         0: [205, 205, 205, 255],   # torso + head  <<< CHANGED
#         1: [120, 160, 235, 255],   # arms + hands <<< CHANGED
#         2: [120, 205, 160, 255],   # legs + feet <<< CHANGED
#     }

#     frames = []

#     # Compute global center and scale
#     all_verts = np.concatenate(smpl_vertices, axis=0)
#     center = all_verts.mean(axis=0)
#     scale = np.max(np.linalg.norm(all_verts - center, axis=1))

#     for i, verts in enumerate(tqdm(smpl_vertices, desc="Rendering frames")):
#         verts_centered = verts - center

#         # Create mesh
#         mesh = trimesh.Trimesh(vertices=verts_centered, faces=smpl_faces, process=False)

#         # Assign vertex colors using merged parts  <<< CHANGED
#         vertex_colors = np.zeros((vertices.shape[0], 4), dtype=np.uint8)  # <<< CHANGED
#         for part_id, color in color_map.items():                          # <<< CHANGED
#             vertex_colors[merged_parts == part_id] = color               # <<< CHANGED
#         mesh.visual.vertex_colors = vertex_colors                         # <<< CHANGED

#         # Create scene
#         scene = pyrender.Scene(bg_color=[255, 255, 255, 0], ambient_light=[0.4, 0.4, 0.4])
#         mesh_node = pyrender.Mesh.from_trimesh(mesh)
#         scene.add(mesh_node)

#         # Camera
#         camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
#         cam_distance = scale * 2.5
#         camera_pose = np.array([
#             [1.0, 0.0, 0.0, 0.0],
#             [0.0, 1.0, 0.0, -0.2],
#             [0.0, 0.0, 1.0, cam_distance],
#             [0.0, 0.0, 0.0, 1.0]
#         ])
#         scene.add(camera, pose=camera_pose)

#         # Lighting
#         light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
#         scene.add(light, pose=np.eye(4))

#         # Render frame
#         r = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480)
#         color, _ = r.render(scene)
#         frames.append(color)
#         r.delete()

#     # Save GIF
#     gif_path = os.path.join(save_dir, gif_name)
#     imageio.mimsave(gif_path, frames, fps=25)
#     print(f"Saved rendered SMPL GIF to: {gif_path}")


# # ============================
# # Command-line execution
# # ============================
# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--npz_file", type=str, required=True)
#     parser.add_argument("--save_dir", type=str, default="./outputs/renders_smpl_mesh")
#     parser.add_argument("--gif_name", type=str, default="smpl_render.gif")
#     parser.add_argument("--part_map_file", type=str, default="smpl_vertex_parts.npy")
#     args = parser.parse_args()

#     render_smpl_mesh(args.npz_file, args.save_dir, args.gif_name, args.part_map_file)

# ////////////////////////////// cluade results 

# import os
# import numpy as np
# import pyrender
# import trimesh
# import imageio
# import pickle
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from tqdm import tqdm


# # ============================
# # Vertex part mapping
# # ============================
# def build_vertex_parts():
#     vertex_parts = np.zeros(6890, dtype=np.int32)
#     vertex_parts[0:2000] = 0    # torso
#     vertex_parts[2000:2700] = 1  # left arm
#     vertex_parts[2700:3400] = 2  # right arm
#     vertex_parts[3400:4800] = 3  # left leg
#     vertex_parts[4800:6200] = 4  # right leg
#     vertex_parts[6200:6890] = 5  # head

#     # Merge into 3 groups
#     merged_parts = np.zeros_like(vertex_parts)
#     merged_parts[np.isin(vertex_parts, [0, 5])] = 0   # torso + head
#     merged_parts[np.isin(vertex_parts, [1, 2])] = 1   # arms + hands
#     merged_parts[np.isin(vertex_parts, [3, 4])] = 2   # legs + feet
#     return merged_parts


# # ============================
# # Render single frame
# # ============================
# def render_frame(verts, faces, merged_parts, color_map, scale):
#     mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

#     # FIX: use verts.shape[0] instead of undefined `vertices`
#     vertex_colors = np.zeros((verts.shape[0], 4), dtype=np.uint8)
#     for part_id, color in color_map.items():
#         vertex_colors[merged_parts == part_id] = color
#     mesh.visual.vertex_colors = vertex_colors

#     scene = pyrender.Scene(bg_color=[235, 235, 235, 255], ambient_light=[0.5, 0.5, 0.5])
#     mesh_node = pyrender.Mesh.from_trimesh(mesh)
#     scene.add(mesh_node)

#     camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
#     cam_distance = scale * 2.5
#     camera_pose = np.array([
#         [1.0, 0.0, 0.0,  0.0],
#         [0.0, 1.0, 0.0, -0.2],
#         [0.0, 0.0, 1.0,  cam_distance],
#         [0.0, 0.0, 0.0,  1.0]
#     ])
#     scene.add(camera, pose=camera_pose)

#     light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
#     scene.add(light, pose=np.eye(4))

#     r = pyrender.OffscreenRenderer(viewport_width=320, viewport_height=480)
#     color, _ = r.render(scene)
#     r.delete()
#     return color


# # ============================
# # Save conference-style figure
# # ============================
# def save_conf_figure(frames, caption, save_path, n_shown=4, border_color="#4CAF50"):
#     """
#     Saves a conference-paper-style figure:
#     - n_shown frames arranged side by side
#     - caption below
#     - colored border around the whole figure
#     """
#     # Evenly sample n_shown frames across the sequence
#     indices = np.linspace(0, len(frames) - 1, n_shown, dtype=int)
#     selected = [frames[i] for i in indices]

#     fig_w = 2.5 * n_shown   # inches per frame
#     fig_h = 4.5             # height + caption room

#     fig, axes = plt.subplots(1, n_shown, figsize=(fig_w, fig_h))
#     fig.patch.set_facecolor("#f5f5f5")

#     for ax, frame in zip(axes, selected):
#         ax.imshow(frame)
#         ax.axis("off")

#     plt.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.12, wspace=0.04)

#     # Caption
#     fig.text(
#         0.5, 0.04, caption,
#         ha="center", va="center",
#         fontsize=13, fontstyle="italic",
#         fontfamily="serif",
#         wrap=True
#     )

#     # Draw border rectangle around entire figure
#     border = patches.FancyBboxPatch(
#         (0.01, 0.01), 0.98, 0.98,
#         boxstyle="square,pad=0",
#         linewidth=3,
#         edgecolor=border_color,
#         facecolor="none",
#         transform=fig.transFigure,
#         clip_on=False
#     )
#     fig.add_artist(border)

#     plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
#     plt.close()
#     print(f"Saved conference figure: {save_path}")


# # ============================
# # Main rendering function
# # ============================
# def render_smpl_mesh(
#     npz_file,
#     save_dir,
#     caption="a person rubs their hands together.",
#     gif_name="smpl_render.gif",
#     fig_name="smpl_conf_figure.png",
#     smpl_model_path="./smpl_models/smpl/SMPL/SMPL_NEUTRAL.pkl",
#     n_shown=4,
#     border_color="#4CAF50"
# ):
#     os.makedirs(save_dir, exist_ok=True)

#     # Load vertices
#     data = np.load(npz_file)
#     if "vertices" not in data:
#         raise ValueError("No 'vertices' key found in .npz file")
#     smpl_vertices = data["vertices"]  # (T, 6890, 3)

#     # Load SMPL faces
#     if not os.path.exists(smpl_model_path):
#         raise FileNotFoundError(f"SMPL model not found: {smpl_model_path}")
#     with open(smpl_model_path, 'rb') as f:
#         smpl_model = pickle.load(f, encoding='latin1')
#     smpl_faces = smpl_model['f']  # (13776, 3)

#     # Build part map
#     merged_parts = build_vertex_parts()

#     color_map = {
#         0: [180, 180, 180, 255],   # torso + head  (light grey)
#         1: [100, 140, 220, 255],   # arms + hands  (blue)
#         2: [100, 190, 140, 255],   # legs + feet   (green)
#     }

#     # Global centering
#     all_verts = np.concatenate(smpl_vertices, axis=0)
#     center = all_verts.mean(axis=0)
#     scale = np.max(np.linalg.norm(all_verts - center, axis=1))

#     frames = []
#     for verts in tqdm(smpl_vertices, desc="Rendering frames"):
#         verts_centered = verts - center
#         frame = render_frame(verts_centered, smpl_faces, merged_parts, color_map, scale)
#         frames.append(frame)

#     # Save GIF
#     gif_path = os.path.join(save_dir, gif_name)
#     imageio.mimsave(gif_path, frames, fps=25)
#     print(f"Saved GIF: {gif_path}")

#     # Save conference figure
#     fig_path = os.path.join(save_dir, fig_name)
#     save_conf_figure(frames, caption, fig_path, n_shown=n_shown, border_color=border_color)


# # ============================
# # CLI
# # ============================
# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--npz_file",     type=str, required=True)
#     parser.add_argument("--save_dir",     type=str, default="./outputs/renders_smpl_mesh")
#     parser.add_argument("--caption",      type=str, default="The man walked up to the door and knocked on it.")
#     parser.add_argument("--gif_name",     type=str, default="smpl_render.gif")
#     parser.add_argument("--fig_name",     type=str, default="smpl_conf_figure.png")
#     parser.add_argument("--smpl_model",   type=str, default="./smpl_models/smpl/SMPL/SMPL_NEUTRAL.pkl")
#     parser.add_argument("--n_shown",      type=int, default=4,       help="Number of frames to show in figure")
#     parser.add_argument("--border_color", type=str, default="#4CAF50", help="Border color hex")
#     args = parser.parse_args()

#     render_smpl_mesh(
#         npz_file=args.npz_file,
#         save_dir=args.save_dir,
#         caption=args.caption,
#         gif_name=args.gif_name,
#         fig_name=args.fig_name,
#         smpl_model_path=args.smpl_model,
#         n_shown=args.n_shown,
#         border_color=args.border_color
#     )
# ///////////////////////////////////////////////////////////////  good view for paper 
# import os
# import numpy as np
# import pyrender
# import trimesh
# import imageio
# import pickle
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from tqdm import tqdm


# # ============================
# # CORRECT SMPL vertex part mapping
# # Using official SMPL body part segmentation
# # ============================
# def build_vertex_parts(smpl_model_path):
#     """
#     Build vertex-to-part mapping using SMPL's official part segmentation.
#     Returns merged_parts array: 0=torso+head, 1=arms+hands, 2=legs+feet
#     """
#     with open(smpl_model_path, 'rb') as f:
#         smpl_model = pickle.load(f, encoding='latin1')

#     # SMPL has a vertex part segmentation stored as 'part_segm' or we derive it
#     # Official SMPL 24 joints -> body part groups:
#     # Torso+Head joints:  0(pelvis),1(L_hip),2(R_hip),3(spine1),6(spine2),
#     #                     9(spine3),12(neck),15(head)
#     # Arm joints:         13(L_collar),14(R_collar),16(L_shoulder),17(R_shoulder),
#     #                     18(L_elbow),19(R_elbow),20(L_wrist),21(R_wrist),
#     #                     22(L_hand),23(R_hand)
#     # Leg joints:         4(L_knee),5(R_knee),7(L_ankle),8(R_ankle),
#     #                     10(L_foot),11(R_foot)

#     # Use LBS weights to assign each vertex to its dominant joint
#     lbs_weights = smpl_model['weights']  # (6890, 24)
#     dominant_joint = np.argmax(lbs_weights, axis=1)  # (6890,)

#     TORSO_HEAD_JOINTS = {0, 1, 2, 3, 6, 9, 12, 15}
#     ARM_JOINTS        = {13, 14, 16, 17, 18, 19, 20, 21, 22, 23}
#     LEG_JOINTS        = {4, 5, 7, 8, 10, 11}

#     merged_parts = np.zeros(6890, dtype=np.int32)
#     for v in range(6890):
#         j = dominant_joint[v]
#         if j in ARM_JOINTS:
#             merged_parts[v] = 1
#         elif j in LEG_JOINTS:
#             merged_parts[v] = 2
#         else:
#             merged_parts[v] = 0  # torso + head

#     return merged_parts


# # ============================
# # Render single frame
# # ============================
# def render_frame(verts, faces, merged_parts, color_map, scale):
#     mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

#     vertex_colors = np.zeros((verts.shape[0], 4), dtype=np.uint8)
#     for part_id, color in color_map.items():
#         vertex_colors[merged_parts == part_id] = color
#     mesh.visual.vertex_colors = vertex_colors

#     scene = pyrender.Scene(bg_color=[235, 235, 235, 255], ambient_light=[0.5, 0.5, 0.5])
#     scene.add(pyrender.Mesh.from_trimesh(mesh))

#     camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
#     cam_distance = scale * 2.5
#     camera_pose = np.array([
#         [1.0, 0.0, 0.0,  0.0],
#         [0.0, 1.0, 0.0, -0.2],
#         [0.0, 0.0, 1.0,  cam_distance],
#         [0.0, 0.0, 0.0,  1.0]
#     ])
#     scene.add(camera, pose=camera_pose)

#     light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
#     scene.add(light, pose=np.eye(4))

#     r = pyrender.OffscreenRenderer(viewport_width=320, viewport_height=480)
#     color, _ = r.render(scene)
#     r.delete()
#     return color


# # ============================
# # Save conference-style figure
# # ============================
# def save_conf_figure(frames, caption, save_path, n_shown=4, border_color="#4CAF50"):
#     indices = np.linspace(0, len(frames) - 1, n_shown, dtype=int)
#     selected = [frames[i] for i in indices]

#     fig_w = 2.5 * n_shown
#     fig_h = 4.5

#     fig, axes = plt.subplots(1, n_shown, figsize=(fig_w, fig_h))
#     fig.patch.set_facecolor("#f5f5f5")

#     for ax, frame in zip(axes, selected):
#         ax.imshow(frame)
#         ax.axis("off")

#     plt.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.12, wspace=0.04)

#     fig.text(
#         0.5, 0.04, caption,
#         ha="center", va="center",
#         fontsize=13, fontstyle="italic",
#         fontfamily="serif", wrap=True
#     )

#     border = patches.FancyBboxPatch(
#         (0.01, 0.01), 0.98, 0.98,
#         boxstyle="square,pad=0",
#         linewidth=3,
#         edgecolor=border_color,
#         facecolor="none",
#         transform=fig.transFigure,
#         clip_on=False
#     )
#     fig.add_artist(border)

#     plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
#     plt.close()
#     print(f"Saved conference figure: {save_path}")


# # ============================
# # Main rendering function
# # ============================
# def render_smpl_mesh(
#     npz_file,
#     save_dir,
#     caption="The man walked up to the door and knocked on it.",
#     gif_name="smpl_render.gif",
#     fig_name="smpl_conf_figure.png",
#     smpl_model_path="./smpl_models/smpl/SMPL/SMPL_NEUTRAL.pkl",
#     n_shown=4,
#     border_color="#4CAF50"
# ):
#     os.makedirs(save_dir, exist_ok=True)

#     # Load vertices
#     data = np.load(npz_file)
#     if "vertices" not in data:
#         raise ValueError("No 'vertices' key found in .npz file")
#     smpl_vertices = data["vertices"]  # (T, 6890, 3)

#     # Load SMPL faces + build correct part map from LBS weights
#     with open(smpl_model_path, 'rb') as f:
#         smpl_model = pickle.load(f, encoding='latin1')
#     smpl_faces = smpl_model['f']

#     merged_parts = build_vertex_parts(smpl_model_path)

#     # ── Color choices (pick one set below) ──────────────────────────────
#     # Option A: Uniform blue (like your reference image)
#     # color_map = {0: [100,149,237,255], 1: [100,149,237,255], 2: [100,149,237,255]}

#     # Option B: Dark saturated 3-part (recommended for paper)
#     color_map = {
#         0: [ 52,  95, 160, 255],   # torso + head  →  deep navy blue
#         1: [180,  50,  50, 255],   # arms + hands  →  dark red
#         2: [ 34, 120,  80, 255],   # legs + feet   →  dark green
#     }

#     # Option C: High contrast
#     # color_map = {
#     #     0: [220, 100,  30, 255],   # torso + head  →  burnt orange
#     #     1: [ 30, 160, 180, 255],   # arms + hands  →  teal
#     #     2: [130,  50, 180, 255],   # legs + feet   →  purple
#     # }
#     # ────────────────────────────────────────────────────────────────────

#     # Global centering
#     all_verts = np.concatenate(smpl_vertices, axis=0)
#     center = all_verts.mean(axis=0)
#     scale = np.max(np.linalg.norm(all_verts - center, axis=1))

#     frames = []
#     for verts in tqdm(smpl_vertices, desc="Rendering frames"):
#         frame = render_frame(verts - center, smpl_faces, merged_parts, color_map, scale)
#         frames.append(frame)

#     # Save GIF
#     gif_path = os.path.join(save_dir, gif_name)
#     imageio.mimsave(gif_path, frames, fps=25)
#     print(f"Saved GIF: {gif_path}")

#     # Save conference figure
#     fig_path = os.path.join(save_dir, fig_name)
#     save_conf_figure(frames, caption, fig_path, n_shown=n_shown, border_color=border_color)


# # ============================
# # CLI
# # ============================
# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--npz_file",     type=str, required=True)
#     parser.add_argument("--save_dir",     type=str, default="./outputs/renders_smpl_mesh")
#     parser.add_argument("--caption",      type=str, default="The man walked up to the door and knocked on it.")
#     parser.add_argument("--gif_name",     type=str, default="smpl_render.gif")
#     parser.add_argument("--fig_name",     type=str, default="smpl_conf_figure.png")
#     parser.add_argument("--smpl_model",   type=str, default="./smpl_models/smpl/SMPL/SMPL_NEUTRAL.pkl")
#     parser.add_argument("--n_shown",      type=int, default=4)
#     parser.add_argument("--border_color", type=str, default="#4CAF50")
#     args = parser.parse_args()

#     render_smpl_mesh(
#         npz_file=args.npz_file,
#         save_dir=args.save_dir,
#         caption=args.caption,
#         gif_name=args.gif_name,
#         fig_name=args.fig_name,
#         smpl_model_path=args.smpl_model,
#         n_shown=args.n_shown,
#         border_color=args.border_color
#     )

# import os
# import numpy as np
# import pyrender
# import trimesh
# import imageio
# import pickle
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from tqdm import tqdm


# # ============================
# # SMPL vertex part mapping via LBS weights
# # ============================
# def build_vertex_parts(smpl_model):
#     lbs_weights    = smpl_model['weights']
#     dominant_joint = np.argmax(lbs_weights, axis=1)

#     TORSO_HEAD = {0, 1, 2, 3, 6, 9, 12, 15}
#     ARM_JOINTS  = {13, 14, 16, 17, 18, 19, 20, 21, 22, 23}
#     LEG_JOINTS  = {4, 5, 7, 8, 10, 11}

#     merged = np.zeros(6890, dtype=np.int32)
#     for v in range(6890):
#         j = dominant_joint[v]
#         if j in ARM_JOINTS:
#             merged[v] = 1
#         elif j in LEG_JOINTS:
#             merged[v] = 2
#     return merged


# # ============================
# # Color map
# # ============================
# COLOR_MAP = {
#     0: [ 52,  95, 160, 255],   # torso + head  →  deep navy
#     1: [180,  50,  50, 255],   # arms + hands  →  dark red
#     2: [ 34, 120,  80, 255],   # legs + feet   →  dark green
# }


# # ============================
# # Build colored trimesh for one pose
# # ============================
# def build_mesh(verts, faces, merged_parts, alpha=255):
#     mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
#     vertex_colors = np.zeros((verts.shape[0], 4), dtype=np.uint8)
#     for part_id, color in COLOR_MAP.items():
#         c = list(color)
#         c[3] = alpha                               # control transparency
#         vertex_colors[merged_parts == part_id] = c
#     mesh.visual.vertex_colors = vertex_colors
#     return mesh


# # ============================
# # Render ghost trail — multiple poses in ONE scene
# # ============================
# def render_ghost_frame(
#     all_verts,       # list of (6890,3) arrays — the selected poses
#     faces,
#     merged_parts,
#     scale,
#     spread_axis='x', # axis to spread ghosts along: 'x' or 'z'
#     spread=0.6,      # distance between ghosts (relative to scale)
#     viewport_w=900,
#     viewport_h=500,
# ):
#     """
#     Place all poses in one scene, spread along `spread_axis`,
#     with earlier poses more transparent (ghost effect).
#     """
#     n = len(all_verts)
#     scene = pyrender.Scene(bg_color=[230, 230, 230, 255], ambient_light=[0.6, 0.6, 0.6])

#     for i, verts in enumerate(all_verts):
#         # Spread ghosts along chosen axis
#         offset = np.zeros(3)
#         offset[{'x': 0, 'y': 1, 'z': 2}[spread_axis]] = (i - (n - 1) / 2.0) * spread * scale

#         # Earlier ghosts more transparent
#         alpha = int(80 + 175 * (i / (n - 1)))   # 80 → 255

#         mesh = build_mesh(verts + offset, faces, merged_parts, alpha=alpha)
#         scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=True))

#     # Camera — pull back to see all ghosts
#     cam_distance = scale * (2.5 + 0.6 * n)
#     camera_pose = np.array([
#         [1.0, 0.0, 0.0,  0.0],
#         [0.0, 1.0, 0.0, -0.2],
#         [0.0, 0.0, 1.0,  cam_distance],
#         [0.0, 0.0, 0.0,  1.0]
#     ])
#     scene.add(pyrender.PerspectiveCamera(yfov=np.pi / 3.5), pose=camera_pose)

#     # Lights from multiple angles for depth
#     for light_pose in [
#         np.eye(4),
#         np.array([[1,0,0,1],[0,1,0,0.5],[0,0,1,1],[0,0,0,1]], dtype=float),
#     ]:
#         scene.add(pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.5), pose=light_pose)

#     r = pyrender.OffscreenRenderer(viewport_width=viewport_w, viewport_height=viewport_h)
#     color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
#     r.delete()
#     return color


# # ============================
# # Save conference figure with ghost trail
# # ============================
# def save_ghost_conf_figure(ghost_frame, caption, save_path, border_color="#4CAF50"):
#     fig, ax = plt.subplots(1, 1, figsize=(10, 5.5))
#     fig.patch.set_facecolor("#f0f0f0")

#     ax.imshow(ghost_frame)
#     ax.axis("off")

#     fig.text(
#         0.5, 0.04, caption,
#         ha="center", va="center",
#         fontsize=13, fontstyle="italic",
#         fontfamily="serif"
#     )

#     border = patches.FancyBboxPatch(
#         (0.01, 0.01), 0.98, 0.98,
#         boxstyle="square,pad=0", linewidth=3,
#         edgecolor=border_color, facecolor="none",
#         transform=fig.transFigure, clip_on=False
#     )
#     fig.add_artist(border)

#     plt.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.10)
#     plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
#     plt.close()
#     print(f"Saved ghost figure: {save_path}")


# # ============================
# # Full pipeline for one query
# # ============================
# def render_query_ghost(
#     npz_file,
#     save_dir,
#     smpl_faces,
#     merged_parts,
#     caption,
#     gif_name,
#     fig_name,
#     n_ghosts=4,          # how many poses to show in the ghost trail
#     spread=0.55,         # spacing between ghosts (fraction of scale)
#     spread_axis='x',     # 'x' = side by side, 'z' = depth
# ):
#     os.makedirs(save_dir, exist_ok=True)

#     data = np.load(npz_file)
#     if "vertices" not in data:
#         raise ValueError(f"No 'vertices' key in {npz_file}")
#     smpl_vertices = data["vertices"]   # (T, 6890, 3)

#     # Global centering & scale
#     all_v  = np.concatenate(smpl_vertices, axis=0)
#     center = all_v.mean(axis=0)
#     scale  = np.max(np.linalg.norm(all_v - center, axis=1))
#     smpl_vertices_centered = smpl_vertices - center[None]

#     # ── 1. Save GIF (normal per-frame render) ──────────────────────
#     print(f"Rendering GIF frames for {os.path.basename(npz_file)} ...")
#     gif_frames = []
#     for verts in tqdm(smpl_vertices_centered, desc="GIF frames"):
#         mesh = build_mesh(verts, smpl_faces, merged_parts, alpha=255)
#         scene = pyrender.Scene(bg_color=[230, 230, 230, 255], ambient_light=[0.5, 0.5, 0.5])
#         scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=True))
#         cam_distance = scale * 2.5
#         camera_pose = np.array([
#             [1,0,0,0],[0,1,0,-0.2],[0,0,1,cam_distance],[0,0,0,1]
#         ], dtype=float)
#         scene.add(pyrender.PerspectiveCamera(yfov=np.pi/3.0), pose=camera_pose)
#         scene.add(pyrender.DirectionalLight(color=[1,1,1], intensity=3.0), pose=np.eye(4))
#         r = pyrender.OffscreenRenderer(640, 480)
#         color, _ = r.render(scene)
#         r.delete()
#         gif_frames.append(color)

#     gif_path = os.path.join(save_dir, gif_name)
#     imageio.mimsave(gif_path, gif_frames, fps=25)
#     print(f"Saved GIF: {gif_path}")

#     # ── 2. Ghost trail figure ──────────────────────────────────────
#     # Evenly sample n_ghosts frames across the sequence
#     indices      = np.linspace(0, len(smpl_vertices_centered) - 1, n_ghosts, dtype=int)
#     ghost_verts  = [smpl_vertices_centered[i] for i in indices]

#     ghost_img = render_ghost_frame(
#         ghost_verts, smpl_faces, merged_parts, scale,
#         spread_axis=spread_axis, spread=spread
#     )

#     fig_path = os.path.join(save_dir, fig_name)
#     save_ghost_conf_figure(ghost_img, caption, fig_path)
#     print(f"Saved ghost figure: {fig_path}")

#     return ghost_img, caption


# # ============================
# # Combined multi-query figure (one row per query)
# # ============================
# def save_combined_figure(rows, save_path, border_color="#4CAF50"):
#     """rows: list of (image_array, caption_str)"""
#     n = len(rows)
#     fig, axes = plt.subplots(n, 1, figsize=(10, 5.5 * n))
#     fig.patch.set_facecolor("#f0f0f0")
#     if n == 1:
#         axes = [axes]

#     for i, (img, caption) in enumerate(rows):
#         axes[i].imshow(img)
#         axes[i].axis("off")
#         axes[i].set_title(caption, fontsize=12, fontstyle="italic",
#                           fontfamily="serif", pad=6)

#     plt.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.03, hspace=0.12)

#     border = patches.FancyBboxPatch(
#         (0.01, 0.01), 0.98, 0.98,
#         boxstyle="square,pad=0", linewidth=3,
#         edgecolor=border_color, facecolor="none",
#         transform=fig.transFigure, clip_on=False
#     )
#     fig.add_artist(border)

#     plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
#     plt.close()
#     print(f"Saved combined figure: {save_path}")


# # ============================
# # Main
# # ============================
# def main():
#     SMPL_MODEL_PATH = "./smpl_models/smpl/SMPL/SMPL_NEUTRAL.pkl"
#     COMBINED_FIG    = "outputs/renders_smpl/test/combined_ghost_figure.png"

#     queries = [
#         {
#             "npz_file":    "outputs/renders_smpl/test/75/M013735_smpl_fit.npz",
#             "save_dir":    "outputs/renders_smpl/test/75",
#             "gif_name":    "M013735_mesh.gif",
#             "fig_name":    "M013735_ghost.png",
#             "caption":     "The man walked up to the door and knocked on it.",
#             "n_ghosts":    4,
#             "spread":      0.55,
#             "spread_axis": "x",
#         },
#         {
#             "npz_file":    "outputs/renders_smpl/test/135/001840_smpl_fit.npz",
#             "save_dir":    "outputs/renders_smpl/test/135",
#             "gif_name":    "001840_mesh.gif",
#             "fig_name":    "001840_ghost.png",
#             "caption":     "A person sits down on a chair.",   # ← your text prompt
#             "n_ghosts":    4,
#             "spread":      0.55,
#             "spread_axis": "x",
#         },
#     ]

#     # Load SMPL once
#     with open(SMPL_MODEL_PATH, 'rb') as f:
#         smpl_model = pickle.load(f, encoding='latin1')
#     smpl_faces   = smpl_model['f']
#     merged_parts = build_vertex_parts(smpl_model)

#     rows = []
#     for q in queries:
#         ghost_img, caption = render_query_ghost(
#             npz_file    = q["npz_file"],
#             save_dir    = q["save_dir"],
#             smpl_faces  = smpl_faces,
#             merged_parts= merged_parts,
#             caption     = q["caption"],
#             gif_name    = q["gif_name"],
#             fig_name    = q["fig_name"],
#             n_ghosts    = q["n_ghosts"],
#             spread      = q["spread"],
#             spread_axis = q["spread_axis"],
#         )
#         rows.append((ghost_img, caption))

#     # Combined paper figure
#     os.makedirs(os.path.dirname(COMBINED_FIG), exist_ok=True)
#     save_combined_figure(rows, COMBINED_FIG)


# if __name__ == "__main__":
#     main()

# import os
# import numpy as np
# import pyrender
# import trimesh
# import imageio
# import pickle
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from tqdm import tqdm


# # ============================
# # FIXED vertex part mapping
# # ============================
# def build_vertex_parts(smpl_model):
#     lbs_weights    = smpl_model['weights']       # (6890, 24)
#     dominant_joint = np.argmax(lbs_weights, axis=1)

#     # SMPL 24 joints — COMPLETE leg set including lower leg + feet
#     TORSO_HEAD = {0, 3, 6, 9, 12, 15}           # spine chain + neck + head
#     ARM_JOINTS  = {13, 14, 16, 17, 18, 19, 20, 21, 22, 23}  # collars→hands
#     # FIX: include hip joints (1,2) AND knee/ankle/foot (4,5,7,8,10,11)
#     LEG_JOINTS  = {1, 2, 4, 5, 7, 8, 10, 11}    # hips + knees + ankles + feet

#     merged = np.zeros(6890, dtype=np.int32)
#     for v in range(6890):
#         j = dominant_joint[v]
#         if j in ARM_JOINTS:
#             merged[v] = 1   # red
#         elif j in LEG_JOINTS:
#             merged[v] = 2   # green
#         else:
#             merged[v] = 0   # blue (torso + head)
#     return merged


# # ============================
# # Color map
# # ============================
# COLOR_MAP = {
#     0: [ 52,  95, 160, 255],   # torso + head  → deep navy blue
#     1: [180,  50,  50, 255],   # arms + hands  → dark red
#     2: [ 34, 139,  80, 255],   # legs + feet   → dark green
# }


# # ============================
# # Build colored trimesh
# # ============================
# def build_mesh(verts, faces, merged_parts, alpha=255):
#     mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
#     vertex_colors = np.zeros((verts.shape[0], 4), dtype=np.uint8)
#     for part_id, color in COLOR_MAP.items():
#         c = list(color)
#         c[3] = alpha
#         vertex_colors[merged_parts == part_id] = c
#     mesh.visual.vertex_colors = vertex_colors
#     return mesh


# # ============================
# # Render ghost trail — all poses in ONE scene
# # ============================
# def render_ghost_frame(
#     all_verts,
#     faces,
#     merged_parts,
#     scale,
#     spread=0.55,
#     spread_axis='x',
#     viewport_w=1200,
#     viewport_h=600,
#     alpha_start=60,      # transparency of first/earliest ghost
#     alpha_end=255,       # fully solid for last/latest pose
# ):
#     n = len(all_verts)
#     scene = pyrender.Scene(
#         bg_color=[230, 230, 230, 255],
#         ambient_light=[0.55, 0.55, 0.55]
#     )

#     axis_idx = {'x': 0, 'y': 1, 'z': 2}[spread_axis]

#     for i, verts in enumerate(all_verts):
#         offset      = np.zeros(3)
#         offset[axis_idx] = (i - (n - 1) / 2.0) * spread * scale

#         # Fade: earliest ghost = alpha_start, latest = alpha_end
#         t     = i / max(n - 1, 1)
#         alpha = int(alpha_start + (alpha_end - alpha_start) * t)

#         mesh = build_mesh(verts + offset, faces, merged_parts, alpha=alpha)
#         scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=True))

#     # Camera pulled back enough to fit all ghosts
#     cam_distance = scale * (2.2 + 0.5 * n)
#     camera_pose  = np.array([
#         [1, 0, 0,  0.0],
#         [0, 1, 0, -0.15],
#         [0, 0, 1,  cam_distance],
#         [0, 0, 0,  1.0]
#     ], dtype=float)
#     scene.add(pyrender.PerspectiveCamera(yfov=np.pi / 3.5), pose=camera_pose)

#     for lp in [np.eye(4),
#                np.array([[1,0,0,1.5],[0,1,0,1],[0,0,1,1],[0,0,0,1]], dtype=float)]:
#         scene.add(pyrender.DirectionalLight(color=[1,1,1], intensity=2.5), pose=lp)

#     r = pyrender.OffscreenRenderer(viewport_width=viewport_w, viewport_height=viewport_h)
#     color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
#     r.delete()
#     return color


# # ============================
# # Save single ghost figure
# # ============================
# def save_ghost_conf_figure(ghost_frame, caption, save_path, border_color="#4CAF50"):
#     fig, ax = plt.subplots(1, 1, figsize=(12, 6))
#     fig.patch.set_facecolor("#f0f0f0")
#     ax.imshow(ghost_frame)
#     ax.axis("off")

#     fig.text(0.5, 0.03, caption,
#              ha="center", va="center",
#              fontsize=13, fontstyle="italic", fontfamily="serif")

#     border = patches.FancyBboxPatch(
#         (0.01, 0.01), 0.98, 0.98,
#         boxstyle="square,pad=0", linewidth=3,
#         edgecolor=border_color, facecolor="none",
#         transform=fig.transFigure, clip_on=False
#     )
#     fig.add_artist(border)
#     plt.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.08)
#     plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
#     plt.close()
#     print(f"Saved: {save_path}")


# # ============================
# # Full pipeline for one query
# # ============================
# def render_query_ghost(
#     npz_file,
#     save_dir,
#     smpl_faces,
#     merged_parts,
#     caption,
#     gif_name,
#     fig_name,
#     n_ghosts=8,          # ← more ghosts = more of the sequence visible
#     spread=0.50,
#     spread_axis='x',
#     alpha_start=60,
#     alpha_end=255,
# ):
#     os.makedirs(save_dir, exist_ok=True)

#     data = np.load(npz_file)
#     if "vertices" not in data:
#         raise ValueError(f"No 'vertices' key in {npz_file}")
#     smpl_vertices = data["vertices"]   # (T, 6890, 3)
#     T = len(smpl_vertices)

#     # Global centering
#     all_v  = np.concatenate(smpl_vertices, axis=0)
#     center = all_v.mean(axis=0)
#     scale  = np.max(np.linalg.norm(all_v - center, axis=1))
#     verts_c = smpl_vertices - center[None]

#     # ── GIF: every frame ──────────────────────────────────────────
#     print(f"\nRendering GIF ({T} frames) for {os.path.basename(npz_file)} ...")
#     gif_frames = []
#     for verts in tqdm(verts_c, desc="GIF"):
#         mesh  = build_mesh(verts, smpl_faces, merged_parts, alpha=255)
#         scene = pyrender.Scene(bg_color=[230,230,230,255], ambient_light=[0.5,0.5,0.5])
#         scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=True))
#         cp = np.array([[1,0,0,0],[0,1,0,-0.2],[0,0,1,scale*2.5],[0,0,0,1]], dtype=float)
#         scene.add(pyrender.PerspectiveCamera(yfov=np.pi/3.0), pose=cp)
#         scene.add(pyrender.DirectionalLight(color=[1,1,1], intensity=3.0), pose=np.eye(4))
#         r = pyrender.OffscreenRenderer(640, 480)
#         color, _ = r.render(scene)
#         r.delete()
#         gif_frames.append(color)

#     gif_path = os.path.join(save_dir, gif_name)
#     imageio.mimsave(gif_path, gif_frames, fps=25)
#     print(f"Saved GIF: {gif_path}")

#     # ── Ghost figure: evenly sample n_ghosts across FULL sequence ──
#     # Always include frame 0 (start) and frame T-1 (end)
#     if n_ghosts >= T:
#         indices = list(range(T))
#     else:
#         # Force first and last, fill middle evenly
#         middle  = np.linspace(0, T - 1, n_ghosts, dtype=int).tolist()
#         indices = sorted(set([0] + middle + [T - 1]))[:n_ghosts]

#     print(f"Ghost frames: {indices}")
#     ghost_verts = [verts_c[i] for i in indices]

#     ghost_img = render_ghost_frame(
#         ghost_verts, smpl_faces, merged_parts, scale,
#         spread=spread, spread_axis=spread_axis,
#         alpha_start=alpha_start, alpha_end=alpha_end,
#     )

#     fig_path = os.path.join(save_dir, fig_name)
#     save_ghost_conf_figure(ghost_img, caption, fig_path)

#     return ghost_img, caption


# # ============================
# # Combined multi-query figure
# # ============================
# def save_combined_figure(rows, save_path, border_color="#4CAF50"):
#     n   = len(rows)
#     fig, axes = plt.subplots(n, 1, figsize=(12, 6 * n))
#     fig.patch.set_facecolor("#f0f0f0")
#     if n == 1:
#         axes = [axes]

#     for i, (img, caption) in enumerate(rows):
#         axes[i].imshow(img)
#         axes[i].axis("off")
#         axes[i].set_title(caption, fontsize=12, fontstyle="italic",
#                           fontfamily="serif", pad=6)

#     plt.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.03, hspace=0.10)

#     border = patches.FancyBboxPatch(
#         (0.01, 0.01), 0.98, 0.98,
#         boxstyle="square,pad=0", linewidth=3,
#         edgecolor=border_color, facecolor="none",
#         transform=fig.transFigure, clip_on=False
#     )
#     fig.add_artist(border)
#     plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
#     plt.close()
#     print(f"Saved combined: {save_path}")


# # ============================
# # Main — define your queries here
# # ============================
# def main():
#     SMPL_MODEL_PATH = "./smpl_models/smpl/SMPL/SMPL_NEUTRAL.pkl"
#     COMBINED_FIG    = "outputs/renders_smpl/test/combined_ghost_figure.png"

#     queries = [
#         {
#             "npz_file":    "outputs/renders_smpl/test/75/M013735_smpl_fit.npz",
#             "save_dir":    "outputs/renders_smpl/test/75",
#             "gif_name":    "M013735_mesh.gif",
#             "fig_name":    "M013735_ghost.png",
#             "caption":     "The man walked up to the door and knocked on it.",
#             "n_ghosts":    8,      # ← increase to show more of the motion
#             "spread":      0.50,
#             "spread_axis": "x",
#             "alpha_start": 60,
#             "alpha_end":   255,
#         },
#         {
#             "npz_file":    "outputs/renders_smpl/test/135/001840_smpl_fit.npz",
#             "save_dir":    "outputs/renders_smpl/test/135",
#             "gif_name":    "001840_mesh.gif",
#             "fig_name":    "001840_ghost.png",
#             "caption":     "A person sits down on a chair.",
#             "n_ghosts":    8,
#             "spread":      0.50,
#             "spread_axis": "x",
#             "alpha_start": 60,
#             "alpha_end":   255,
#         },
#     ]

#     with open(SMPL_MODEL_PATH, 'rb') as f:
#         smpl_model = pickle.load(f, encoding='latin1')
#     smpl_faces   = smpl_model['f']
#     merged_parts = build_vertex_parts(smpl_model)

#     rows = []
#     for q in queries:
#         ghost_img, caption = render_query_ghost(
#             npz_file     = q["npz_file"],
#             save_dir     = q["save_dir"],
#             smpl_faces   = smpl_faces,
#             merged_parts = merged_parts,
#             caption      = q["caption"],
#             gif_name     = q["gif_name"],
#             fig_name     = q["fig_name"],
#             n_ghosts     = q["n_ghosts"],
#             spread       = q["spread"],
#             spread_axis  = q["spread_axis"],
#             alpha_start  = q["alpha_start"],
#             alpha_end    = q["alpha_end"],
#         )
#         rows.append((ghost_img, caption))

#     os.makedirs(os.path.dirname(COMBINED_FIG), exist_ok=True)
#     save_combined_figure(rows, COMBINED_FIG)


# if __name__ == "__main__":
#     main()

# import os
# import numpy as np
# import pyrender
# import trimesh
# import imageio
# import pickle
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from tqdm import tqdm


# # ============================
# # FIXED vertex part mapping
# # ============================
# def build_vertex_parts(smpl_model):
#     lbs_weights    = smpl_model['weights']
#     dominant_joint = np.argmax(lbs_weights, axis=1)

#     TORSO_HEAD = {0, 3, 6, 9, 12, 15}
#     ARM_JOINTS  = {13, 14, 16, 17, 18, 19, 20, 21, 22, 23}
#     LEG_JOINTS  = {1, 2, 4, 5, 7, 8, 10, 11}

#     merged = np.zeros(6890, dtype=np.int32)
#     for v in range(6890):
#         j = dominant_joint[v]
#         if j in ARM_JOINTS:
#             merged[v] = 1
#         elif j in LEG_JOINTS:
#             merged[v] = 2
#         else:
#             merged[v] = 0
#     return merged


# COLOR_MAP = {
#     0: [ 52,  95, 160, 255],
#     1: [180,  50,  50, 255],
#     2: [ 34, 139,  80, 255],
# }


# def build_mesh(verts, faces, merged_parts, alpha=255):
#     mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
#     vertex_colors = np.zeros((verts.shape[0], 4), dtype=np.uint8)
#     for part_id, color in COLOR_MAP.items():
#         c = list(color)
#         c[3] = alpha
#         vertex_colors[merged_parts == part_id] = c
#     mesh.visual.vertex_colors = vertex_colors
#     return mesh


# # ============================
# # Alpha curve: solid → faded → solid
# # First and last frame fully opaque,
# # middle frames use a sine curve dip
# # ============================
# def compute_alphas(n, alpha_min=45, alpha_max=255):
#     """
#     Returns array of n alpha values.
#     Frame 0 and frame n-1 = alpha_max (fully solid).
#     Middle frames follow a sine dip down to alpha_min.
#     """
#     alphas = []
#     for i in range(n):
#         t = i / max(n - 1, 1)           # 0.0 → 1.0
#         # sine curve: peaks at 0 and 1, dips in the middle
#         sine_val = np.sin(t * np.pi)    # 0 → 1 → 0
#         alpha = int(alpha_max - (alpha_max - alpha_min) * sine_val)
#         alphas.append(alpha)
#     return alphas


# # ============================
# # Render ALL frames as ghost trail in ONE scene
# # ============================
# def render_ghost_frame(
#     all_verts,
#     faces,
#     merged_parts,
#     scale,
#     spread=0.18,         # tighter spacing since we have many frames
#     spread_axis='x',
#     viewport_w=1600,
#     viewport_h=620,
#     alpha_min=45,
#     alpha_max=255,
# ):
#     n        = len(all_verts)
#     alphas   = compute_alphas(n, alpha_min=alpha_min, alpha_max=alpha_max)
#     axis_idx = {'x': 0, 'y': 1, 'z': 2}[spread_axis]

#     scene = pyrender.Scene(
#         bg_color=[230, 230, 230, 255],
#         ambient_light=[0.55, 0.55, 0.55]
#     )

#     for i, (verts, alpha) in enumerate(zip(all_verts, alphas)):
#         offset           = np.zeros(3)
#         offset[axis_idx] = (i - (n - 1) / 2.0) * spread * scale
#         mesh             = build_mesh(verts + offset, faces, merged_parts, alpha=alpha)
#         scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=True))

#     # Camera: wide enough to frame all ghosts
#     total_spread  = (n - 1) * spread * scale
#     cam_distance  = max(scale * 2.8, total_spread * 1.1)
#     camera_pose   = np.array([
#         [1, 0, 0,  0.0],
#         [0, 1, 0, -0.15],
#         [0, 0, 1,  cam_distance],
#         [0, 0, 0,  1.0]
#     ], dtype=float)
#     scene.add(
#         pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=viewport_w / viewport_h),
#         pose=camera_pose
#     )

#     for lp in [
#         np.eye(4),
#         np.array([[1,0,0,2],[0,1,0,1],[0,0,1,1],[0,0,0,1]], dtype=float),
#     ]:
#         scene.add(pyrender.DirectionalLight(color=[1,1,1], intensity=2.5), pose=lp)

#     r = pyrender.OffscreenRenderer(viewport_width=viewport_w, viewport_height=viewport_h)
#     color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
#     r.delete()
#     return color


# # ============================
# # Save ghost conference figure
# # ============================
# def save_ghost_conf_figure(ghost_frame, caption, save_path, border_color="#4CAF50"):
#     fig, ax = plt.subplots(1, 1, figsize=(16, 6))
#     fig.patch.set_facecolor("#f0f0f0")
#     ax.imshow(ghost_frame)
#     ax.axis("off")

#     fig.text(0.5, 0.02, caption,
#              ha="center", va="center",
#              fontsize=13, fontstyle="italic", fontfamily="serif")

#     border = patches.FancyBboxPatch(
#         (0.01, 0.01), 0.98, 0.98,
#         boxstyle="square,pad=0", linewidth=3,
#         edgecolor=border_color, facecolor="none",
#         transform=fig.transFigure, clip_on=False
#     )
#     fig.add_artist(border)
#     plt.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.07)
#     plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
#     plt.close()
#     print(f"Saved: {save_path}")


# # ============================
# # Full pipeline for one query
# # ============================
# def render_query_ghost(
#     npz_file,
#     save_dir,
#     smpl_faces,
#     merged_parts,
#     caption,
#     gif_name,
#     fig_name,
#     spread=0.18,
#     spread_axis='x',
#     alpha_min=45,
#     alpha_max=255,
#     stride=1,            # use every Nth frame (1=all, 2=every other, etc.)
# ):
#     os.makedirs(save_dir, exist_ok=True)

#     data = np.load(npz_file)
#     if "vertices" not in data:
#         raise ValueError(f"No 'vertices' key in {npz_file}")
#     smpl_vertices = data["vertices"]   # (T, 6890, 3)
#     T = len(smpl_vertices)

#     all_v  = np.concatenate(smpl_vertices, axis=0)
#     center = all_v.mean(axis=0)
#     scale  = np.max(np.linalg.norm(all_v - center, axis=1))
#     verts_c = smpl_vertices - center[None]

#     # ── GIF: every frame ──────────────────────────────────────────
#     print(f"\nRendering GIF ({T} frames) for {os.path.basename(npz_file)} ...")
#     gif_frames = []
#     for verts in tqdm(verts_c, desc="GIF"):
#         mesh  = build_mesh(verts, smpl_faces, merged_parts, alpha=255)
#         scene = pyrender.Scene(bg_color=[230,230,230,255], ambient_light=[0.5,0.5,0.5])
#         scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=True))
#         cp = np.array([[1,0,0,0],[0,1,0,-0.2],[0,0,1,scale*2.5],[0,0,0,1]], dtype=float)
#         scene.add(pyrender.PerspectiveCamera(yfov=np.pi/3.0), pose=cp)
#         scene.add(pyrender.DirectionalLight(color=[1,1,1], intensity=3.0), pose=np.eye(4))
#         r = pyrender.OffscreenRenderer(640, 480)
#         color, _ = r.render(scene)
#         r.delete()
#         gif_frames.append(color)

#     gif_path = os.path.join(save_dir, gif_name)
#     imageio.mimsave(gif_path, gif_frames, fps=25)
#     print(f"Saved GIF: {gif_path}")

#     # ── Ghost: ALL frames (strided if sequence is very long) ───────
#     # Always include first and last frame exactly
#     if stride > 1:
#         indices = list(range(0, T, stride))
#         if indices[-1] != T - 1:
#             indices.append(T - 1)
#     else:
#         indices = list(range(T))

#     print(f"Ghost trail: {len(indices)} frames (stride={stride}, T={T})")
#     ghost_verts = [verts_c[i] for i in indices]

#     ghost_img = render_ghost_frame(
#         ghost_verts, smpl_faces, merged_parts, scale,
#         spread=spread, spread_axis=spread_axis,
#         alpha_min=alpha_min, alpha_max=alpha_max,
#     )

#     fig_path = os.path.join(save_dir, fig_name)
#     save_ghost_conf_figure(ghost_img, caption, fig_path)
#     return ghost_img, caption


# # ============================
# # Combined multi-query figure
# # ============================
# def save_combined_figure(rows, save_path, border_color="#4CAF50"):
#     n   = len(rows)
#     fig, axes = plt.subplots(n, 1, figsize=(16, 6.5 * n))
#     fig.patch.set_facecolor("#f0f0f0")
#     if n == 1:
#         axes = [axes]

#     for i, (img, caption) in enumerate(rows):
#         axes[i].imshow(img)
#         axes[i].axis("off")
#         axes[i].set_title(caption, fontsize=13, fontstyle="italic",
#                           fontfamily="serif", pad=8)

#     plt.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.02, hspace=0.10)

#     border = patches.FancyBboxPatch(
#         (0.01, 0.01), 0.98, 0.98,
#         boxstyle="square,pad=0", linewidth=3,
#         edgecolor=border_color, facecolor="none",
#         transform=fig.transFigure, clip_on=False
#     )
#     fig.add_artist(border)
#     plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
#     plt.close()
#     print(f"Saved combined: {save_path}")


# # ============================
# # Main
# # ============================
# def main():
#     SMPL_MODEL_PATH = "./smpl_models/smpl/SMPL/SMPL_NEUTRAL.pkl"
#     COMBINED_FIG    = "outputs/renders_smpl/test/combined_ghost_figure.png"

#     queries = [
#         {
#             "npz_file":    "outputs/renders_smpl/test/75/M013735_smpl_fit.npz",
#             "save_dir":    "outputs/renders_smpl/test/75",
#             "gif_name":    "M013735_mesh.gif",
#             "fig_name":    "M013735_ghost.png",
#             "caption":     "The man walked up to the door and knocked on it.",
#             "spread":      0.18,   # tighter = more overlap (ghostly)
#             "spread_axis": "x",
#             "alpha_min":   45,     # how faded the middle frames get
#             "alpha_max":   255,
#             "stride":      1,      # 1 = every frame, 2 = every other, etc.
#         },
#         {
#             "npz_file":    "outputs/renders_smpl/test/135/001840_smpl_fit.npz",
#             "save_dir":    "outputs/renders_smpl/test/135",
#             "gif_name":    "001840_mesh.gif",
#             "fig_name":    "001840_ghost.png",
#             "caption":     "A person sits down on a chair.",
#             "spread":      0.18,
#             "spread_axis": "x",
#             "alpha_min":   45,
#             "alpha_max":   255,
#             "stride":      1,
#         },
#     ]

#     with open(SMPL_MODEL_PATH, 'rb') as f:
#         smpl_model = pickle.load(f, encoding='latin1')
#     smpl_faces   = smpl_model['f']
#     merged_parts = build_vertex_parts(smpl_model)

#     rows = []
#     for q in queries:
#         ghost_img, caption = render_query_ghost(
#             npz_file     = q["npz_file"],
#             save_dir     = q["save_dir"],
#             smpl_faces   = smpl_faces,
#             merged_parts = merged_parts,
#             caption      = q["caption"],
#             gif_name     = q["gif_name"],
#             fig_name     = q["fig_name"],
#             spread       = q["spread"],
#             spread_axis  = q["spread_axis"],
#             alpha_min    = q["alpha_min"],
#             alpha_max    = q["alpha_max"],
#             stride       = q["stride"],
#         )
#         rows.append((ghost_img, caption))

#     os.makedirs(os.path.dirname(COMBINED_FIG), exist_ok=True)
#     save_combined_figure(rows, COMBINED_FIG)


# if __name__ == "__main__":
#     main()

# import os
# import numpy as np
# import pyrender
# import trimesh
# import imageio
# import pickle
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from tqdm import tqdm


# # ============================
# # Vertex part mapping
# # ============================
# def build_vertex_parts(smpl_model):
#     lbs_weights    = smpl_model['weights']
#     dominant_joint = np.argmax(lbs_weights, axis=1)
#     ARM_JOINTS = {13, 14, 16, 17, 18, 19, 20, 21, 22, 23}
#     LEG_JOINTS = {1, 2, 4, 5, 7, 8, 10, 11}
#     merged = np.zeros(6890, dtype=np.int32)
#     for v in range(6890):
#         j = dominant_joint[v]
#         if j in ARM_JOINTS:   merged[v] = 1
#         elif j in LEG_JOINTS: merged[v] = 2
#     return merged

# COLOR_MAP = {
#     0: [ 52,  95, 160, 255],
#     1: [180,  50,  50, 255],
#     2: [ 34, 139,  80, 255],
# }

# def build_mesh(verts, faces, merged_parts, alpha=255):
#     mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
#     vc   = np.zeros((verts.shape[0], 4), dtype=np.uint8)
#     for pid, color in COLOR_MAP.items():
#         c = list(color); c[3] = alpha
#         vc[merged_parts == pid] = c
#     mesh.visual.vertex_colors = vc
#     return mesh


# # ============================
# # KEY FRAME EXTRACTION
# # Picks frames with the most pose change
# # ============================
# def extract_keyframes(verts_c, n_keys=8, always_include_ends=True):
#     """
#     Compute per-frame pose velocity (mean vertex displacement vs previous frame).
#     Use farthest-point sampling in 'time weighted by motion' to pick maximally
#     diverse key poses — so slow/static parts get fewer frames, fast/changing
#     parts get more.
#     """
#     T = len(verts_c)

#     # Frame-to-frame velocity: mean displacement of all vertices
#     velocities = np.zeros(T)
#     for t in range(1, T):
#         velocities[t] = np.mean(np.linalg.norm(verts_c[t] - verts_c[t-1], axis=1))

#     # Cumulative motion arc — use as a 1D "distance" space
#     cum_motion = np.cumsum(velocities)
#     if cum_motion[-1] == 0:
#         # No motion at all, just evenly space
#         return list(np.linspace(0, T-1, n_keys, dtype=int))

#     cum_norm = cum_motion / cum_motion[-1]  # normalize 0→1

#     # Sample evenly in motion-arc space → more frames where motion is fast
#     target_arcs = np.linspace(0, 1, n_keys)
#     indices = []
#     for arc in target_arcs:
#         idx = int(np.argmin(np.abs(cum_norm - arc)))
#         indices.append(idx)

#     indices = sorted(set(indices))

#     if always_include_ends:
#         indices = sorted(set([0] + indices + [T - 1]))

#     # If we got more than n_keys due to deduplication, trim middle ones
#     while len(indices) > n_keys:
#         # Remove the index with smallest motion gap to its neighbor
#         gaps = [cum_motion[indices[i+1]] - cum_motion[indices[i]]
#                 for i in range(1, len(indices)-1)]  # skip first & last
#         drop = 1 + int(np.argmin(gaps))             # +1 to offset from 0
#         indices.pop(drop)

#     print(f"  Keyframes selected: {indices} ({len(indices)} total)")
#     return indices


# # ============================
# # Alpha: solid ends, faded middle
# # ============================
# def compute_alphas(n, alpha_min=50, alpha_max=255):
#     alphas = []
#     for i in range(n):
#         t        = i / max(n - 1, 1)
#         sine_val = np.sin(t * np.pi)
#         alphas.append(int(alpha_max - (alpha_max - alpha_min) * sine_val))
#     return alphas


# # ============================
# # Render keyframes — diagonal 3D camera + floor plane
# # ============================
# def make_floor(center_x, center_z, width, depth, y_level):
#     """Flat grey floor plane under the figures."""
#     verts = np.array([
#         [center_x - width/2, y_level, center_z - depth/2],
#         [center_x + width/2, y_level, center_z - depth/2],
#         [center_x + width/2, y_level, center_z + depth/2],
#         [center_x - width/2, y_level, center_z + depth/2],
#     ])
#     faces = np.array([[0,1,2],[0,2,3]])
#     mesh  = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
#     mesh.visual.vertex_colors = np.array([[160,160,160,200]]*4, dtype=np.uint8)
#     return mesh


# def render_keyframe_scene(
#     key_verts,       # list of (6890,3) — selected key poses
#     faces,
#     merged_parts,
#     scale,
#     spread_x=0.55,   # left-right spacing between figures
#     spread_z=0.20,   # depth spacing (diagonal look)
#     viewport_w=1400,
#     viewport_h=700,
#     alpha_min=50,
#     alpha_max=255,
#     camera_angle='diagonal',  # 'diagonal', 'side', 'front'
#     add_floor=True,
# ):
#     n      = len(key_verts)
#     alphas = compute_alphas(n, alpha_min=alpha_min, alpha_max=alpha_max)

#     scene = pyrender.Scene(
#         bg_color=[220, 220, 220, 255],
#         ambient_light=[0.5, 0.5, 0.5]
#     )

#     y_min = min(v[:,1].min() for v in key_verts)

#     positions = []
#     for i, (verts, alpha) in enumerate(zip(key_verts, alphas)):
#         t       = i / max(n-1, 1)
#         off_x   = (i - (n-1)/2.0) * spread_x * scale
#         off_z   = (i - (n-1)/2.0) * spread_z * scale  # diagonal depth
#         offset  = np.array([off_x, 0, -off_z])
#         positions.append(offset)

#         placed = verts + offset
#         mesh   = build_mesh(placed, faces, merged_parts, alpha=alpha)
#         scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=True))

#     # Floor
#     if add_floor:
#         all_x  = [p[0] for p in positions]
#         cx     = np.mean(all_x)
#         width  = (max(all_x) - min(all_x)) + scale * 1.5
#         depth  = scale * (1.5 + n * 0.15)
#         floor  = make_floor(cx, 0, width, depth, y_min - 0.02)
#         scene.add(pyrender.Mesh.from_trimesh(floor))

#     # Camera poses
#     cam_distance = scale * (2.5 + 0.3 * n)
#     if camera_angle == 'diagonal':
#         # Elevated, offset to the side — like Image 1
#         angle     = np.radians(35)
#         elev      = np.radians(25)
#         cam_x     = cam_distance * np.sin(angle)
#         cam_y     = cam_distance * np.sin(elev)
#         cam_z     = cam_distance * np.cos(angle)
#         # Look-at: build rotation matrix pointing toward origin
#         forward   = np.array([-cam_x, -cam_y, -cam_z])
#         forward  /= np.linalg.norm(forward)
#         right     = np.cross(forward, np.array([0,1,0])); right /= np.linalg.norm(right)
#         up        = np.cross(right, forward)
#         R         = np.column_stack([right, up, -forward])
#         cam_pose  = np.eye(4)
#         cam_pose[:3,:3] = R
#         cam_pose[:3, 3] = [cam_x, cam_y, cam_z]
#     elif camera_angle == 'side':
#         cam_pose = np.array([[1,0,0,0],[0,1,0,-0.15],[0,0,1,cam_distance],[0,0,0,1]], dtype=float)
#     else:  # front
#         cam_pose = np.array([[1,0,0,0],[0,1,0,-0.15],[0,0,1,cam_distance],[0,0,0,1]], dtype=float)

#     scene.add(
#         pyrender.PerspectiveCamera(yfov=np.pi/3.2, aspectRatio=viewport_w/viewport_h),
#         pose=cam_pose
#     )

#     # Multi-angle lighting for depth
#     for lp in [
#         np.eye(4),
#         np.array([[1,0,0,2],[0,1,0,2],[0,0,1,1],[0,0,0,1]], dtype=float),
#         np.array([[1,0,0,-1],[0,1,0,1],[0,0,1,0.5],[0,0,0,1]], dtype=float),
#     ]:
#         scene.add(pyrender.DirectionalLight(color=[1,1,1], intensity=2.2), pose=lp)

#     r = pyrender.OffscreenRenderer(viewport_width=viewport_w, viewport_height=viewport_h)
#     color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
#     r.delete()
#     return color


# # ============================
# # Save figure
# # ============================
# def save_ghost_conf_figure(ghost_frame, caption, save_path, border_color="#4CAF50"):
#     fig, ax = plt.subplots(1, 1, figsize=(14, 7))
#     fig.patch.set_facecolor("#f0f0f0")
#     ax.imshow(ghost_frame)
#     ax.axis("off")
#     fig.text(0.5, 0.02, caption, ha="center", va="center",
#              fontsize=14, fontstyle="italic", fontfamily="serif")
#     border = patches.FancyBboxPatch(
#         (0.01,0.01), 0.98, 0.98, boxstyle="square,pad=0",
#         linewidth=3, edgecolor=border_color, facecolor="none",
#         transform=fig.transFigure, clip_on=False)
#     fig.add_artist(border)
#     plt.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.07)
#     plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
#     plt.close()
#     print(f"Saved: {save_path}")


# # ============================
# # Full pipeline per query
# # ============================
# def render_query(
#     npz_file, save_dir, smpl_faces, merged_parts,
#     caption, gif_name, fig_name,
#     n_keys=8,
#     spread_x=0.55,
#     spread_z=0.20,
#     camera_angle='diagonal',
#     alpha_min=50,
#     alpha_max=255,
# ):
#     os.makedirs(save_dir, exist_ok=True)
#     data = np.load(npz_file)
#     if "vertices" not in data:
#         raise ValueError(f"No 'vertices' key in {npz_file}")
#     smpl_vertices = data["vertices"]
#     T = len(smpl_vertices)

#     all_v   = np.concatenate(smpl_vertices, axis=0)
#     center  = all_v.mean(axis=0)
#     scale   = np.max(np.linalg.norm(all_v - center, axis=1))
#     verts_c = smpl_vertices - center[None]

#     # GIF
#     print(f"\nRendering GIF ({T} frames) ...")
#     gif_frames = []
#     for verts in tqdm(verts_c, desc="GIF"):
#         mesh  = build_mesh(verts, smpl_faces, merged_parts, alpha=255)
#         scene = pyrender.Scene(bg_color=[230,230,230,255], ambient_light=[0.5,0.5,0.5])
#         scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=True))
#         cp = np.array([[1,0,0,0],[0,1,0,-0.2],[0,0,1,scale*2.5],[0,0,0,1]], dtype=float)
#         scene.add(pyrender.PerspectiveCamera(yfov=np.pi/3.0), pose=cp)
#         scene.add(pyrender.DirectionalLight(color=[1,1,1], intensity=3.0), pose=np.eye(4))
#         r = pyrender.OffscreenRenderer(640, 480)
#         color, _ = r.render(scene)
#         r.delete()
#         gif_frames.append(color)
#     imageio.mimsave(os.path.join(save_dir, gif_name), gif_frames, fps=25)
#     print(f"Saved GIF: {os.path.join(save_dir, gif_name)}")

#     # Keyframe extraction
#     print(f"Extracting keyframes ...")
#     indices   = extract_keyframes(verts_c, n_keys=n_keys)
#     key_verts = [verts_c[i] for i in indices]

#     # Render keyframe scene
#     ghost_img = render_keyframe_scene(
#         key_verts, smpl_faces, merged_parts, scale,
#         spread_x=spread_x, spread_z=spread_z,
#         camera_angle=camera_angle,
#         alpha_min=alpha_min, alpha_max=alpha_max,
#     )
#     save_ghost_conf_figure(ghost_img, caption, os.path.join(save_dir, fig_name))
#     return ghost_img, caption


# # ============================
# # Combined figure
# # ============================
# def save_combined_figure(rows, save_path, border_color="#4CAF50"):
#     n = len(rows)
#     fig, axes = plt.subplots(n, 1, figsize=(14, 7.5 * n))
#     fig.patch.set_facecolor("#f0f0f0")
#     if n == 1: axes = [axes]
#     for i, (img, caption) in enumerate(rows):
#         axes[i].imshow(img); axes[i].axis("off")
#         axes[i].set_title(caption, fontsize=13, fontstyle="italic",
#                           fontfamily="serif", pad=8)
#     plt.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.02, hspace=0.10)
#     border = patches.FancyBboxPatch(
#         (0.01,0.01), 0.98, 0.98, boxstyle="square,pad=0",
#         linewidth=3, edgecolor=border_color, facecolor="none",
#         transform=fig.transFigure, clip_on=False)
#     fig.add_artist(border)
#     plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
#     plt.close()
#     print(f"Saved combined: {save_path}")


# # ============================
# # Main
# # ============================
# def main():
#     SMPL_MODEL_PATH = "./smpl_models/smpl/SMPL/SMPL_NEUTRAL.pkl"
#     COMBINED_FIG    = "outputs/renders_smpl/test/combined_ghost_figure.png"

#     queries = [
#         {
#             "npz_file":      "outputs/renders_smpl/test/75/M013735_smpl_fit.npz",
#             "save_dir":      "outputs/renders_smpl/test/75",
#             "gif_name":      "M013735_mesh.gif",
#             "fig_name":      "M013735_ghost.png",
#             "caption":       "The man walked up to the door and knocked on it.",
#             "n_keys":        8,
#             "spread_x":      0.55,
#             "spread_z":      0.20,
#             "camera_angle":  "diagonal",
#             "alpha_min":     50,
#             "alpha_max":     255,
#         },
#         {
#             "npz_file":      "outputs/renders_smpl/test/135/001840_smpl_fit.npz",
#             "save_dir":      "outputs/renders_smpl/test/135",
#             "gif_name":      "001840_mesh.gif",
#             "fig_name":      "001840_ghost.png",
#             "caption":       "A person sits down on a chair.",
#             "n_keys":        8,
#             "spread_x":      0.55,
#             "spread_z":      0.20,
#             "camera_angle":  "diagonal",
#             "alpha_min":     50,
#             "alpha_max":     255,
#         },
#     ]

#     with open(SMPL_MODEL_PATH, 'rb') as f:
#         smpl_model = pickle.load(f, encoding='latin1')
#     smpl_faces   = smpl_model['f']
#     merged_parts = build_vertex_parts(smpl_model)

#     rows = []
#     for q in queries:
#         ghost_img, caption = render_query(
#             npz_file=q["npz_file"], save_dir=q["save_dir"],
#             smpl_faces=smpl_faces, merged_parts=merged_parts,
#             caption=q["caption"], gif_name=q["gif_name"], fig_name=q["fig_name"],
#             n_keys=q["n_keys"], spread_x=q["spread_x"], spread_z=q["spread_z"],
#             camera_angle=q["camera_angle"],
#             alpha_min=q["alpha_min"], alpha_max=q["alpha_max"],
#         )
#         rows.append((ghost_img, caption))

#     os.makedirs(os.path.dirname(COMBINED_FIG), exist_ok=True)
#     save_combined_figure(rows, COMBINED_FIG)

# if __name__ == "__main__":
#     main()

# import os
# import numpy as np
# import pyrender
# import trimesh
# import imageio
# import pickle
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from tqdm import tqdm


# def build_vertex_parts(smpl_model):
#     lbs_weights    = smpl_model['weights']
#     dominant_joint = np.argmax(lbs_weights, axis=1)
#     ARM_JOINTS = {13, 14, 16, 17, 18, 19, 20, 21, 22, 23}
#     LEG_JOINTS = {1, 2, 4, 5, 7, 8, 10, 11}
#     merged = np.zeros(6890, dtype=np.int32)
#     for v in range(6890):
#         j = dominant_joint[v]
#         if j in ARM_JOINTS:   merged[v] = 1
#         elif j in LEG_JOINTS: merged[v] = 2
#     return merged


# COLOR_MAP = {
#     0: [ 52,  95, 160, 255],
#     1: [180,  50,  50, 255],
#     2: [ 34, 139,  80, 255],
# }


# def build_mesh(verts, faces, merged_parts, alpha=255):
#     mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
#     vc   = np.zeros((verts.shape[0], 4), dtype=np.uint8)
#     for pid, color in COLOR_MAP.items():
#         c = list(color); c[3] = alpha
#         vc[merged_parts == pid] = c
#     mesh.visual.vertex_colors = vc
#     return mesh


# def compute_alphas(n, alpha_min=50, alpha_max=255):
#     alphas = []
#     for i in range(n):
#         t = i / max(n - 1, 1)
#         sine_val = np.sin(t * np.pi)
#         alphas.append(int(alpha_max - (alpha_max - alpha_min) * sine_val))
#     return alphas


# def extract_keyframes(verts_c, n_keys=8):
#     T = len(verts_c)
#     velocities = np.zeros(T)
#     for t in range(1, T):
#         velocities[t] = np.mean(np.linalg.norm(verts_c[t] - verts_c[t-1], axis=1))
#     cum_motion = np.cumsum(velocities)
#     if cum_motion[-1] == 0:
#         return list(np.linspace(0, T-1, n_keys, dtype=int))
#     cum_norm   = cum_motion / cum_motion[-1]
#     target_arcs = np.linspace(0, 1, n_keys)
#     indices = sorted(set(
#         [0] + [int(np.argmin(np.abs(cum_norm - a))) for a in target_arcs] + [T-1]
#     ))
#     while len(indices) > n_keys:
#         gaps = [cum_motion[indices[i+1]] - cum_motion[indices[i]]
#                 for i in range(1, len(indices)-1)]
#         indices.pop(1 + int(np.argmin(gaps)))
#     print(f"  Keyframes: {indices} ({len(indices)} total)")
#     return indices


# # ============================
# # UPDATED: Floor + diagonal camera
# # ============================
# def make_floor_mesh(cx, cz, width, depth, y_level):
#     """Grey floor plane with slight perspective tilt."""
#     verts = np.array([
#         [cx - width/2, y_level, cz - depth/2],
#         [cx + width/2, y_level, cz - depth/2],
#         [cx + width/2, y_level, cz + depth/2],
#         [cx - width/2, y_level, cz + depth/2],
#     ])
#     faces = np.array([[0,1,2],[0,2,3]])
#     mesh  = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
#     mesh.visual.vertex_colors = np.array([[110,108,105,255]]*4, dtype=np.uint8)
#     return mesh


# def make_diagonal_camera_pose(cx, cy, cz, target=None):
#     """
#     Build a camera pose looking at `target` from position (cx,cy,cz).
#     Returns 4x4 pose matrix.
#     """
#     if target is None:
#         target = np.array([0.0, 0.0, 0.0])
#     pos     = np.array([cx, cy, cz])
#     forward = target - pos
#     forward = forward / np.linalg.norm(forward)
#     world_up = np.array([0.0, 1.0, 0.0])
#     right    = np.cross(forward, world_up)
#     right    = right / np.linalg.norm(right)
#     up       = np.cross(right, forward)

#     # pyrender camera looks along -Z, so we negate forward
#     R = np.column_stack([right, up, -forward])
#     pose = np.eye(4)
#     pose[:3, :3] = R
#     pose[:3,  3] = pos
#     return pose


# def render_keyframe_scene(
#     key_verts,
#     faces,
#     merged_parts,
#     scale,
#     spread_x=0.60,
#     spread_z=0.18,        # depth offset — gives diagonal feel
#     viewport_w=1400,
#     viewport_h=680,
#     alpha_min=50,
#     alpha_max=255,
# ):
#     n      = len(key_verts)
#     alphas = compute_alphas(n, alpha_min=alpha_min, alpha_max=alpha_max)

#     scene  = pyrender.Scene(
#         bg_color=[215, 213, 210, 255],   # warm light grey like Image 1
#         ambient_light=[0.45, 0.45, 0.45]
#     )

#     # Find global floor level
#     y_min = min(v[:,1].min() for v in key_verts)

#     positions = []
#     for i, (verts, alpha) in enumerate(zip(key_verts, alphas)):
#         # Spread left-right AND slightly in depth for diagonal feel
#         off_x = (i - (n-1)/2.0) * spread_x * scale
#         off_z = -(i - (n-1)/2.0) * spread_z * scale  # negative = further back on left
#         offset = np.array([off_x, 0.0, off_z])
#         positions.append(offset)

#         placed = verts + offset
#         mesh   = build_mesh(placed, faces, merged_parts, alpha=alpha)
#         scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=True))

#     # ── Floor ──────────────────────────────────────────────────────
#     xs      = [p[0] for p in positions]
#     zs      = [p[2] for p in positions]
#     cx_floor = np.mean(xs)
#     cz_floor = np.mean(zs)
#     f_width  = (max(xs) - min(xs)) + scale * 2.2   # wider floor
#     f_depth  = scale * (2.0 + n * 0.15)            # deeper floor
#     floor    = make_floor_mesh(cx_floor, cz_floor, f_width, f_depth, y_min - 0.01)
#     scene.add(pyrender.Mesh.from_trimesh(floor))

#     # ── Diagonal camera (elevated, offset to side) ─────────────────
#     # Scene centre
#     scene_cx = cx_floor
#     scene_cy = y_min + scale * 0.5    # aim at body midpoint, not feet

#     # Camera sits elevated and to one side — matches Image 1 perspective
#     cam_dist = scale * (1.6 + 0.18 * n)
#     cam_x    = scene_cx + cam_dist * 0.30
#     cam_y    = scene_cy + cam_dist * 0.38
#     cam_z    = cam_dist * 0.78

#     target    = np.array([scene_cx, scene_cy, cz_floor])
#     cam_pose  = make_diagonal_camera_pose(cam_x, cam_y, cam_z, target)

#     scene.add(
#         pyrender.PerspectiveCamera(
#             yfov=np.pi / 3.2,
#             aspectRatio=viewport_w / viewport_h
#         ),
#         pose=cam_pose
#     )

#     # ── Lighting (3-point for depth/shadow feel) ───────────────────
#     # Key light — from upper right front
#     key_pose = np.eye(4)
#     key_pose[:3, 3] = [scene_cx + scale, scene_cy + scale*2, scale*2]
#     scene.add(pyrender.DirectionalLight(color=[1.0, 0.98, 0.95], intensity=3.5),
#               pose=key_pose)

#     # Fill light — from left, softer
#     fill_pose = np.eye(4)
#     fill_pose[:3, 3] = [scene_cx - scale*1.5, scene_cy + scale, scale]
#     scene.add(pyrender.DirectionalLight(color=[0.8, 0.88, 1.0], intensity=1.5),
#               pose=fill_pose)

#     # Rim light — from behind, subtle
#     rim_pose = np.eye(4)
#     rim_pose[:3, 3] = [scene_cx, scene_cy + scale, -scale*2]
#     scene.add(pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.8),
#               pose=rim_pose)

#     r = pyrender.OffscreenRenderer(viewport_width=viewport_w, viewport_height=viewport_h)
#     color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
#     r.delete()
#     return color


# # ============================
# # Save figure
# # ============================
# def save_ghost_conf_figure(ghost_frame, caption, save_path, border_color="#4CAF50"):
#     fig, ax = plt.subplots(1, 1, figsize=(14, 7))
#     fig.patch.set_facecolor("#f0f0f0")
#     ax.imshow(ghost_frame)
#     ax.axis("off")
#     fig.text(0.5, 0.02, caption, ha="center", va="center",
#              fontsize=14, fontstyle="italic", fontfamily="serif")
#     border = patches.FancyBboxPatch(
#         (0.01,0.01), 0.98, 0.98, boxstyle="square,pad=0",
#         linewidth=3, edgecolor=border_color, facecolor="none",
#         transform=fig.transFigure, clip_on=False)
#     fig.add_artist(border)
#     plt.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.07)
#     plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
#     plt.close()
#     print(f"Saved: {save_path}")


# # ============================
# # Full pipeline per query
# # ============================
# def render_query(
#     npz_file, save_dir, smpl_faces, merged_parts,
#     caption, gif_name, fig_name,
#     n_keys=8, spread_x=0.60, spread_z=0.18,
#     alpha_min=50, alpha_max=255,
# ):
#     os.makedirs(save_dir, exist_ok=True)
#     data = np.load(npz_file)
#     if "vertices" not in data:
#         raise ValueError(f"No 'vertices' key in {npz_file}")
#     smpl_vertices = data["vertices"]
#     T = len(smpl_vertices)

#     all_v   = np.concatenate(smpl_vertices, axis=0)
#     center  = all_v.mean(axis=0)
#     scale   = np.max(np.linalg.norm(all_v - center, axis=1))
#     verts_c = smpl_vertices - center[None]

#     # GIF
#     print(f"\nRendering GIF ({T} frames) ...")
#     gif_frames = []
#     for verts in tqdm(verts_c, desc="GIF"):
#         mesh  = build_mesh(verts, smpl_faces, merged_parts, alpha=255)
#         scene = pyrender.Scene(bg_color=[215,213,210,255], ambient_light=[0.5,0.5,0.5])
#         scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=True))
#         cp = np.array([[1,0,0,0],[0,1,0,-0.2],[0,0,1,scale*2.5],[0,0,0,1]], dtype=float)
#         scene.add(pyrender.PerspectiveCamera(yfov=np.pi/3.0), pose=cp)
#         scene.add(pyrender.DirectionalLight(color=[1,1,1], intensity=3.0), pose=np.eye(4))
#         r = pyrender.OffscreenRenderer(640, 480)
#         color, _ = r.render(scene)
#         r.delete()
#         gif_frames.append(color)
#     imageio.mimsave(os.path.join(save_dir, gif_name), gif_frames, fps=25)

#     # Keyframes + ghost scene
#     indices   = extract_keyframes(verts_c, n_keys=n_keys)
#     key_verts = [verts_c[i] for i in indices]
#     ghost_img = render_keyframe_scene(
#         key_verts, smpl_faces, merged_parts, scale,
#         spread_x=spread_x, spread_z=spread_z,
#         alpha_min=alpha_min, alpha_max=alpha_max,
#     )
#     save_ghost_conf_figure(ghost_img, caption, os.path.join(save_dir, fig_name))
#     return ghost_img, caption


# # ============================
# # Combined figure
# # ============================
# def save_combined_figure(rows, save_path, border_color="#4CAF50"):
#     n = len(rows)
#     fig, axes = plt.subplots(n, 1, figsize=(14, 7.5 * n))
#     fig.patch.set_facecolor("#f0f0f0")
#     if n == 1: axes = [axes]
#     for i, (img, caption) in enumerate(rows):
#         axes[i].imshow(img); axes[i].axis("off")
#         axes[i].set_title(caption, fontsize=13, fontstyle="italic",
#                           fontfamily="serif", pad=8)
#     plt.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.02, hspace=0.10)
#     border = patches.FancyBboxPatch(
#         (0.01,0.01), 0.98, 0.98, boxstyle="square,pad=0",
#         linewidth=3, edgecolor=border_color, facecolor="none",
#         transform=fig.transFigure, clip_on=False)
#     fig.add_artist(border)
#     plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
#     plt.close()
#     print(f"Saved combined: {save_path}")


# # ============================
# # Main
# # ============================
# def main():
#     SMPL_MODEL_PATH = "./smpl_models/smpl/SMPL/SMPL_NEUTRAL.pkl"
#     COMBINED_FIG    = "outputs/renders_smpl/test/combined_ghost_figure.png"

#     queries = [
#         {
#             "npz_file":  "outputs/renders_smpl/test/75/M013735_smpl_fit.npz",
#             "save_dir":  "outputs/renders_smpl/test/75",
#             "gif_name":  "M013735_mesh.gif",
#             "fig_name":  "M013735_ghost.png",
#             "caption":   "The man walked up to the door and knocked on it.",
#             "n_keys":    8,
#             "spread_x":  0.60,
#             "spread_z":  0.18,
#             "alpha_min": 50,
#             "alpha_max": 255,
#         },
#         {
#             "npz_file":  "outputs/renders_smpl/test/135/001840_smpl_fit.npz",
#             "save_dir":  "outputs/renders_smpl/test/135",
#             "gif_name":  "001840_mesh.gif",
#             "fig_name":  "001840_ghost.png",
#             "caption":   "A person sits down on a chair.",
#             "n_keys":    8,
#             "spread_x":  0.60,
#             "spread_z":  0.18,
#             "alpha_min": 50,
#             "alpha_max": 255,
#         },
#     ]

#     with open(SMPL_MODEL_PATH, 'rb') as f:
#         smpl_model = pickle.load(f, encoding='latin1')
#     smpl_faces   = smpl_model['f']
#     merged_parts = build_vertex_parts(smpl_model)

#     rows = []
#     for q in queries:
#         ghost_img, caption = render_query(
#             npz_file=q["npz_file"], save_dir=q["save_dir"],
#             smpl_faces=smpl_faces, merged_parts=merged_parts,
#             caption=q["caption"], gif_name=q["gif_name"], fig_name=q["fig_name"],
#             n_keys=q["n_keys"], spread_x=q["spread_x"], spread_z=q["spread_z"],
#             alpha_min=q["alpha_min"], alpha_max=q["alpha_max"],
#         )
#         rows.append((ghost_img, caption))

#     os.makedirs(os.path.dirname(COMBINED_FIG), exist_ok=True)
#     save_combined_figure(rows, COMBINED_FIG)

# if __name__ == "__main__":
#     main()

import os
import numpy as np
import pyrender
import trimesh
import imageio
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm


# ============================
# Vertex part mapping
# ============================
def build_vertex_parts(smpl_model):
    lbs_weights    = smpl_model['weights']
    dominant_joint = np.argmax(lbs_weights, axis=1)
    ARM_JOINTS = {13, 14, 16, 17, 18, 19, 20, 21, 22, 23}
    LEG_JOINTS = {1, 2, 4, 5, 7, 8, 10, 11}
    merged = np.zeros(6890, dtype=np.int32)
    for v in range(6890):
        j = dominant_joint[v]
        if j in ARM_JOINTS:   merged[v] = 1
        elif j in LEG_JOINTS: merged[v] = 2
    return merged


COLOR_MAP = {
    0: [ 52,  95, 160, 255],   # torso + head → navy blue
    1: [180,  50,  50, 255],   # arms + hands → dark red
    2: [ 34, 139,  80, 255],   # legs + feet  → dark green
}


def build_mesh(verts, faces, merged_parts, alpha=255):
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    vc   = np.zeros((verts.shape[0], 4), dtype=np.uint8)
    for pid, color in COLOR_MAP.items():
        c = list(color); c[3] = alpha
        vc[merged_parts == pid] = c
    mesh.visual.vertex_colors = vc
    return mesh


# ============================
# Alpha: solid ends, faded middle
# ============================
def compute_alphas(n, alpha_min=50, alpha_max=255):
    alphas = []
    for i in range(n):
        t = i / max(n - 1, 1)
        alphas.append(int(alpha_max - (alpha_max - alpha_min) * np.sin(t * np.pi)))
    return alphas


# ============================
# Motion-based keyframe extraction
# ============================
def extract_keyframes(verts_c, n_keys=8):
    T = len(verts_c)
    velocities = np.zeros(T)
    for t in range(1, T):
        velocities[t] = np.mean(np.linalg.norm(verts_c[t] - verts_c[t-1], axis=1))
    cum_motion = np.cumsum(velocities)
    if cum_motion[-1] == 0:
        return list(np.linspace(0, T-1, n_keys, dtype=int))
    cum_norm    = cum_motion / cum_motion[-1]
    target_arcs = np.linspace(0, 1, n_keys)
    indices = sorted(set(
        [0] + [int(np.argmin(np.abs(cum_norm - a))) for a in target_arcs] + [T-1]
    ))
    while len(indices) > n_keys:
        gaps = [cum_motion[indices[i+1]] - cum_motion[indices[i]]
                for i in range(1, len(indices)-1)]
        indices.pop(1 + int(np.argmin(gaps)))
    print(f"  Keyframes: {indices} ({len(indices)} total)")
    return indices


# ============================
# Realistic floor with grid lines
# ============================
def make_floor_mesh(cx, cz, width, depth, y_level):
    """
    Tiled floor plane — base quad + darker grid lines baked as thin quads.
    Gives a realistic studio/gym floor look.
    """
    meshes = []

    # Base floor — medium grey
    verts = np.array([
        [cx - width/2, y_level, cz - depth/2],
        [cx + width/2, y_level, cz - depth/2],
        [cx + width/2, y_level, cz + depth/2],
        [cx - width/2, y_level, cz + depth/2],
    ])
    faces = np.array([[0,1,2],[0,2,3]])
    base  = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    base.visual.vertex_colors = np.array([[118,115,110,255]]*4, dtype=np.uint8)
    meshes.append(base)

    # Grid lines — thin raised quads so they're always visible
    tile_size = width / 8.0
    line_w    = tile_size * 0.04
    y_grid    = y_level + 0.002   # just above floor
    grid_color = np.array([80, 78, 75, 200], dtype=np.uint8)

    # X-direction lines (run along X axis)
    n_lines_z = int(depth / tile_size) + 2
    for k in range(n_lines_z):
        z = cz - depth/2 + k * tile_size
        v = np.array([
            [cx - width/2, y_grid, z - line_w/2],
            [cx + width/2, y_grid, z - line_w/2],
            [cx + width/2, y_grid, z + line_w/2],
            [cx - width/2, y_grid, z + line_w/2],
        ])
        f = np.array([[0,1,2],[0,2,3]])
        m = trimesh.Trimesh(vertices=v, faces=f, process=False)
        m.visual.vertex_colors = np.array([grid_color]*4)
        meshes.append(m)

    # Z-direction lines (run along Z axis)
    n_lines_x = int(width / tile_size) + 2
    for k in range(n_lines_x):
        x = cx - width/2 + k * tile_size
        v = np.array([
            [x - line_w/2, y_grid, cz - depth/2],
            [x + line_w/2, y_grid, cz - depth/2],
            [x + line_w/2, y_grid, cz + depth/2],
            [x - line_w/2, y_grid, cz + depth/2],
        ])
        f = np.array([[0,1,2],[0,2,3]])
        m = trimesh.Trimesh(vertices=v, faces=f, process=False)
        m.visual.vertex_colors = np.array([grid_color]*4)
        meshes.append(m)

    return trimesh.util.concatenate(meshes)


# ============================
# Camera look-at matrix
# ============================
def make_camera_pose(cam_x, cam_y, cam_z, target):
    pos     = np.array([cam_x, cam_y, cam_z])
    forward = target - pos
    forward = forward / np.linalg.norm(forward)
    world_up = np.array([0.0, 1.0, 0.0])
    right    = np.cross(forward, world_up)
    right    = right / np.linalg.norm(right)
    up       = np.cross(right, forward)
    R        = np.column_stack([right, up, -forward])
    pose     = np.eye(4)
    pose[:3, :3] = R
    pose[:3,  3] = pos
    return pose


# ============================
# Render ghost keyframe scene
# ============================
def render_keyframe_scene(
    key_verts,
    faces,
    merged_parts,
    scale,
    spread_x=0.42,
    spread_z=0.28,
    viewport_w=1400,
    viewport_h=700,
    alpha_min=50,
    alpha_max=255,
):
    n      = len(key_verts)
    alphas = compute_alphas(n, alpha_min=alpha_min, alpha_max=alpha_max)

    scene  = pyrender.Scene(
        bg_color=[200, 198, 195, 255],
        ambient_light=[0.35, 0.35, 0.35]
    )

    y_min = min(v[:,1].min() for v in key_verts)

    positions = []
    for i, (verts, alpha) in enumerate(zip(key_verts, alphas)):
        off_x = (i - (n-1)/2.0) * spread_x * scale
        off_z = -(i - (n-1)/2.0) * spread_z * scale
        offset = np.array([off_x, 0.0, off_z])
        positions.append(offset)
        mesh = build_mesh(verts + offset, faces, merged_parts, alpha=alpha)
        scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=True))

    # ── Realistic tiled floor ──────────────────────────────────────
    xs       = [p[0] for p in positions]
    zs       = [p[2] for p in positions]
    cx_floor = np.mean(xs)
    cz_floor = np.mean(zs)
    f_width  = (max(xs) - min(xs)) + scale * 3.5
    f_depth  = scale * (4.0 + n * 0.22)
    floor    = make_floor_mesh(cx_floor, cz_floor, f_width, f_depth, y_min - 0.005)
    scene.add(pyrender.Mesh.from_trimesh(floor))

    # ── Camera: elevated + angled for floor visibility ─────────────
    # scene_cx = cx_floor
    # scene_cy = y_min + scale * 0.5
    # cam_dist = scale * (1.5 + 0.16 * n)

    # cam_x    = scene_cx + cam_dist * 0.15   # slight right offset
    # cam_y    = scene_cy + cam_dist * 1.10   # high elevation — shows floor
    # cam_z    = cam_dist * 1.10              # back enough to see all figures

    # # Aim at foot level and slightly in front of figures
    # target   = np.array([scene_cx, y_min + scale * 0.05, cz_floor + scale * 0.2])
    # cam_pose = make_camera_pose(cam_x, cam_y, cam_z, target)
    # ── Improved Camera: clean front + slight elevation ─────────────
    scene_cx = cx_floor
    scene_cy = y_min + scale * 0.5

    cam_dist = scale * (1.8 + 0.1 * n)

    # Front-facing camera (aligned with motion)
    cam_x = scene_cx
    cam_y = scene_cy + cam_dist * 0.8   # slight elevation
    cam_z = cam_dist * 1.2              # distance in front

    # Look directly at center of motion
    target = np.array([scene_cx, scene_cy, cz_floor])

    cam_pose = make_camera_pose(cam_x, cam_y, cam_z, target)

    scene.add(
        pyrender.PerspectiveCamera(
            yfov=np.pi / 2.6,
            aspectRatio=viewport_w / viewport_h
        ),
        pose=cam_pose
    )

    # ── 3-point lighting ───────────────────────────────────────────
    # Key light: warm, upper right front
    kp = np.eye(4); kp[:3,3] = [scene_cx + scale*2, scene_cy + scale*3, scale*2]
    scene.add(pyrender.DirectionalLight(color=[1.00, 0.97, 0.92], intensity=4.5), pose=kp)

    # Fill light: cool, left
    fp = np.eye(4); fp[:3,3] = [scene_cx - scale*2, scene_cy + scale*1.5, scale]
    scene.add(pyrender.DirectionalLight(color=[0.75, 0.85, 1.00], intensity=2.0), pose=fp)

    # Rim light: behind figures, subtle edge separation
    rp = np.eye(4); rp[:3,3] = [scene_cx, scene_cy + scale, -scale*3]
    scene.add(pyrender.DirectionalLight(color=[1.00, 1.00, 1.00], intensity=1.2), pose=rp)

    # Ground bounce: soft upward fill (simulates floor reflection)
    gp = np.eye(4); gp[:3,3] = [scene_cx, y_min - scale, 0]
    scene.add(pyrender.DirectionalLight(color=[0.90, 0.88, 0.85], intensity=0.8), pose=gp)

    r = pyrender.OffscreenRenderer(viewport_width=viewport_w, viewport_height=viewport_h)
    color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
    r.delete()
    return color


# ============================
# Save single query figure
# ============================
def save_ghost_conf_figure(ghost_frame, caption, save_path, border_color="#4CAF50"):
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    fig.patch.set_facecolor("#efefef")
    ax.imshow(ghost_frame)
    ax.axis("off")
    fig.text(0.5, 0.02, caption,
             ha="center", va="center",
             fontsize=14, fontstyle="italic", fontfamily="serif")
    border = patches.FancyBboxPatch(
        (0.01, 0.01), 0.98, 0.98,
        boxstyle="square,pad=0", linewidth=3,
        edgecolor=border_color, facecolor="none",
        transform=fig.transFigure, clip_on=False)
    fig.add_artist(border)
    plt.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.07)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved figure: {save_path}")


# ============================
# GIF render (front-facing, per frame)
# ============================
def render_gif_frame(verts, faces, merged_parts, scale):
    mesh  = build_mesh(verts, faces, merged_parts, alpha=255)
    scene = pyrender.Scene(bg_color=[215,213,210,255], ambient_light=[0.5,0.5,0.5])
    scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=True))
    cp = np.array([[1,0,0,0],[0,1,0,-0.2],[0,0,1,scale*2.5],[0,0,0,1]], dtype=float)
    scene.add(pyrender.PerspectiveCamera(yfov=np.pi/3.0), pose=cp)
    scene.add(pyrender.DirectionalLight(color=[1,1,1], intensity=3.5), pose=np.eye(4))
    r = pyrender.OffscreenRenderer(640, 480)
    color, _ = r.render(scene)
    r.delete()
    return color


# ============================
# Full pipeline per query
# ============================
def render_query(
    npz_file, save_dir, smpl_faces, merged_parts,
    caption, gif_name, fig_name,
    n_keys=8,
    spread_x=0.42,
    spread_z=0.28,
    alpha_min=50,
    alpha_max=255,
):
    os.makedirs(save_dir, exist_ok=True)

    data = np.load(npz_file)
    if "vertices" not in data:
        raise ValueError(f"No 'vertices' key in {npz_file}")
    smpl_vertices = data["vertices"]   # (T, 6890, 3)
    T = len(smpl_vertices)

    all_v   = np.concatenate(smpl_vertices, axis=0)
    center  = all_v.mean(axis=0)
    scale   = np.max(np.linalg.norm(all_v - center, axis=1))
    verts_c = smpl_vertices - center[None]

    # ── GIF ────────────────────────────────────────────────────────
    print(f"\nRendering GIF ({T} frames): {os.path.basename(npz_file)}")
    gif_frames = []
    for verts in tqdm(verts_c, desc="GIF"):
        gif_frames.append(render_gif_frame(verts, smpl_faces, merged_parts, scale))
    gif_path = os.path.join(save_dir, gif_name)
    imageio.mimsave(gif_path, gif_frames, fps=25)
    print(f"Saved GIF: {gif_path}")

    # ── Keyframes + ghost scene ────────────────────────────────────
    print(f"Extracting keyframes ...")
    indices   = extract_keyframes(verts_c, n_keys=n_keys)
    key_verts = [verts_c[i] for i in indices]

    ghost_img = render_keyframe_scene(
        key_verts, smpl_faces, merged_parts, scale,
        spread_x=spread_x, spread_z=spread_z,
        alpha_min=alpha_min, alpha_max=alpha_max,
    )
    fig_path = os.path.join(save_dir, fig_name)
    save_ghost_conf_figure(ghost_img, caption, fig_path)
    return ghost_img, caption


# ============================
# Combined 2-row paper figure
# ============================
def save_combined_figure(rows, save_path, border_color="#4CAF50"):
    n   = len(rows)
    fig, axes = plt.subplots(n, 1, figsize=(14, 7 * n))
    fig.patch.set_facecolor("#efefef")
    if n == 1: axes = [axes]
    for i, (img, caption) in enumerate(rows):
        axes[i].imshow(img)
        axes[i].axis("off")
        axes[i].set_title(caption, fontsize=13, fontstyle="italic",
                          fontfamily="serif", pad=8)
    plt.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.02, hspace=0.08)
    border = patches.FancyBboxPatch(
        (0.01, 0.01), 0.98, 0.98,
        boxstyle="square,pad=0", linewidth=3,
        edgecolor=border_color, facecolor="none",
        transform=fig.transFigure, clip_on=False)
    fig.add_artist(border)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved combined: {save_path}")


# ============================
# Main — add your queries here
# ============================
def main():
    SMPL_MODEL_PATH = "./smpl_models/smpl/SMPL/SMPL_NEUTRAL.pkl"
    COMBINED_FIG    = "outputs/renders_smpl/test/combined_ghost_figure.png"

    queries = [
        {
            "npz_file":  "outputs/renders_smpl/test/22/004965_smpl_fit.npz",
            "save_dir":  "outputs/renders_smpl/test/22",
            "gif_name":  "004965_mesh.gif",
            "fig_name":  "004965_ghost.png",
            "caption":   "a person walks up to something, picks it up, brings it back to where they were, and begins to make a washing motion with their hand..",
            "n_keys":    8,
            "spread_x":  0.42,
            "spread_z":  0.28,
            "alpha_min": 50,
            "alpha_max": 255,
        },
         {
            "npz_file":  "outputs/renders_smpl/test/75/M013735_smpl_fit.npz",
            "save_dir":  "outputs/renders_smpl/test/75",
            "gif_name":  "M013735_mesh.gif",
            "fig_name":  "M013735_ghost.png",
            "caption":   "A person giving a round of applause.",
            "n_keys":    8,
            "spread_x":  0.42,
            "spread_z":  0.28,
            "alpha_min": 50,
            "alpha_max": 255,
        },
        {
            "npz_file":  "outputs/renders_smpl/test/135/M001840_smpl_fit.npz",
            "save_dir":  "outputs/renders_smpl/test/135",
            "gif_name":  "M001840_mesh.gif",
            "fig_name":  "001840_ghost.png",
            "caption":   "a man is standing and brings both hands to his face then steps out with left foot and performs a low kick.",
            "n_keys":    8,
            "spread_x":  0.42,
            "spread_z":  0.28,
            "alpha_min": 50,
            "alpha_max": 255,
        },
          {
            "npz_file":  "outputs/renders_smpl/test/155/M003897_smpl_fit.npz",
            "save_dir":  "outputs/renders_smpl/test/155",
            "gif_name":  "M003897_mesh.gif",
            "fig_name":  "M003897_ghost.png",
            "caption":   "a man gets on his knees and crawls from right to left, then stands up again.",
            "n_keys":    8,
            "spread_x":  0.42,
            "spread_z":  0.28,
            "alpha_min": 50,
            "alpha_max": 255,
        },
          {
            "npz_file":  "outputs/renders_smpl/test/203/M005433_smpl_fit.npz",
            "save_dir":  "outputs/renders_smpl/test/203",
            "gif_name":  "M005433_mesh.gif",
            "fig_name":  "M005433_ghost.png",
            "caption":   "person backed up and sat down.",
            "n_keys":    8,
            "spread_x":  0.42,
            "spread_z":  0.28,
            "alpha_min": 50,
            "alpha_max": 255,
        },
          {
            "npz_file":  "outputs/renders_smpl/test/209/009577_smpl_fit.npz",
            "save_dir":  "outputs/renders_smpl/test/209",
            "gif_name":  "009577_mesh.gif",
            "fig_name":  "009577_ghost.png",
            "caption":   "a person puts his hands together in front of him then rests them on his side.",
            "n_keys":    8,
            "spread_x":  0.42,
            "spread_z":  0.28,
            "alpha_min": 50,
            "alpha_max": 255,
        },
          {
            "npz_file":  "outputs/renders_smpl/test/243/009041_smpl_fit.npz",
            "save_dir":  "outputs/renders_smpl/test/243",
            "gif_name":  "009041.gif",
            "fig_name":  "009041_ghost.png",
            "caption":   " person standing up throws something forward from above their head, then throws something again forward from above their head with more force which makes them take one step forward with their right foot.",
            "n_keys":    8,
            "spread_x":  0.42,
            "spread_z":  0.28,
            "alpha_min": 50,
            "alpha_max": 255,
        },
    ]

    # Load SMPL once
    with open(SMPL_MODEL_PATH, 'rb') as f:
        smpl_model = pickle.load(f, encoding='latin1')
    smpl_faces   = smpl_model['f']
    merged_parts = build_vertex_parts(smpl_model)

    rows = []
    for q in queries:
        ghost_img, caption = render_query(
            npz_file     = q["npz_file"],
            save_dir     = q["save_dir"],
            smpl_faces   = smpl_faces,
            merged_parts = merged_parts,
            caption      = q["caption"],
            gif_name     = q["gif_name"],
            fig_name     = q["fig_name"],
            n_keys       = q["n_keys"],
            spread_x     = q["spread_x"],
            spread_z     = q["spread_z"],
            alpha_min    = q["alpha_min"],
            alpha_max    = q["alpha_max"],
        )
        rows.append((ghost_img, caption))

    os.makedirs(os.path.dirname(COMBINED_FIG), exist_ok=True)
    save_combined_figure(rows, COMBINED_FIG)
    print("\nAll done!")


if __name__ == "__main__":
    main()