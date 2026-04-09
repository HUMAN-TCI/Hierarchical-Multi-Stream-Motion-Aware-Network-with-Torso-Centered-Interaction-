# # import os
# # import sys
# # import numpy as np
# # import matplotlib.pyplot as plt

# # # Add project root and necessary folders to Python path
# # ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# # sys.path.append(ROOT_DIR)
# # sys.path.append(os.path.join(ROOT_DIR, "common"))
# # sys.path.append(os.path.join(ROOT_DIR, "utils"))

# # from common import mmm
# # from utils.skeleton import Skeleton  # Now correctly pointing to your utils folder
# # from matplotlib.animation import FuncAnimation

# # def load_mmm_motion(file_path):
# #     """ Load motion data from MMM format. """
# #     print(f"[INFO] Loading MMM motion from: {file_path}")
# #     motion = mmm.MMMConverter.load(file_path)
# #     print(f"[INFO] Motion loaded with {len(motion['frames'])} frames and {len(motion['joints'])} joints.")
# #     return motion

# # def get_joint_positions(skeleton, motion_data):
# #     """ Get 3D joint positions from motion and skeleton """
# #     print("[INFO] Getting joint positions from skeleton and motion data...")
# #     positions = []
# #     for i, frame in enumerate(motion_data['frames']):
# #         joint_pos = skeleton.forward_kinematics(frame)
# #         if i == 0:
# #             print(f"[DEBUG] Sample joint positions from first frame:\n{joint_pos[:5]}")
# #         positions.append(joint_pos)
# #     print(f"[INFO] Computed positions for {len(positions)} frames.")
# #     return np.array(positions)

# # def plot_3d_motion(positions, joint_names, connections=None, title="3D Motion", interval=50):
# #     """ Plot the 3D motion using matplotlib """
# #     print("[INFO] Preparing to plot 3D motion...")

# #     fig = plt.figure()
# #     ax = fig.add_subplot(111, projection='3d')

# #     def update(frame_idx):
# #         ax.clear()
# #         ax.set_title(f"{title} - Frame {frame_idx}")
# #         pos = positions[frame_idx]

# #         ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c='blue', marker='o')

# #         # Draw bones if connections provided
# #         if connections:
# #             for parent, child in connections:
# #                 if parent >= len(pos) or child >= len(pos):
# #                     continue
# #                 x = [pos[parent, 0], pos[child, 0]]
# #                 y = [pos[parent, 1], pos[child, 1]]
# #                 z = [pos[parent, 2], pos[child, 2]]
# #                 ax.plot(x, y, z, c='black')

# #         ax.set_xlim([-1, 1])
# #         ax.set_ylim([-1, 1])
# #         ax.set_zlim([0, 2])

# #     ani = FuncAnimation(fig, update, frames=len(positions), interval=interval, repeat=True)
# #     plt.show()

# # def main():
# #     motion_file = "path_to_your_mmm_motion_file.xml"  # <-- Update this with your actual path

# #     if not os.path.exists(motion_file):
# #         print(f"[ERROR] Motion file not found: {motion_file}")
# #         return

# #     motion_data = load_mmm_motion(motion_file)

# #     # Use MMM skeleton
# #     print("[INFO] Initializing MMM skeleton...")
# #     skeleton = mmm.MMMConverter.get_skeleton()
# #     print(f"[INFO] Skeleton initialized with {len(skeleton.joint_names)} joints.")
# #     print("[INFO] Joint names:", skeleton.joint_names)

# #     joint_positions = get_joint_positions(skeleton, motion_data)

# #     # Define connections
# #     connections = []
# #     for joint, parent in skeleton.joint_parents.items():
# #         if parent != -1:
# #             connections.append((parent, joint))
# #     print(f"[INFO] Total bone connections defined: {len(connections)}")

# #     # Visualize
# #     plot_3d_motion(joint_positions, skeleton.joint_names, connections)

# # if __name__ == "__main__":
# #     main()

# import os
# import sys
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation

# # Add paths
# ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# sys.path.append(ROOT_DIR)
# sys.path.append(os.path.join(ROOT_DIR, "common"))
# sys.path.append(os.path.join(ROOT_DIR, "utils"))

# from common import mmm
# from utils.skeleton import Skeleton  # Ensure this matches your file structure

# def load_mmm_motion(file_path):
#     print(f"[INFO] Loading MMM motion from: {file_path}")
#     motion = mmm.MMMConverter.load(file_path)
#     print(f"[INFO] Loaded {len(motion['frames'])} frames with {len(motion['joints'])} joints.")
#     return motion

# def get_joint_positions(skeleton, motion_data):
#     print("[INFO] Calculating joint positions...")
#     positions = []
#     for i, frame in enumerate(motion_data['frames']):
#         joint_pos = skeleton.forward_kinematics(frame)
#         if i == 0:
#             print(f"[DEBUG] Sample positions (first frame):\n{joint_pos[:5]}")
#         positions.append(joint_pos)
#     return np.array(positions)

# def plot_3d_motion(positions, joint_names, connections=None, title="3D Motion", interval=50):
#     print("[INFO] Plotting 3D motion...")

#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')

#     def update(frame_idx):
#         ax.clear()
#         ax.set_title(f"{title} - Frame {frame_idx}")
#         pos = positions[frame_idx]

#         ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c='blue', marker='o')

#         if connections:
#             for parent, child in connections:
#                 x = [pos[parent, 0], pos[child, 0]]
#                 y = [pos[parent, 1], pos[child, 1]]
#                 z = [pos[parent, 2], pos[child, 2]]
#                 ax.plot(x, y, z, c='black')

#         ax.set_xlim([-1, 1])
#         ax.set_ylim([-1, 1])
#         ax.set_zlim([0, 2])

#     ani = FuncAnimation(fig, update, frames=len(positions), interval=interval, repeat=True)
#     plt.show()

# def main():
#     motion_file = "path_to_your_mmm_motion_file.xml"  # Replace this

#     if not os.path.exists(motion_file):
#         print(f"[ERROR] File not found: {motion_file}")
#         return

#     motion_data = load_mmm_motion(motion_file)
#     skeleton = mmm.MMMConverter.get_skeleton()
#     print(f"[INFO] Loaded skeleton with {len(skeleton.joint_names)} joints.")

#     joint_positions = get_joint_positions(skeleton, motion_data)

#     connections = [(parent, joint) for joint, parent in skeleton.joint_parents.items() if parent != -1]

#     plot_3d_motion(joint_positions, skeleton.joint_names, connections)

# if __name__ == "__main__":
#     main()
