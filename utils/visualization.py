# # # import matplotlib.pyplot as plt
# # # from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# # # import numpy as np
# # # from matplotlib.animation import FuncAnimation, PillowWriter

# # # def plot_3d_motion(save_path, kinematic_tree, joints, title, figsize=(10, 10), fps=120, radius=4, dist=5):
# # #     title_sp = title.split(' ')
# # #     if len(title_sp) > 10:
# # #         title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:])])

# # #     # Define initial plot setup
# # #     def init():
# # #         ax.set_xlim3d([-radius / 2, radius / 2])
# # #         ax.set_ylim3d([0, radius])
# # #         ax.set_zlim3d([0, radius])
# # #         fig.suptitle(title, fontsize=20)
# # #         ax.grid(False)

# # #     # Draw the XZ plane grid
# # #     def plot_xzPlane(minx, maxx, miny, minz, maxz):
# # #         verts = [[(minx, miny, minz), (minx, miny, maxz), (maxx, miny, maxz), (maxx, miny, minz)]]
# # #         xz_plane = Poly3DCollection(verts, color=(0.5, 0.5, 0.5, 0.5))
# # #         ax.add_collection3d(xz_plane)

# # #     # Reshape and setup data
# # #     data = joints.copy().reshape(len(joints), -1, 3)
# # #     fig = plt.figure(figsize=figsize)
# # #     ax = fig.add_subplot(projection="3d")
# # #     init()

# # #     MINS = data.min(axis=0).min(axis=0)
# # #     MAXS = data.max(axis=0).max(axis=0)
    
# # #     # Fresh vibrant color palette for chains
# # #     colors = [
# # #         '#e6194b',  # red
# # #         '#3cb44b',  # green
# # #         '#ffe119',  # yellow
# # #         '#4363d8',  # blue
# # #         '#f58231',  # orange
# # #         '#911eb4',  # purple
# # #         '#46f0f0',  # cyan
# # #         '#f032e6',  # magenta
# # #         '#bcf60c',  # lime
# # #         '#fabebe',  # pink
# # #         '#008080',  # teal
# # #         '#e6beff',  # lavender
# # #         '#9a6324',  # brown
# # #         '#fffac8',  # beige
# # #         '#800000'   # maroon
# # #     ]

# # #     # Extend colors if fewer than chains
# # #     while len(colors) < len(kinematic_tree):
# # #         colors.extend(colors)

# # #     colors = colors[:len(kinematic_tree)]  # Trim to exact length
    
# # #     frame_number = data.shape[0]

# # #     height_offset = MINS[1]
# # #     data[:, :, 1] -= height_offset
# # #     trajec = data[:, 0, [0, 2]]
    
# # #     data[..., 0] -= data[:, 0:1, 0]
# # #     data[..., 2] -= data[:, 0:1, 2]

# # #     # Optional debug print to verify chains
# # #     # for idx, chain in enumerate(kinematic_tree):
# # #     #     print(f"Chain {idx} joints: {chain}")

# # #     # Update function for each animation frame
# # #     def update(index):
# # #         ax.clear()
# # #         init()
# # #         ax.view_init(elev=120, azim=-90)
# # #         ax.dist = dist
        
# # #         plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 
# # #                      0, MINS[2] - trajec[index, 1], MAXS[2] - trajec[index, 1])

# # #         if index > 1:
# # #             ax.plot3D(trajec[:index, 0] - trajec[index, 0], np.zeros_like(trajec[:index, 0]), 
# # #                       trajec[:index, 1] - trajec[index, 1], linewidth=1.0, color='blue')
        
# # #         for i, (chain, color) in enumerate(zip(kinematic_tree, colors)):
# # #             linewidth = 4.0 if i < 5 else 2.0
# # #             ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], 
# # #                       linewidth=linewidth, color=color)

# # #         plt.axis('off')
# # #         ax.set_xticklabels([])
# # #         ax.set_yticklabels([])
# # #         ax.set_zticklabels([])

# # #     ani = FuncAnimation(fig, update, frames=frame_number, interval=1000/fps, repeat=False)

# # # import matplotlib.pyplot as plt
# # # from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# # # import numpy as np
# # # from matplotlib.animation import FuncAnimation

# # # def plot_3d_motion(save_path, kinematic_tree, joints, title, figsize=(10, 10), fps=120, radius=4, dist=5):
# # #     title = '\n'.join([title[i:i+60] for i in range(0, len(title), 60)])

# # #     data = joints.copy().reshape(len(joints), -1, 3)
# # #     fig = plt.figure(figsize=figsize)
# # #     ax = fig.add_subplot(projection="3d")

# # #     MINS = data.min(axis=0).min(axis=0)
# # #     MAXS = data.max(axis=0).max(axis=0)
    
# # #     colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4',
# # #               '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff',
# # #               '#9a6324', '#fffac8', '#800000']
# # #     while len(colors) < len(kinematic_tree):
# # #         colors.extend(colors)
# # #     colors = colors[:len(kinematic_tree)]

# # #     frame_number = data.shape[0]
# # #     height_offset = MINS[1]
# # #     data[:, :, 1] -= height_offset

# # #     trajec = data[:, 0, [0, 2]]
# # #     data[..., 0] -= data[:, 0:1, 0]
# # #     data[..., 2] -= data[:, 0:1, 2]

# # #     def init():
# # #         ax.set_xlim3d([-radius / 2, radius / 2])
# # #         ax.set_ylim3d([0, radius])
# # #         ax.set_zlim3d([0, radius])
# # #         fig.suptitle(title, fontsize=20)
# # #         ax.grid(False)

# # #     def plot_xzPlane(minx, maxx, miny, minz, maxz):
# # #         verts = [[(minx, miny, minz), (minx, miny, maxz), (maxx, miny, maxz), (maxx, miny, minz)]]
# # #         xz_plane = Poly3DCollection(verts, color=(0.5, 0.5, 0.5, 0.5))
# # #         ax.add_collection3d(xz_plane)

# # #     def update(index):
# # #         ax.clear()
# # #         init()
# # #         ax.view_init(elev=120, azim=-90)
# # #         ax.dist = dist

# # #         plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0],
# # #                      0, MINS[2] - trajec[index, 1], MAXS[2] - trajec[index, 1])

# # #         if index > 1:
# # #             ax.plot3D(trajec[:index, 0] - trajec[index, 0], 
# # #                       np.zeros_like(trajec[:index, 0]),
# # #                       trajec[:index, 1] - trajec[index, 1], linewidth=1.0, color='blue')

# # #         for i, (chain, color) in enumerate(zip(kinematic_tree, colors)):
# # #             linewidth = 4.0 if i < 5 else 2.0
# # #             ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2],
# # #                       linewidth=linewidth, color=color)

# # #         ax.set_xticklabels([])
# # #         ax.set_yticklabels([])
# # #         ax.set_zticklabels([])
# # #         plt.axis('off')

# # #     ani = FuncAnimation(fig, update, frames=frame_number, interval=1000/fps, repeat=False)
# # #     if save_path:
# # #         ani.save(save_path, writer='pillow', fps=fps)
# # #     else:
# # #         plt.show()
# # # //////////////////////////169 to 527/////////////////////////////////
# # # import matplotlib.pyplot as plt
# # # from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# # # import numpy as np
# # # from matplotlib.animation import FuncAnimation

# # # def plot_3d_motion(save_path, kinematic_tree, joints, title,
# # #                    figsize=(10, 10), fps=120, radius=4, dist=5):
# # #     title = '\n'.join([title[i:i+60] for i in range(0, len(title), 60)])

# # #     data = joints.copy().reshape(len(joints), -1, 3)
# # #     fig = plt.figure(figsize=figsize)
# # #     ax = fig.add_subplot(projection="3d")

# # #     MINS = data.min(axis=0).min(axis=0)
# # #     MAXS = data.max(axis=0).max(axis=0)

# # #     # Define color groups: torso/head (red), arms (green), legs (blue)
# # #     color_map = {
# # #         'torso': '#e6194b',  # red
# # #         'arms': '#3cb44b',   # green
# # #         'legs': '#4363d8'    # blue
# # #     }

# # #     # Define chain groupings
# # #     torso_ids = list(range(0, 5))         # Adjust according to your skeleton definition
# # #     arm_ids = list(range(5, 9))           # Left arm chain, right arm chain
# # #     leg_ids = list(range(9, len(kinematic_tree)))  # Left leg chain, right leg chain

# # #     # Assign colors
# # #     colors = []
# # #     for idx in range(len(kinematic_tree)):
# # #         if idx in torso_ids:
# # #             colors.append(color_map['torso'])
# # #         elif idx in arm_ids:
# # #             colors.append(color_map['arms'])
# # #         elif idx in leg_ids:
# # #             colors.append(color_map['legs'])
# # #         else:
# # #             colors.append('#000000')  # fallback (black)

# # #     frame_number = data.shape[0]
# # #     height_offset = MINS[1]
# # #     data[:, :, 1] -= height_offset

# # #     trajec = data[:, 0, [0, 2]]
# # #     data[..., 0] -= data[:, 0:1, 0]
# # #     data[..., 2] -= data[:, 0:1, 2]

# # #     def init():
# # #         ax.set_xlim3d([-radius / 2, radius / 2])
# # #         ax.set_ylim3d([0, radius])
# # #         ax.set_zlim3d([0, radius])
# # #         fig.suptitle(title, fontsize=20)
# # #         ax.grid(False)

# # #     def plot_xzPlane(minx, maxx, miny, minz, maxz):
# # #         verts = [[(minx, miny, minz), (minx, miny, maxz), (maxx, miny, maxz), (maxx, miny, minz)]]
# # #         xz_plane = Poly3DCollection(verts, color=(0.5, 0.5, 0.5, 0.5))
# # #         ax.add_collection3d(xz_plane)

# # #     def update(index):
# # #         ax.clear()
# # #         init()
# # #         ax.view_init(elev=120, azim=-90)
# # #         ax.dist = dist

# # #         plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0],
# # #                      0, MINS[2] - trajec[index, 1], MAXS[2] - trajec[index, 1])

# # #         if index > 1:
# # #             ax.plot3D(trajec[:index, 0] - trajec[index, 0], 
# # #                       np.zeros_like(trajec[:index, 0]),
# # #                       trajec[:index, 1] - trajec[index, 1], linewidth=1.0, color='blue')

# # #         for i, (chain, color) in enumerate(zip(kinematic_tree, colors)):
# # #             linewidth = 3.0
# # #             ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2],
# # #                       linewidth=linewidth, color=color)

# # #         ax.set_xticklabels([])
# # #         ax.set_yticklabels([])
# # #         ax.set_zticklabels([])
# # #         plt.axis('off')

# # #     ani = FuncAnimation(fig, update, frames=frame_number, interval=1000/fps, repeat=False)
# # #     if save_path:
# # #         ani.save(save_path, writer='pillow', fps=fps)
# # #     else:
# # #         plt.show()

# # # import os
# # # import numpy as np
# # # import matplotlib
# # # matplotlib.use('Agg')
# # # import matplotlib.pyplot as plt
# # # import matplotlib.patches as patches
# # # from matplotlib.patches import FancyBboxPatch
# # # from mpl_toolkits.mplot3d import Axes3D
# # # from mpl_toolkits.mplot3d.art3d import Line3DCollection
# # # import imageio
# # # from tqdm import tqdm


# # # # ── Color scheme for skeleton parts ───────────────────────────────────────────
# # # PART_COLORS = {
# # #     'left_leg':   '#22a84e',   # green
# # #     'right_leg':  '#1a8a3e',
# # #     'spine':      '#3460b4',   # blue
# # #     'left_arm':   '#c03030',   # red
# # #     'right_arm':  '#a02020',
# # #     'head':       '#3460b4',
# # # }

# # # # HumanML3D 22-joint kinematic chain → part label
# # # CHAIN_PART_MAP = {
# # #     (0,  2):  'left_leg',
# # #     (2,  5):  'left_leg',
# # #     (5,  8):  'left_leg',
# # #     (8,  11): 'left_leg',
# # #     (0,  1):  'right_leg',
# # #     (1,  4):  'right_leg',
# # #     (4,  7):  'right_leg',
# # #     (7,  10): 'right_leg',
# # #     (0,  3):  'spine',
# # #     (3,  6):  'spine',
# # #     (6,  9):  'spine',
# # #     (9,  12): 'spine',
# # #     (12, 15): 'head',
# # #     (9,  14): 'left_arm',
# # #     (14, 17): 'left_arm',
# # #     (17, 19): 'left_arm',
# # #     (19, 21): 'left_arm',
# # #     (9,  13): 'right_arm',
# # #     (13, 16): 'right_arm',
# # #     (16, 18): 'right_arm',
# # #     (18, 20): 'right_arm',
# # # }


# # # def _wrap_caption(text, max_chars=72):
# # #     """Clean and wrap caption text for paper figures."""
# # #     # Strip debug info — keep only the human-readable description
# # #     if ';' in text:
# # #         parts = text.split(';')
# # #         # Find the longest readable part (not the nDCG part)
# # #         clean = [p.strip() for p in parts if 'nDCG' not in p and len(p.strip()) > 10]
# # #         text = '. '.join(clean).strip(' .')
# # #     # Collapse whitespace
# # #     text = ' '.join(text.split())
# # #     # Wrap
# # #     words = text.split()
# # #     lines, line = [], []
# # #     for w in words:
# # #         if sum(len(x)+1 for x in line) + len(w) > max_chars:
# # #             lines.append(' '.join(line))
# # #             line = [w]
# # #         else:
# # #             line.append(w)
# # #     if line:
# # #         lines.append(' '.join(line))
# # #     return '\n'.join(lines)


# # # def _draw_skeleton_frame(ax, joints, kinematic_chain, alpha=1.0, lw=2.5):
# # #     """Draw one skeleton frame onto a 3D axis."""
# # #     for bone in kinematic_chain:
# # #         i, j = bone
# # #         color = PART_COLORS.get(CHAIN_PART_MAP.get(bone, 'spine'), '#3460b4')
# # #         xs = [joints[i, 0], joints[j, 0]]
# # #         ys = [joints[i, 1], joints[j, 1]]
# # #         zs = [joints[i, 2], joints[j, 2]]
# # #         ax.plot(xs, ys, zs,
# # #                 color=color, alpha=alpha,
# # #                 linewidth=lw, solid_capstyle='round')
# # #     # Joint dots
# # #     ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2],
# # #                c='white', s=8, zorder=5, alpha=alpha * 0.8,
# # #                edgecolors='#333333', linewidths=0.4)


# # # def _setup_3d_ax(fig, rect, joints_all, elev=15, azim=-75):
# # #     """Create and configure a 3D axis with proper bounds."""
# # #     ax = fig.add_axes(rect, projection='3d')

# # #     # Compute tight bounds from ALL frames
# # #     xmin, xmax = joints_all[:, :, 0].min(), joints_all[:, :, 0].max()
# # #     ymin, ymax = joints_all[:, :, 1].min(), joints_all[:, :, 1].max()
# # #     zmin, zmax = joints_all[:, :, 2].min(), joints_all[:, :, 2].max()

# # #     cx = (xmin + xmax) / 2
# # #     cy = (ymin + ymax) / 2
# # #     cz = (zmin + zmax) / 2
# # #     span = max(xmax - xmin, ymax - ymin, zmax - zmin) * 0.6

# # #     ax.set_xlim(cx - span, cx + span)
# # #     ax.set_ylim(cy - span, cy + span)
# # #     ax.set_zlim(zmin - 0.05, zmin + span * 2.0)

# # #     ax.view_init(elev=elev, azim=azim)

# # #     # Clean look
# # #     ax.set_axis_off()
# # #     ax.grid(False)
# # #     ax.xaxis.pane.fill = False
# # #     ax.yaxis.pane.fill = False
# # #     ax.zaxis.pane.fill = False
# # #     ax.xaxis.pane.set_edgecolor('none')
# # #     ax.yaxis.pane.set_edgecolor('none')
# # #     ax.zaxis.pane.set_edgecolor('none')

# # #     # Floor plane
# # #     floor_y = zmin - 0.02
# # #     xx = np.array([[cx-span, cx+span], [cx-span, cx+span]])
# # #     yy = np.array([[cy-span, cy-span], [cy+span, cy+span]])
# # #     zz = np.full_like(xx, floor_y)
# # #     ax.plot_surface(xx, yy, zz, alpha=0.18, color='#888888', zorder=0)

# # #     # Floor grid
# # #     for gx in np.linspace(cx-span, cx+span, 6):
# # #         ax.plot([gx, gx], [cy-span, cy+span], [floor_y, floor_y],
# # #                 color='#aaaaaa', alpha=0.25, linewidth=0.5, zorder=1)
# # #     for gy in np.linspace(cy-span, cy+span, 6):
# # #         ax.plot([cx-span, cx+span], [gy, gy], [floor_y, floor_y],
# # #                 color='#aaaaaa', alpha=0.25, linewidth=0.5, zorder=1)

# # #     return ax, (cx, cy, cz, span, zmin)


# # # # ── Green border ───────────────────────────────────────────────────────────────
# # # BORDER_COLOR = '#4CAF50'

# # # def _add_border(fig):
# # #     fig.add_artist(FancyBboxPatch(
# # #         (0.005, 0.005), 0.990, 0.990,
# # #         boxstyle='square,pad=0', linewidth=2.5,
# # #         edgecolor=BORDER_COLOR, facecolor='none',
# # #         transform=fig.transFigure, clip_on=False
# # #     ))


# # # # ==============================================================================
# # # # 1. FIXED plot_3d_motion  — caption BELOW, no overlap
# # # # ==============================================================================
# # # def plot_3d_motion(
# # #     save_path,
# # #     kinematic_chain,
# # #     joints,               # (T, 22, 3)
# # #     title='',
# # #     fps=20,
# # #     radius=1.5,
# # #     dist=2,
# # #     figsize=(8, 7),
# # #     elev=15,
# # #     azim=-75,
# # #     bg_color='#f2f2f0',
# # # ):
# # #     """
# # #     Render each frame of a skeleton motion as a GIF.
# # #     Caption appears BELOW the figure, never overlapping the skeleton.
# # #     """
# # #     T = joints.shape[0]
# # #     caption = _wrap_caption(title) if title else ''
# # #     n_caption_lines = caption.count('\n') + 1 if caption else 0

# # #     # Figure height: fixed motion area + caption area at bottom
# # #     fig_h = figsize[1]
# # #     cap_h = 0.06 * n_caption_lines if caption else 0.0   # fraction of figure

# # #     frames = []
# # #     for t in tqdm(range(T), desc=f"GIF {os.path.basename(save_path)}"):
# # #         fig = plt.figure(figsize=(figsize[0], fig_h), facecolor=bg_color)

# # #         # Motion axes occupies top portion, leaving room for caption
# # #         motion_rect = [0.02, cap_h + 0.02, 0.96, 0.96 - cap_h]
# # #         ax, bounds = _setup_3d_ax(fig, motion_rect, joints, elev=elev, azim=azim)

# # #         _draw_skeleton_frame(ax, joints[t], kinematic_chain, alpha=1.0, lw=2.5)

# # #         # Caption — bottom centre, below motion
# # #         if caption:
# # #             fig.text(
# # #                 0.50, cap_h * 0.45,
# # #                 caption,
# # #                 ha='center', va='center',
# # #                 fontsize=9.5, fontstyle='italic',
# # #                 fontfamily='DejaVu Serif',
# # #                 color='#222222',
# # #                 wrap=False,
# # #                 transform=fig.transFigure
# # #             )

# # #         _add_border(fig)

        
# # #         # buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
# # #         # w, h = fig.canvas.get_width_height()
# # #         # frame = buf.reshape(h, w, 3)
# # #         # frames.append(frame)
# # #         fig.canvas.draw()
# # #         buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
# # #         w, h = fig.canvas.get_width_height()
# # #         frame = buf.reshape(h, w, 4)[:, :, :3]   # RGBA → RGB
# # #         plt.close(fig)

# # #     imageio.mimsave(save_path, frames, fps=fps)


# # # # ==============================================================================
# # # # 2. NEW plot_skeleton_ghost — keyframe ghost strip for paper figures
# # # # ==============================================================================
# # # def _extract_keyframes(joints, n_keys=6):
# # #     """Motion-arc keyframe extraction — picks frames with most pose change."""
# # #     T = len(joints)
# # #     vel = np.zeros(T)
# # #     for t in range(1, T):
# # #         vel[t] = np.mean(np.linalg.norm(joints[t] - joints[t-1], axis=1))
# # #     cum = np.cumsum(vel)
# # #     if cum[-1] == 0:
# # #         return list(np.linspace(0, T-1, n_keys, dtype=int))
# # #     cum_n = cum / cum[-1]
# # #     targets = np.linspace(0, 1, n_keys)
# # #     idxs = sorted(set(
# # #         [0] + [int(np.argmin(np.abs(cum_n - a))) for a in targets] + [T-1]
# # #     ))
# # #     while len(idxs) > n_keys:
# # #         gaps = [cum[idxs[i+1]] - cum[idxs[i]] for i in range(1, len(idxs)-1)]
# # #         idxs.pop(1 + int(np.argmin(gaps)))
# # #     return idxs


# # # def plot_skeleton_ghost(
# # #     save_path,
# # #     kinematic_chain,
# # #     joints,               # (T, 22, 3)
# # #     caption='',
# # #     n_keys=6,
# # #     figsize=(16, 5),
# # #     elev=18,
# # #     azim=-80,
# # #     bg_color='#f2f2f0',
# # #     border_color='#4CAF50',
# # #     dpi=150,
# # # ):
# # #     """
# # #     Save a conference-paper ghost-trail figure showing key skeleton poses.

# # #     Layout:
# # #       - n_keys subplots side by side, each showing one key pose
# # #       - First and last frames fully opaque, middle frames faded
# # #       - Caption centred below
# # #       - Green border around whole figure
# # #     """
# # #     idxs   = _extract_keyframes(joints, n_keys=n_keys)
# # #     n      = len(idxs)

# # #     # Alpha: sin curve — solid ends, faded middle
# # #     alphas = []
# # #     for i in range(n):
# # #         t = i / max(n-1, 1)
# # #         alphas.append(float(1.0 - 0.78 * np.sin(t * np.pi)))

# # #     clean_caption = _wrap_caption(caption) if caption else ''
# # #     cap_lines     = clean_caption.count('\n') + 1 if clean_caption else 0
# # #     cap_h_in      = cap_lines * 0.22 + 0.15   # inches for caption area

# # #     fig = plt.figure(
# # #         figsize=(figsize[0], figsize[1] + cap_h_in),
# # #         facecolor=bg_color
# # #     )

# # #     cap_frac  = cap_h_in / (figsize[1] + cap_h_in)
# # #     plot_frac = 1.0 - cap_frac

# # #     # Add subplots in the top plot_frac of the figure
# # #     axes = []
# # #     for col in range(n):
# # #         left   = 0.01 + col * (0.98 / n)
# # #         width  = (0.98 / n) - 0.005
# # #         bottom = cap_frac + 0.02
# # #         height = plot_frac - 0.04
# # #         ax = fig.add_axes([left, bottom, width, height], projection='3d')
# # #         axes.append(ax)

# # #     # Shared bounds across all frames
# # #     xmin, xmax = joints[:,:,0].min(), joints[:,:,0].max()
# # #     ymin, ymax = joints[:,:,1].min(), joints[:,:,1].max()
# # #     zmin, zmax = joints[:,:,2].min(), joints[:,:,2].max()
# # #     cx = (xmin+xmax)/2; cy = (ymin+ymax)/2
# # #     span = max(xmax-xmin, ymax-ymin, zmax-zmin) * 0.58

# # #     for col, (frame_idx, alpha) in enumerate(zip(idxs, alphas)):
# # #         ax = axes[col]
# # #         jf = joints[frame_idx]

# # #         ax.set_xlim(cx-span, cx+span)
# # #         ax.set_ylim(cy-span, cy+span)
# # #         ax.set_zlim(zmin-0.05, zmin + span*2.0)
# # #         ax.view_init(elev=elev, azim=azim)
# # #         ax.set_axis_off(); ax.grid(False)
# # #         ax.xaxis.pane.fill = False
# # #         ax.yaxis.pane.fill = False
# # #         ax.zaxis.pane.fill = False
# # #         ax.xaxis.pane.set_edgecolor('none')
# # #         ax.yaxis.pane.set_edgecolor('none')
# # #         ax.zaxis.pane.set_edgecolor('none')

# # #         # Floor
# # #         floor_z = zmin - 0.02
# # #         xx = np.array([[cx-span, cx+span],[cx-span, cx+span]])
# # #         yy = np.array([[cy-span, cy-span],[cy+span, cy+span]])
# # #         zz = np.full_like(xx, floor_z)
# # #         ax.plot_surface(xx, yy, zz,
# # #                         alpha=0.15 * alpha, color='#888888', zorder=0)
# # #         for gx in np.linspace(cx-span, cx+span, 5):
# # #             ax.plot([gx,gx],[cy-span,cy+span],[floor_z,floor_z],
# # #                     color='#aaaaaa', alpha=0.20*alpha, linewidth=0.5)
# # #         for gy in np.linspace(cy-span, cy+span, 5):
# # #             ax.plot([cx-span,cx+span],[gy,gy],[floor_z,floor_z],
# # #                     color='#aaaaaa', alpha=0.20*alpha, linewidth=0.5)

# # #         # Skeleton
# # #         _draw_skeleton_frame(ax, jf, kinematic_chain,
# # #                              alpha=alpha, lw=2.8 if alpha > 0.85 else 1.8)

# # #         # Frame label (t=N) on fully opaque frames
# # #         if alpha > 0.85:
# # #             ax.set_title(f't={frame_idx}', fontsize=7,
# # #                          color='#444444', pad=1, fontfamily='monospace')

# # #     # Caption
# # #     if clean_caption:
# # #         fig.text(
# # #             0.50, cap_frac * 0.45,
# # #             clean_caption,
# # #             ha='center', va='center',
# # #             fontsize=10.5, fontstyle='italic',
# # #             fontfamily='DejaVu Serif',
# # #             color='#111111',
# # #             transform=fig.transFigure
# # #         )

# # #     # Border
# # #     fig.add_artist(FancyBboxPatch(
# # #         (0.005, 0.005), 0.990, 0.990,
# # #         boxstyle='square,pad=0', linewidth=2.5,
# # #         edgecolor=border_color, facecolor='none',
# # #         transform=fig.transFigure, clip_on=False
# # #     ))

# # #     plt.savefig(save_path, dpi=dpi, bbox_inches='tight',
# # #                 facecolor=fig.get_facecolor())
# # #     plt.close(fig)
# # #     print(f"Saved skeleton ghost figure: {save_path}")
# # import os
# # import re
# # import matplotlib
# # matplotlib.use('Agg')   # MUST be before any other matplotlib import
# # import matplotlib.pyplot as plt
# # import matplotlib.patches as patches
# # from matplotlib.patches import FancyBboxPatch
# # from mpl_toolkits.mplot3d import Axes3D
# # import numpy as np
# # import imageio
# # from tqdm import tqdm


# # # ── Color scheme ───────────────────────────────────────────────────────────────
# # PART_COLORS = {
# #     'left_leg':  '#22a84e',
# #     'right_leg': '#1a8a3e',
# #     'spine':     '#3460b4',
# #     'left_arm':  '#c03030',
# #     'right_arm': '#a02020',
# #     'head':      '#3460b4',
# # }

# # CHAIN_PART_MAP = {
# #     (0,  2):  'left_leg',
# #     (2,  5):  'left_leg',
# #     (5,  8):  'left_leg',
# #     (8,  11): 'left_leg',
# #     (0,  1):  'right_leg',
# #     (1,  4):  'right_leg',
# #     (4,  7):  'right_leg',
# #     (7,  10): 'right_leg',
# #     (0,  3):  'spine',
# #     (3,  6):  'spine',
# #     (6,  9):  'spine',
# #     (9,  12): 'spine',
# #     (12, 15): 'head',
# #     (9,  14): 'left_arm',
# #     (14, 17): 'left_arm',
# #     (17, 19): 'left_arm',
# #     (19, 21): 'left_arm',
# #     (9,  13): 'right_arm',
# #     (13, 16): 'right_arm',
# #     (16, 18): 'right_arm',
# #     (18, 20): 'right_arm',
# # }

# # BORDER_COLOR = '#4CAF50'


# # # ── Helpers ────────────────────────────────────────────────────────────────────
# # def _wrap_caption(text, max_chars=72):
# #     """Extract the single most relevant description only."""
# #     if not text:
# #         return ''

# #     # Remove leading index like "0; " or "1; "
# #     text = re.sub(r'^\d+;\s*', '', text.strip())

# #     # Split on semicolons — take ONLY THE FIRST clean human part
# #     if ';' in text:
# #         parts = text.split(';')
# #         for p in parts:
# #             p = p.strip()
# #             # Skip any part containing metrics or file paths
# #             if any(k in p for k in ['nDCG', 'spacy', 'spice', '=', 'npy', '.gif']):
# #                 continue
# #             if len(p) > 8:
# #                 text = p   # ← take first valid part and STOP
# #                 break

# #     # Strip any remaining metric patterns
# #     text = re.sub(r'\(nDCG[^)]*\)', '', text)
# #     text = re.sub(r'spacy\s*=\s*[\d.]+', '', text)
# #     text = re.sub(r'spice\s*=\s*[\d.]+', '', text)
# #     text = re.sub(r'\(\s*\)', '', text)
# #     text = text.strip(' .,;)(')
# #     text = ' '.join(text.split())

# #     if text:
# #         text = text[0].upper() + text[1:]

# #     # Word-wrap
# #     words = text.split()
# #     lines, line = [], []
# #     for w in words:
# #         if sum(len(x) + 1 for x in line) + len(w) > max_chars:
# #             lines.append(' '.join(line))
# #             line = [w]
# #         else:
# #             line.append(w)
# #     if line:
# #         lines.append(' '.join(line))
# #     return '\n'.join(lines)


# # def _draw_skeleton_frame(ax, joints, kinematic_chain, alpha=1.0, lw=2.5):
# #     """Draw one skeleton pose onto a 3D axis."""
# #     for bone in kinematic_chain:
# #         i, j = bone
# #         color = PART_COLORS.get(CHAIN_PART_MAP.get(bone, 'spine'), '#3460b4')
# #         ax.plot(
# #             [joints[i, 0], joints[j, 0]],
# #             [joints[i, 1], joints[j, 1]],
# #             [joints[i, 2], joints[j, 2]],
# #             color=color, alpha=alpha,
# #             linewidth=lw, solid_capstyle='round'
# #         )
# #     ax.scatter(
# #         joints[:, 0], joints[:, 1], joints[:, 2],
# #         c='white', s=8, zorder=5,
# #         alpha=min(alpha * 0.9, 1.0),
# #         edgecolors='#333333', linewidths=0.4
# #     )


# # def _configure_ax(ax, joints_all, elev=15, azim=-75):
# #     """Set axis bounds, camera, floor plane, and grid."""
# #     xmin, xmax = joints_all[:, :, 0].min(), joints_all[:, :, 0].max()
# #     ymin, ymax = joints_all[:, :, 1].min(), joints_all[:, :, 1].max()
# #     zmin, zmax = joints_all[:, :, 2].min(), joints_all[:, :, 2].max()

# #     cx = (xmin + xmax) / 2
# #     cy = (ymin + ymax) / 2
# #     span = max(xmax - xmin, ymax - ymin, zmax - zmin) * 0.60

# #     ax.set_xlim(cx - span, cx + span)
# #     ax.set_ylim(cy - span, cy + span)
# #     ax.set_zlim(zmin - 0.05, zmin + span * 2.1)
# #     ax.view_init(elev=elev, azim=azim)

# #     ax.set_axis_off()
# #     ax.grid(False)
# #     for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
# #         pane.fill = False
# #         pane.set_edgecolor('none')

# #     # Floor plane
# #     fz = zmin - 0.02
# #     xx = np.array([[cx-span, cx+span], [cx-span, cx+span]])
# #     yy = np.array([[cy-span, cy-span], [cy+span, cy+span]])
# #     zz = np.full_like(xx, fz)
# #     ax.plot_surface(xx, yy, zz, alpha=0.20, color='#888888', zorder=0)

# #     # Floor grid
# #     for gx in np.linspace(cx-span, cx+span, 6):
# #         ax.plot([gx, gx], [cy-span, cy+span], [fz, fz],
# #                 color='#aaaaaa', alpha=0.28, linewidth=0.5)
# #     for gy in np.linspace(cy-span, cy+span, 6):
# #         ax.plot([cx-span, cx+span], [gy, gy], [fz, fz],
# #                 color='#aaaaaa', alpha=0.28, linewidth=0.5)

# #     return cx, cy, zmin, span


# # def _add_border(fig, color=BORDER_COLOR):
# #     fig.add_artist(FancyBboxPatch(
# #         (0.005, 0.005), 0.990, 0.990,
# #         boxstyle='square,pad=0', linewidth=2.5,
# #         edgecolor=color, facecolor='none',
# #         transform=fig.transFigure, clip_on=False
# #     ))


# # def _fig_to_rgb(fig):
# #     """
# #     Convert a matplotlib figure to an (H, W, 3) uint8 numpy array.
# #     Compatible with matplotlib >= 3.8 (buffer_rgba replaces tostring_rgb).
# #     """
# #     fig.canvas.draw()
# #     # buffer_rgba works on all modern matplotlib versions
# #     buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
# #     w, h = fig.canvas.get_width_height()
# #     return buf.reshape(h, w, 4)[:, :, :3]   # drop alpha → RGB


# # # ==============================================================================
# # # 1. plot_3d_motion — per-frame GIF, caption below (no overlap)
# # # ==============================================================================
# # def plot_3d_motion(
# #     save_path,
# #     kinematic_chain,
# #     joints,               # (T, 22, 3)
# #     title='',
# #     fps=20,
# #     radius=1.5,
# #     dist=2,
# #     figsize=(8, 7),
# #     elev=15,
# #     azim=-75,
# #     bg_color='#f2f2f0',
# # ):
# #     T = joints.shape[0]
# #     caption = _wrap_caption(title)
# #     n_lines = caption.count('\n') + 1 if caption else 0
# #     cap_frac = min(0.06 * n_lines, 0.22) if caption else 0.0

# #     frames = []
# #     for t in tqdm(range(T), desc=f"GIF {os.path.basename(save_path)}"):
# #         fig = plt.figure(figsize=figsize, facecolor=bg_color, dpi=100)

# #         # Motion axes — top portion
# #         motion_bottom = cap_frac + 0.01
# #         ax = fig.add_axes(
# #             [0.02, motion_bottom, 0.96, 0.97 - motion_bottom],
# #             projection='3d'
# #         )
# #         _configure_ax(ax, joints, elev=elev, azim=azim)
# #         _draw_skeleton_frame(ax, joints[t], kinematic_chain, alpha=1.0, lw=2.5)

# #         # Caption — bottom strip, never overlaps motion
# #         if caption:
# #             fig.text(
# #                 0.50, cap_frac * 0.42,
# #                 caption,
# #                 ha='center', va='center',
# #                 fontsize=9.5, fontstyle='italic',
# #                 fontfamily='DejaVu Serif',
# #                 color='#222222',
# #                 transform=fig.transFigure
# #             )

# #         _add_border(fig)

# #         frame = _fig_to_rgb(fig)   # ← fixed: uses buffer_rgba
# #         frames.append(frame)
# #         plt.close(fig)

# #     imageio.mimsave(save_path, frames, fps=fps)
# #     print(f"Saved GIF: {save_path}")


# # # ==============================================================================
# # # 2. plot_skeleton_ghost — keyframe ghost strip for paper
# # # ==============================================================================
# # def _extract_keyframes(joints, n_keys=6):
# #     T = len(joints)
# #     vel = np.zeros(T)
# #     for t in range(1, T):
# #         vel[t] = np.mean(np.linalg.norm(joints[t] - joints[t-1], axis=1))
# #     cum = np.cumsum(vel)
# #     if cum[-1] == 0:
# #         return list(np.linspace(0, T-1, n_keys, dtype=int))
# #     cum_n   = cum / cum[-1]
# #     targets = np.linspace(0, 1, n_keys)
# #     idxs    = sorted(set(
# #         [0] + [int(np.argmin(np.abs(cum_n - a))) for a in targets] + [T-1]
# #     ))
# #     while len(idxs) > n_keys:
# #         gaps = [cum[idxs[i+1]] - cum[idxs[i]] for i in range(1, len(idxs)-1)]
# #         idxs.pop(1 + int(np.argmin(gaps)))
# #     return idxs

# # def plot_skeleton_ghost(
# #     save_path,
# #     kinematic_chain,
# #     joints,               # (T, 22, 3)
# #     caption='',
# #     n_keys=6,
# #     figsize=(14, 6),
# #     elev=20,
# #     azim=-75,
# #     bg_color='#f2f2f0',
# #     border_color=BORDER_COLOR,
# #     dpi=150,
# #     spread=0.55,          # spacing between ghost figures (fraction of scale)
# # ):
# #     """
# #     All key poses in ONE single 3D scene, spread along X axis.
# #     First and last solid, middle faded — like the SMPL ghost renderer.
# #     """
# #     idxs   = _extract_keyframes(joints, n_keys=n_keys)
# #     n      = len(idxs)
# #     alphas = [float(1.0 - 0.78 * np.sin(i / max(n - 1, 1) * np.pi))
# #               for i in range(n)]

# #     clean_cap = _wrap_caption(caption)
# #     cap_lines = clean_cap.count('\n') + 1 if clean_cap else 0
# #     cap_h_in  = cap_lines * 0.22 + 0.18
# #     total_h   = figsize[1] + cap_h_in
# #     cap_frac  = cap_h_in / total_h
# #     plot_frac = 1.0 - cap_frac

# #     fig = plt.figure(figsize=(figsize[0], total_h), facecolor=bg_color)

# #     # ONE axes covering the whole top area
# #     ax = fig.add_axes(
# #         [0.0, cap_frac + 0.01, 1.0, plot_frac - 0.02],
# #         projection='3d'
# #     )

# #     # Compute scale from all joints
# #     scale = max(
# #         joints[:,:,0].max() - joints[:,:,0].min(),
# #         joints[:,:,1].max() - joints[:,:,1].min(),
# #         joints[:,:,2].max() - joints[:,:,2].min(),
# #     ) * 0.52

# #     # Center of the motion
# #     cx0 = (joints[:,:,0].max() + joints[:,:,0].min()) / 2
# #     cy0 = (joints[:,:,2].max() + joints[:,:,2].min()) / 2   # depth (Z→Y)
# #     zmin = joints[:,:,1].min()   # foot level (Y→Z)

# #     # Total spread width
# #     total_spread = (n - 1) * spread * scale

# #     # Draw each key pose offset along X
# #     for i, (fidx, alpha) in enumerate(zip(idxs, alphas)):
# #         jf = joints[fidx].copy()

# #         # Offset along X to spread figures side by side
# #         x_offset = -total_spread/2 + i * spread * scale

# #         for bone in kinematic_chain:
# #             bi, bj = bone
# #             color = PART_COLORS.get(CHAIN_PART_MAP.get(bone, 'spine'), '#3460b4')
# #             ax.plot(
# #                 [jf[bi,0] - cx0 + x_offset,  jf[bj,0] - cx0 + x_offset],
# #                 [jf[bi,2] - cy0,              jf[bj,2] - cy0],        # depth
# #                 [jf[bi,1] - zmin,             jf[bj,1] - zmin],        # height
# #                 color=color, alpha=alpha,
# #                 linewidth=3.0 if alpha > 0.85 else 1.8,
# #                 solid_capstyle='round'
# #             )
# #         ax.scatter(
# #             jf[:,0] - cx0 + x_offset,
# #             jf[:,2] - cy0,
# #             jf[:,1] - zmin,
# #             c='white', s=8, zorder=5,
# #             alpha=min(alpha * 0.85, 1.0),
# #             edgecolors='#555555', linewidths=0.3
# #         )

# #     # Axis limits — fit all spread figures
# #     pad = scale * 0.5
# #     ax.set_xlim(-total_spread/2 - pad, total_spread/2 + pad)
# #     ax.set_ylim(-scale, scale)
# #     ax.set_zlim(-0.02, scale * 2.2)

# #     ax.view_init(elev=elev, azim=azim)
# #     ax.set_axis_off(); ax.grid(False)
# #     for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
# #         pane.fill = False
# #         pane.set_edgecolor('none')

# #     # Floor — wide enough to cover all figures
# #     fz = -0.01
# #     hw = total_spread/2 + pad
# #     hd = scale * 0.9
# #     xx = np.array([[-hw, hw], [-hw, hw]])
# #     yy = np.array([[-hd, -hd], [hd, hd]])
# #     zz = np.full_like(xx, fz)
# #     ax.plot_surface(xx, yy, zz, alpha=0.28, color='#888888', zorder=0)
# #     for gx in np.linspace(-hw, hw, 10):
# #         ax.plot([gx,gx], [-hd,hd], [fz,fz],
# #                 color='#aaaaaa', alpha=0.28, linewidth=0.5)
# #     for gy in np.linspace(-hd, hd, 6):
# #         ax.plot([-hw,hw], [gy,gy], [fz,fz],
# #                 color='#aaaaaa', alpha=0.28, linewidth=0.5)

# #     # Caption
# #     if clean_cap:
# #         fig.text(
# #             0.50, cap_frac * 0.38,
# #             clean_cap,
# #             ha='center', va='center',
# #             fontsize=10, fontstyle='italic',
# #             fontfamily='DejaVu Serif',
# #             color='#111111',
# #             transform=fig.transFigure
# #         )

# #     # Border
# #     fig.add_artist(FancyBboxPatch(
# #         (0.005, 0.005), 0.990, 0.990,
# #         boxstyle='square,pad=0', linewidth=2.5,
# #         edgecolor=border_color, facecolor='none',
# #         transform=fig.transFigure, clip_on=False
# #     ))

# #     plt.savefig(save_path, dpi=dpi, bbox_inches='tight',
# #                 facecolor=fig.get_facecolor())
# #     plt.close(fig)
# #     print(f"Saved ghost figure: {save_path}")

# # # def plot_skeleton_ghost(
# # #     save_path,
# # #     kinematic_chain,
# # #     joints,               # (T, 22, 3)
# # #     caption='',
# # #     n_keys=6,
# # #     figsize=(16, 5),
# # #     elev=18,
# # #     azim=-80,
# # #     bg_color='#f2f2f0',
# # #     border_color=BORDER_COLOR,
# # #     dpi=150,
# # # ):
# # #     """
# # #     Save a ghost-trail figure: n_keys key poses side by side.
# # #     First and last frames are solid; middle frames are faded.
# # #     Caption sits cleanly below. Green border around the whole figure.
# # #     """
# # #     idxs = _extract_keyframes(joints, n_keys=n_keys)
# # #     n    = len(idxs)

# # #     # Sine-curve alpha: solid ends, faded middle
# # #     alphas = [float(1.0 - 0.78 * np.sin(i / max(n-1, 1) * np.pi))
# # #               for i in range(n)]

# # #     clean_cap  = _wrap_caption(caption)
# # #     cap_lines  = clean_cap.count('\n') + 1 if clean_cap else 0
# # #     cap_h_in   = cap_lines * 0.22 + 0.18

# # #     total_h    = figsize[1] + cap_h_in
# # #     cap_frac   = cap_h_in / total_h
# # #     plot_frac  = 1.0 - cap_frac

# # #     fig = plt.figure(figsize=(figsize[0], total_h),
# # #                      facecolor=bg_color)

# # #     # Shared axis bounds
# # #     xmin, xmax = joints[:,:,0].min(), joints[:,:,0].max()
# # #     ymin, ymax = joints[:,:,1].min(), joints[:,:,1].max()
# # #     zmin, zmax = joints[:,:,2].min(), joints[:,:,2].max()
# # #     cx   = (xmin+xmax)/2;  cy = (ymin+ymax)/2
# # #     span = max(xmax-xmin, ymax-ymin, zmax-zmin) * 0.58
# # #     fz   = zmin - 0.02

# # #     panel_w = 0.98 / n
# # #     for col, (fidx, alpha) in enumerate(zip(idxs, alphas)):
# # #         left   = 0.01 + col * panel_w
# # #         bottom = cap_frac + 0.02
# # #         width  = panel_w - 0.006
# # #         height = plot_frac - 0.04

# # #         ax = fig.add_axes([left, bottom, width, height], projection='3d')

# # #         ax.set_xlim(cx-span, cx+span)
# # #         ax.set_ylim(cy-span, cy+span)
# # #         ax.set_zlim(zmin-0.05, zmin+span*2.1)
# # #         ax.view_init(elev=elev, azim=azim)
# # #         ax.set_axis_off(); ax.grid(False)
# # #         for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
# # #             pane.fill = False
# # #             pane.set_edgecolor('none')

# # #         # Floor
# # #         xx = np.array([[cx-span,cx+span],[cx-span,cx+span]])
# # #         yy = np.array([[cy-span,cy-span],[cy+span,cy+span]])
# # #         zz = np.full_like(xx, fz)
# # #         ax.plot_surface(xx, yy, zz,
# # #                         alpha=0.18 * alpha, color='#888888', zorder=0)
# # #         for gx in np.linspace(cx-span, cx+span, 5):
# # #             ax.plot([gx,gx],[cy-span,cy+span],[fz,fz],
# # #                     color='#aaaaaa', alpha=0.22*alpha, linewidth=0.5)
# # #         for gy in np.linspace(cy-span, cy+span, 5):
# # #             ax.plot([cx-span,cx+span],[gy,gy],[fz,fz],
# # #                     color='#aaaaaa', alpha=0.22*alpha, linewidth=0.5)

# # #         # Skeleton
# # #         lw = 2.8 if alpha > 0.85 else 1.8
# # #         _draw_skeleton_frame(ax, joints[fidx], kinematic_chain,
# # #                              alpha=alpha, lw=lw)

# # #         # Frame label on solid poses only
# # #         if alpha > 0.85:
# # #             ax.set_title(f't={fidx}', fontsize=7, color='#444444',
# # #                          pad=1, fontfamily='monospace')

# # #     # Caption
# # #     if clean_cap:
# # #         fig.text(
# # #             0.50, cap_frac * 0.42,
# # #             clean_cap,
# # #             ha='center', va='center',
# # #             fontsize=11, fontstyle='italic',
# # #             fontfamily='DejaVu Serif',
# # #             color='#111111',
# # #             transform=fig.transFigure
# # #         )

# # #     # Border
# # #     fig.add_artist(FancyBboxPatch(
# # #         (0.005, 0.005), 0.990, 0.990,
# # #         boxstyle='square,pad=0', linewidth=2.5,
# # #         edgecolor=border_color, facecolor='none',
# # #         transform=fig.transFigure, clip_on=False
# # #     ))

# # #     plt.savefig(save_path, dpi=dpi, bbox_inches='tight',
# # #                 facecolor=fig.get_facecolor())
# # #     plt.close(fig)
# # #     print(f"Saved ghost figure: {save_path}")
# import os
# import re
# import matplotlib
# matplotlib.use('Agg')   # MUST be before any other matplotlib import
# import matplotlib.pyplot as plt
# from matplotlib.patches import FancyBboxPatch
# from mpl_toolkits.mplot3d import Axes3D
# import numpy as np
# import imageio
# from tqdm import tqdm


# # ── Color scheme ───────────────────────────────────────────────────────────────
# PART_COLORS = {
#     'left_leg':  '#22a84e',
#     'right_leg': '#1a8a3e',
#     'spine':     '#3460b4',
#     'left_arm':  '#c03030',
#     'right_arm': '#a02020',
#     'head':      '#3460b4',
# }

# CHAIN_PART_MAP = {
#     (0,  2):  'left_leg',
#     (2,  5):  'left_leg',
#     (5,  8):  'left_leg',
#     (8,  11): 'left_leg',
#     (0,  1):  'right_leg',
#     (1,  4):  'right_leg',
#     (4,  7):  'right_leg',
#     (7,  10): 'right_leg',
#     (0,  3):  'spine',
#     (3,  6):  'spine',
#     (6,  9):  'spine',
#     (9,  12): 'spine',
#     (12, 15): 'head',
#     (9,  14): 'left_arm',
#     (14, 17): 'left_arm',
#     (17, 19): 'left_arm',
#     (19, 21): 'left_arm',
#     (9,  13): 'right_arm',
#     (13, 16): 'right_arm',
#     (16, 18): 'right_arm',
#     (18, 20): 'right_arm',
# }

# BORDER_COLOR = '#4CAF50'


# # ── Caption: extract first clean sentence only ─────────────────────────────────
# def _wrap_caption(text, max_chars=80):
#     """Extract the single most relevant description only."""
#     if not text:
#         return ''

#     # Remove leading index like "0; " or "1; "
#     text = re.sub(r'^\d+;\s*', '', text.strip())

#     # Split on semicolons — take ONLY THE FIRST clean human part
#     if ';' in text:
#         parts = text.split(';')
#         for p in parts:
#             p = p.strip()
#             if any(k in p for k in ['nDCG', 'spacy', 'spice', '=', 'npy', '.gif']):
#                 continue
#             if len(p) > 8:
#                 text = p
#                 break

#     # Strip any remaining metric patterns
#     text = re.sub(r'\(nDCG[^)]*\)', '', text)
#     text = re.sub(r'spacy\s*=\s*[\d.]+', '', text)
#     text = re.sub(r'spice\s*=\s*[\d.]+', '', text)
#     text = re.sub(r'nDCG\s*[:\-]?\s*[\d.]*', '', text)
#     text = re.sub(r'\(\s*\)', '', text)
#     text = text.strip(' .,;)(')
#     text = ' '.join(text.split())

#     if text:
#         text = text[0].upper() + text[1:]

#     # Word-wrap
#     words = text.split()
#     lines, line = [], []
#     for w in words:
#         if sum(len(x) + 1 for x in line) + len(w) > max_chars:
#             lines.append(' '.join(line))
#             line = [w]
#         else:
#             line.append(w)
#     if line:
#         lines.append(' '.join(line))
#     return '\n'.join(lines)


# # ── Skeleton drawing ───────────────────────────────────────────────────────────
# def _draw_bones(ax, joints, kinematic_chain, alpha=1.0, lw=2.5,
#                 x_off=0.0, y_off=0.0, z_off=0.0):
#     """
#     Draw one skeleton pose.
#     HumanML3D: joints[:,0]=X(width), joints[:,1]=Y(height), joints[:,2]=Z(depth)
#     Matplotlib 3D: 3rd arg = vertical axis
#     So we plot: ax_x=joints_X, ax_y=joints_Z, ax_z=joints_Y  (Y↔Z swap)
#     """
#     for bone in kinematic_chain:
#         i, j = bone
#         color = PART_COLORS.get(CHAIN_PART_MAP.get(bone, 'spine'), '#3460b4')
#         ax.plot(
#             [joints[i, 0] + x_off, joints[j, 0] + x_off],
#             [joints[i, 2] + y_off, joints[j, 2] + y_off],
#             [joints[i, 1] + z_off, joints[j, 1] + z_off],
#             color=color, alpha=alpha,
#             linewidth=lw, solid_capstyle='round', zorder=3
#         )
#     ax.scatter(
#         joints[:, 0] + x_off,
#         joints[:, 2] + y_off,
#         joints[:, 1] + z_off,
#         c='white', s=10, zorder=4,
#         alpha=min(alpha * 0.9, 1.0),
#         edgecolors='#333333', linewidths=0.4
#     )


# def _add_floor(ax, cx, cy, span_x, span_y, fz, alpha=1.0):
#     """Draw a grey tiled floor plane."""
#     hw_x = span_x * 0.9
#     hw_y = span_y * 0.7
#     xx = np.array([[cx - hw_x, cx + hw_x], [cx - hw_x, cx + hw_x]])
#     yy = np.array([[cy - hw_y, cy - hw_y], [cy + hw_y, cy + hw_y]])
#     zz = np.full_like(xx, fz)
#     ax.plot_surface(xx, yy, zz, alpha=0.28 * alpha, color='#999999', zorder=0)
#     n_gx, n_gy = 9, 6
#     for gx in np.linspace(cx - hw_x, cx + hw_x, n_gx):
#         ax.plot([gx, gx], [cy - hw_y, cy + hw_y], [fz, fz],
#                 color='#aaaaaa', alpha=0.30 * alpha, linewidth=0.5, zorder=1)
#     for gy in np.linspace(cy - hw_y, cy + hw_y, n_gy):
#         ax.plot([cx - hw_x, cx + hw_x], [gy, gy], [fz, fz],
#                 color='#aaaaaa', alpha=0.30 * alpha, linewidth=0.5, zorder=1)


# def _clean_ax(ax):
#     ax.set_axis_off()
#     ax.grid(False)
#     for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
#         pane.fill = False
#         pane.set_edgecolor('none')


# def _add_border(fig, color=BORDER_COLOR):
#     fig.add_artist(FancyBboxPatch(
#         (0.005, 0.005), 0.990, 0.990,
#         boxstyle='square,pad=0', linewidth=2.5,
#         edgecolor=color, facecolor='none',
#         transform=fig.transFigure, clip_on=False
#     ))


# def _fig_to_rgb(fig):
#     """Figure → (H,W,3) uint8. Uses buffer_rgba (matplotlib ≥ 3.8)."""
#     fig.canvas.draw()
#     buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
#     w, h = fig.canvas.get_width_height()
#     return buf.reshape(h, w, 4)[:, :, :3]


# # ── Keyframe extraction ────────────────────────────────────────────────────────
# def _extract_keyframes(joints, n_keys=7):
#     """
#     Motion-arc sampling: evenly spaced in cumulative-motion space.
#     Fast sections → more keyframes. Always includes first and last frame.
#     """
#     T = len(joints)
#     vel = np.zeros(T)
#     for t in range(1, T):
#         vel[t] = np.mean(np.linalg.norm(joints[t] - joints[t - 1], axis=1))
#     cum = np.cumsum(vel)
#     if cum[-1] == 0:
#         return list(np.linspace(0, T - 1, n_keys, dtype=int))
#     cum_n   = cum / cum[-1]
#     targets = np.linspace(0, 1, n_keys)
#     idxs    = sorted(set(
#         [0] + [int(np.argmin(np.abs(cum_n - a))) for a in targets] + [T - 1]
#     ))
#     while len(idxs) > n_keys:
#         gaps = [cum[idxs[i+1]] - cum[idxs[i]] for i in range(1, len(idxs)-1)]
#         idxs.pop(1 + int(np.argmin(gaps)))
#     return idxs


# # ==============================================================================
# # 1.  plot_3d_motion — animated GIF, skeleton upright, caption below
# # ==============================================================================
# def plot_3d_motion(
#     save_path,
#     kinematic_chain,
#     joints,               # (T, 22, 3) HumanML3D, Y=up
#     title='',
#     fps=20,
#     radius=1.5,           # kept for API compat
#     dist=2,               # kept for API compat
#     figsize=(7, 6),
#     elev=12,              # low elevation → skeleton fills frame vertically
#     azim=-80,             # face slightly from the side
#     bg_color='#f5f5f3',
# ):
#     T       = joints.shape[0]
#     caption = _wrap_caption(title)
#     n_lines = caption.count('\n') + 1 if caption else 0
#     # Reserve a small strip at bottom for caption
#     cap_frac = min(0.048 * n_lines, 0.16) if caption else 0.0

#     # ── Compute tight global bounds (swapped coords) ──────────────────────────
#     # plot-X = joints-X (width), plot-Y = joints-Z (depth), plot-Z = joints-Y (height)
#     px = joints[:, :, 0]   # width
#     py = joints[:, :, 2]   # depth
#     pz = joints[:, :, 1]   # height

#     cx   = (px.min() + px.max()) / 2
#     cy   = (py.min() + py.max()) / 2
#     fz   = pz.min()                          # foot level

#     # Tight span: skeleton should fill ~85% of axes
#     span = max(px.max()-px.min(), py.max()-py.min(), pz.max()-pz.min()) * 0.55

#     frames = []
#     for t in tqdm(range(T), desc=f"GIF {os.path.basename(save_path)}"):
#         fig = plt.figure(figsize=figsize, facecolor=bg_color, dpi=100)

#         bot = cap_frac + 0.005
#         ax  = fig.add_axes([0.0, bot, 1.0, 0.995 - bot], projection='3d')

#         ax.set_xlim(cx - span, cx + span)
#         ax.set_ylim(cy - span * 0.7, cy + span * 0.7)
#         ax.set_zlim(fz - 0.01, fz + span * 2.0)   # stand upright
#         ax.view_init(elev=elev, azim=azim)
#         _clean_ax(ax)
#         _add_floor(ax, cx, cy, span, span * 0.7, fz - 0.005)
#         _draw_bones(ax, joints[t], kinematic_chain, alpha=1.0, lw=2.8)

#         if caption:
#             fig.text(0.50, cap_frac * 0.38, caption,
#                      ha='center', va='center',
#                      fontsize=9, fontstyle='italic',
#                      fontfamily='DejaVu Serif', color='#222222',
#                      transform=fig.transFigure)

#         _add_border(fig)
#         frames.append(_fig_to_rgb(fig))
#         plt.close(fig)

#     imageio.mimsave(save_path, frames, fps=fps)
#     print(f"Saved GIF: {save_path}")


# # ==============================================================================
# # 2.  plot_skeleton_ghost — all key poses in ONE scene, spread along X
# # ==============================================================================
# def plot_skeleton_ghost(
#     save_path,
#     kinematic_chain,
#     joints,               # (T, 22, 3)
#     caption='',
#     n_keys=7,
#     figsize=(15, 5),
#     elev=14,              # low elevation shows floor well
#     azim=-70,             # slight angle to separate overlapping figures
#     bg_color='#f5f5f3',
#     border_color=BORDER_COLOR,
#     dpi=150,
#     spread=0.90,          # spacing between ghost figures (× scale)
# ):
#     """
#     All key poses in ONE single 3D scene, spread along X.
#     First/last frames solid, middle frames faded (sine alpha curve).
#     Clean floor underneath. Single caption below.
#     """
#     idxs   = _extract_keyframes(joints, n_keys=n_keys)
#     n      = len(idxs)

#     # Sine alpha: solid ends (1.0), faded middle (min ~0.20)
#     alphas = [float(1.0 - 0.80 * np.sin(i / max(n - 1, 1) * np.pi))
#               for i in range(n)]

#     clean_cap = _wrap_caption(caption)
#     cap_lines = clean_cap.count('\n') + 1 if clean_cap else 0
#     cap_h_in  = cap_lines * 0.22 + 0.20
#     total_h   = figsize[1] + cap_h_in
#     cap_frac  = cap_h_in / total_h
#     plot_frac = 1.0 - cap_frac

#     fig = plt.figure(figsize=(figsize[0], total_h), facecolor=bg_color)
#     ax  = fig.add_axes(
#         [0.0, cap_frac + 0.01, 1.0, plot_frac - 0.02],
#         projection='3d'
#     )

#     # ── Global bounds from all frames (swapped coords) ────────────────────────
#     px_all = joints[:, :, 0]   # width
#     py_all = joints[:, :, 2]   # depth
#     pz_all = joints[:, :, 1]   # height

#     # Centre of the original motion
#     cx0  = (px_all.min() + px_all.max()) / 2
#     cy0  = (py_all.min() + py_all.max()) / 2
#     fz0  = pz_all.min()   # foot level

#     # Scale based on body height (Z range)
#     scale = max(
#         px_all.max() - px_all.min(),
#         py_all.max() - py_all.min(),
#         pz_all.max() - pz_all.min()
#     ) * 0.55

#     total_spread = (n - 1) * spread * scale

#     # ── Draw each key pose ────────────────────────────────────────────────────
#     for i, (fidx, alpha) in enumerate(zip(idxs, alphas)):
#         jf     = joints[fidx]
#         x_off  = -total_spread / 2 + i * spread * scale - cx0
#         y_off  = -cy0
#         z_off  = -fz0   # shift feet to z=0

#         lw = 3.0 if alpha > 0.82 else 1.8
#         _draw_bones(ax, jf, kinematic_chain,
#                     alpha=alpha, lw=lw,
#                     x_off=x_off, y_off=y_off, z_off=z_off)

#     # ── Axis limits ───────────────────────────────────────────────────────────
#     pad = scale * 0.55
#     ax.set_xlim(-total_spread / 2 - pad, total_spread / 2 + pad)
#     ax.set_ylim(-scale * 0.65, scale * 0.65)
#     ax.set_zlim(-0.02, scale * 2.1)

#     ax.view_init(elev=elev, azim=azim)
#     _clean_ax(ax)

#     # ── Floor ─────────────────────────────────────────────────────────────────
#     hw   = total_spread / 2 + pad
#     hdep = scale * 0.65
#     _add_floor(ax, 0, 0, hw, hdep, -0.01)

#     # ── Caption ───────────────────────────────────────────────────────────────
#     if clean_cap:
#         fig.text(0.50, cap_frac * 0.38, clean_cap,
#                  ha='center', va='center',
#                  fontsize=11, fontstyle='italic',
#                  fontfamily='DejaVu Serif', color='#111111',
#                  transform=fig.transFigure)

#     # ── Border ────────────────────────────────────────────────────────────────
#     fig.add_artist(FancyBboxPatch(
#         (0.005, 0.005), 0.990, 0.990,
#         boxstyle='square,pad=0', linewidth=2.5,
#         edgecolor=border_color, facecolor='none',
#         transform=fig.transFigure, clip_on=False
#     ))

#     plt.savefig(save_path, dpi=dpi, bbox_inches='tight',
#                 facecolor=fig.get_facecolor())
#     plt.close(fig)
#     print(f"Saved ghost figure: {save_path}")
"""
utils/visualization.py
Final, complete implementation.

Key fixes vs previous versions:
  1. Floor follows the skeleton — computed from the SELECTED KEY FRAMES only,
     not the full sequence, so the floor always sits under all visible poses.
  2. Keyframe selection uses velocity + pose-change scoring to pick truly
     meaningful frames (action peaks) rather than uniform motion-arc spacing.
  3. GIF skeleton fills the frame — tight bounds per-frame with a floor that
     tracks the current root position.
  4. Single clean caption — first human sentence only, all metrics stripped.
  5. buffer_rgba used everywhere (matplotlib >= 3.8 compatible).
"""

import os
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401 (needed for 3d)
import numpy as np
import imageio
from tqdm import tqdm


# ── Colour scheme ──────────────────────────────────────────────────────────────
PART_COLORS = {
    'left_leg':  '#22a84e',
    'right_leg': '#1a8a3e',
    'spine':     '#3460b4',
    'left_arm':  '#c03030',
    'right_arm': '#a02020',
    'head':      '#3460b4',
}

CHAIN_PART_MAP = {
    (0,  2): 'left_leg',  (2,  5): 'left_leg',
    (5,  8): 'left_leg',  (8, 11): 'left_leg',
    (0,  1): 'right_leg', (1,  4): 'right_leg',
    (4,  7): 'right_leg', (7, 10): 'right_leg',
    (0,  3): 'spine',     (3,  6): 'spine',
    (6,  9): 'spine',     (9, 12): 'spine',
    (12,15): 'head',
    (9, 14): 'left_arm',  (14,17): 'left_arm',
    (17,19): 'left_arm',  (19,21): 'left_arm',
    (9, 13): 'right_arm', (13,16): 'right_arm',
    (16,18): 'right_arm', (18,20): 'right_arm',
}

BORDER_COLOR = '#4CAF50'


# ── Caption ────────────────────────────────────────────────────────────────────
def _wrap_caption(text, max_chars=80):
    """Return only the first clean human sentence, all metrics stripped."""
    if not text:
        return ''
    text = re.sub(r'^\d+;\s*', '', text.strip())
    if ';' in text:
        for p in text.split(';'):
            p = p.strip()
            if not any(k in p for k in
                       ['nDCG','spacy','spice','=','npy','.gif']):
                if len(p) > 8:
                    text = p
                    break
    text = re.sub(r'\(nDCG[^)]*\)', '', text)
    text = re.sub(r'spacy\s*=\s*[\d.]+', '', text)
    text = re.sub(r'spice\s*=\s*[\d.]+', '', text)
    text = re.sub(r'\(\s*\)', '', text)
    text = text.strip(' .,;)(')
    text = ' '.join(text.split())
    if text:
        text = text[0].upper() + text[1:]
    words = text.split()
    lines, line = [], []
    for w in words:
        if sum(len(x)+1 for x in line)+len(w) > max_chars:
            lines.append(' '.join(line)); line = [w]
        else:
            line.append(w)
    if line:
        lines.append(' '.join(line))
    return '\n'.join(lines)


# ── Coordinate helpers ─────────────────────────────────────────────────────────
# HumanML3D: joints[:,0]=X(width), joints[:,1]=Y(height UP), joints[:,2]=Z(depth)
# Matplotlib 3-D: 3rd positional arg = vertical axis
# → plot as  ax_x=joints_X,  ax_y=joints_Z,  ax_z=joints_Y

def _px(j): return j[:, 0]          # plot-X  = joints X
def _py(j): return j[:, 2]          # plot-Y  = joints Z (depth)
def _pz(j): return j[:, 1]          # plot-Z  = joints Y (height)


def _fig_to_rgb(fig):
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    return buf.reshape(h, w, 4)[:, :, :3]


def _clean_ax(ax):
    ax.set_axis_off(); ax.grid(False)
    for p in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        p.fill = False; p.set_edgecolor('none')


def _add_border(fig, color=BORDER_COLOR):
    fig.add_artist(FancyBboxPatch(
        (0.005,0.005), 0.990,0.990,
        boxstyle='square,pad=0', linewidth=2.5,
        edgecolor=color, facecolor='none',
        transform=fig.transFigure, clip_on=False))


# ── Floor ──────────────────────────────────────────────────────────────────────
def _draw_floor(ax, cx, cy, half_w, half_d, fz, alpha=1.0, n_gx=8, n_gy=6):
    """
    Draw a grey tiled floor.
    cx, cy  = centre in plot-X / plot-Y space
    half_w  = half-width  (plot-X direction)
    half_d  = half-depth  (plot-Y direction)
    fz      = floor height in plot-Z (= foot level)
    """
    xx = np.array([[cx-half_w, cx+half_w],[cx-half_w, cx+half_w]])
    yy = np.array([[cy-half_d, cy-half_d],[cy+half_d, cy+half_d]])
    zz = np.full_like(xx, fz)
    ax.plot_surface(xx, yy, zz, alpha=0.28*alpha, color='#999999', zorder=0)
    for gx in np.linspace(cx-half_w, cx+half_w, n_gx):
        ax.plot([gx,gx],[cy-half_d,cy+half_d],[fz,fz],
                color='#aaaaaa', alpha=0.30*alpha, lw=0.5, zorder=1)
    for gy in np.linspace(cy-half_d, cy+half_d, n_gy):
        ax.plot([cx-half_w,cx+half_w],[gy,gy],[fz,fz],
                color='#aaaaaa', alpha=0.30*alpha, lw=0.5, zorder=1)


# ── Skeleton drawing ───────────────────────────────────────────────────────────
def _draw_pose(ax, joints, kinematic_chain,
               alpha=1.0, lw=2.5, dx=0.0, dy=0.0, dz=0.0):
    """
    Draw one pose with optional (dx,dy,dz) offset in PLOT space.
    dx = offset in plot-X, dy = offset in plot-Y, dz = offset in plot-Z
    """
    for bone in kinematic_chain:
        i, j = bone
        color = PART_COLORS.get(CHAIN_PART_MAP.get(bone,'spine'), '#3460b4')
        ax.plot(
            [_px(joints)[i]+dx, _px(joints)[j]+dx],
            [_py(joints)[i]+dy, _py(joints)[j]+dy],
            [_pz(joints)[i]+dz, _pz(joints)[j]+dz],
            color=color, alpha=alpha, lw=lw,
            solid_capstyle='round', zorder=3)
    ax.scatter(
        _px(joints)+dx, _py(joints)+dy, _pz(joints)+dz,
        c='white', s=10, zorder=4,
        alpha=min(alpha*0.9, 1.0),
        edgecolors='#333333', linewidths=0.4)


# ── Keyframe extraction ────────────────────────────────────────────────────────
def _extract_keyframes(joints, n_keys=6):
    """
    Pick n_keys frames that best represent the motion.

    Strategy:
      1. Compute per-frame velocity  (mean joint displacement).
      2. Compute per-frame pose-change score  (sum of joint angle changes
         approximated as L2 distance between consecutive normalised poses).
      3. Combine: score = velocity * 0.6 + pose_change * 0.4
      4. Sample evenly along the cumulative score arc so that high-action
         regions get more keyframes.
      5. Always include frame 0 and frame T-1.
    """
    T = len(joints)
    if T <= n_keys:
        return list(range(T))

    # Velocity
    vel = np.zeros(T)
    for t in range(1, T):
        vel[t] = np.mean(np.linalg.norm(joints[t] - joints[t-1], axis=1))

    # Pose-change (normalise each frame by body scale then L2 diff)
    scale  = np.mean(np.linalg.norm(joints - joints[:,0:1,:], axis=2)) + 1e-8
    normed = (joints - joints[:,0:1,:]) / scale
    pose_d = np.zeros(T)
    for t in range(1, T):
        pose_d[t] = np.linalg.norm(normed[t] - normed[t-1])

    score = vel * 0.6 + pose_d * 0.4
    cum   = np.cumsum(score)
    if cum[-1] == 0:
        return list(np.linspace(0, T-1, n_keys, dtype=int))

    cum_n   = cum / cum[-1]
    targets = np.linspace(0, 1, n_keys)
    idxs    = sorted(set(
        [0] +
        [int(np.argmin(np.abs(cum_n - a))) for a in targets] +
        [T-1]
    ))
    # Trim back to n_keys if deduplication gave too many
    while len(idxs) > n_keys:
        gaps = [cum[idxs[i+1]]-cum[idxs[i]] for i in range(1, len(idxs)-1)]
        idxs.pop(1 + int(np.argmin(gaps)))

    return idxs


# ==============================================================================
# 1.  plot_3d_motion  — animated GIF
# ==============================================================================
def plot_3d_motion(
    save_path,
    kinematic_chain,
    joints,               # (T, 22, 3)
    title='',
    fps=20,
    radius=1.5,           # API compat
    dist=2,               # API compat
    figsize=(7, 6),
    elev=12,
    azim=-80,
    bg_color='#f5f5f3',
):
    T       = joints.shape[0]
    caption = _wrap_caption(title)
    n_lines = caption.count('\n')+1 if caption else 0
    cap_frac = min(0.048*n_lines, 0.16) if caption else 0.0

    # Global body scale (used for floor size and axis limits)
    body_h  = _pz(joints.reshape(-1,3)).max() - _pz(joints.reshape(-1,3)).min()
    half_fl = body_h * 0.8   # floor half-width/depth

    frames = []
    for t in tqdm(range(T), desc=f"GIF {os.path.basename(save_path)}"):
        fig = plt.figure(figsize=figsize, facecolor=bg_color, dpi=100)
        bot = cap_frac + 0.005
        ax  = fig.add_axes([0.0, bot, 1.0, 0.995-bot], projection='3d')

        j = joints[t]

        # Root position in plot space — floor follows the skeleton
        root_x = _px(j)[0]
        root_y = _py(j)[0]
        fz     = _pz(j).min() - 0.01   # foot level this frame

        # Tight axis limits around current pose
        span = body_h * 0.62
        ax.set_xlim(root_x - span*1.4, root_x + span*1.4)
        ax.set_ylim(root_y - span*0.8,  root_y + span*0.8)
        ax.set_zlim(fz - 0.01, fz + body_h*1.25)

        ax.view_init(elev=elev, azim=azim)
        _clean_ax(ax)

        # Floor centred on root — moves with the skeleton
        _draw_floor(ax, root_x, root_y, half_fl, half_fl*0.65, fz)

        _draw_pose(ax, j, kinematic_chain, alpha=1.0, lw=2.8)

        if caption:
            fig.text(0.50, cap_frac*0.38, caption,
                     ha='center', va='center',
                     fontsize=9, fontstyle='italic',
                     fontfamily='DejaVu Serif', color='#222222',
                     transform=fig.transFigure)

        _add_border(fig)
        frames.append(_fig_to_rgb(fig))
        plt.close(fig)

    imageio.mimsave(save_path, frames, fps=fps)
    print(f"Saved GIF: {save_path}")


# ==============================================================================
# 2.  plot_skeleton_ghost  — key-pose ghost strip, single scene
# ==============================================================================
def plot_skeleton_ghost(
    save_path,
    kinematic_chain,
    joints,               # (T, 22, 3)
    caption='',
    n_keys=6,
    figsize=(14, 5),
    elev=14,
    azim=-75,
    bg_color='#f5f5f3',
    border_color=BORDER_COLOR,
    dpi=150,
    spread=0.85,
):
    """
    All key poses in ONE single 3D scene, spread along plot-X.
    Floor is computed from the bounding box of the KEY FRAMES only
    so it always covers all visible figures.
    First/last frames solid, middle faded (sine alpha curve).
    """
    idxs   = _extract_keyframes(joints, n_keys=n_keys)
    n      = len(idxs)
    alphas = [float(1.0 - 0.80*np.sin(i/max(n-1,1)*np.pi)) for i in range(n)]

    # ── Caption layout ────────────────────────────────────────────────────────
    clean_cap = _wrap_caption(caption)
    cap_lines = clean_cap.count('\n')+1 if clean_cap else 0
    cap_h_in  = cap_lines*0.22 + 0.22
    total_h   = figsize[1] + cap_h_in
    cap_frac  = cap_h_in / total_h

    fig = plt.figure(figsize=(figsize[0], total_h), facecolor=bg_color)
    ax  = fig.add_axes(
        [0.0, cap_frac+0.01, 1.0, (1.0-cap_frac)-0.02],
        projection='3d')

    # ── Compute per-pose offsets ──────────────────────────────────────────────
    # Each pose is centred on its own root joint, then spread along plot-X.
    # This means the floor only needs to cover the spread range, not the
    # full trajectory — so figures never walk off the edge.

    body_h = (_pz(joints.reshape(-1,3)).max() -
               _pz(joints.reshape(-1,3)).min())
    scale  = body_h * 0.62        # characteristic size for spacing

    total_spread_x = (n-1) * spread * scale
    offsets_x = [-total_spread_x/2 + i*spread*scale for i in range(n)]

    # Collect all posed points in plot space to compute tight bounds
    all_px, all_py, all_pz = [], [], []
    for i, fidx in enumerate(idxs):
        j = joints[fidx]
        # Centre each pose on its root in X and Y; shift Z so feet = 0
        root_x = _px(j)[0]; root_y = _py(j)[0]; foot_z = _pz(j).min()
        dx =  offsets_x[i] - root_x
        dy =               - root_y
        dz =               - foot_z
        all_px.extend((_px(j)+dx).tolist())
        all_py.extend((_py(j)+dy).tolist())
        all_pz.extend((_pz(j)+dz).tolist())

    all_px = np.array(all_px)
    all_py = np.array(all_py)
    all_pz = np.array(all_pz)

    cx_scene = (all_px.min()+all_px.max())/2
    cy_scene = (all_py.min()+all_py.max())/2
    hw_x     = (all_px.max()-all_px.min())/2 + scale*0.55
    hw_y     = max((all_py.max()-all_py.min())/2, scale*0.55) + scale*0.15

    # ── Draw floor FIRST (covers ALL visible figures) ─────────────────────────
    _draw_floor(ax, cx_scene, cy_scene, hw_x, hw_y, fz=-0.01)

    # ── Draw each key pose ────────────────────────────────────────────────────
    for i, (fidx, alpha) in enumerate(zip(idxs, alphas)):
        j      = joints[fidx]
        root_x = _px(j)[0]; root_y = _py(j)[0]; foot_z = _pz(j).min()
        dx = offsets_x[i] - root_x
        dy =              - root_y
        dz =              - foot_z
        lw = 3.0 if alpha > 0.82 else 1.8
        _draw_pose(ax, j, kinematic_chain,
                   alpha=alpha, lw=lw, dx=dx, dy=dy, dz=dz)

    # ── Axis limits ───────────────────────────────────────────────────────────
    ax.set_xlim(cx_scene-hw_x, cx_scene+hw_x)
    ax.set_ylim(cy_scene-hw_y, cy_scene+hw_y)
    ax.set_zlim(-0.02, body_h*1.20)

    ax.view_init(elev=elev, azim=azim)
    _clean_ax(ax)

    # ── Caption ───────────────────────────────────────────────────────────────
    if clean_cap:
        fig.text(0.50, cap_frac*0.38, clean_cap,
                 ha='center', va='center',
                 fontsize=11, fontstyle='italic',
                 fontfamily='DejaVu Serif', color='#111111',
                 transform=fig.transFigure)

    fig.add_artist(FancyBboxPatch(
        (0.005,0.005), 0.990,0.990,
        boxstyle='square,pad=0', linewidth=2.5,
        edgecolor=border_color, facecolor='none',
        transform=fig.transFigure, clip_on=False))

    plt.savefig(save_path, dpi=dpi, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved ghost figure: {save_path}")
