import numpy as np

file_path = r"C:\text-to-motion-retrieval_Exp\dataset\KIT-ML\new_joints\00001.npy"

data = np.load(file_path)

print("Type:", type(data))
print("Shape:", data.shape)
print("Dtype:", data.dtype)

# First few entries
print(data[:5])
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Load motion
file_path = r"C:\text-to-motion-retrieval_Exp\dataset\KIT-ML\new_joints\00001.npy"

motion = np.load(file_path)

print("Shape:", motion.shape)

# 21 joints connections (KIT-ML common skeleton approximation)
edges = [
    (0,1),(1,2),(2,3),(3,4),        # right leg
    (0,5),(5,6),(6,7),              # left leg
    (0,8),(8,9),(9,10),             # spine to head
    (8,11),(11,12),(12,13),         # left arm
    (8,14),(14,15),(15,16)          # right arm
]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def update(frame):
    ax.clear()
    joints = motion[frame]

    x = joints[:, 0]
    y = joints[:, 1]
    z = joints[:, 2]

    ax.scatter(x, y, z, c='red')

    # draw bones
    for i, j in edges:
        ax.plot(
            [x[i], x[j]],
            [y[i], y[j]],
            [z[i], z[j]],
            c='blue'
        )

    ax.set_xlim([-200, 200])
    ax.set_ylim([300, 900])
    ax.set_zlim([-200, 200])
    ax.set_title(f"Frame {frame}")

ani = FuncAnimation(fig, update, frames=len(motion), interval=80)

plt.show()