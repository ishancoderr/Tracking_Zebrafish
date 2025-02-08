import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class VisualizeData:
    def __init__(self):
        pass

    async def generate_heatmap(self, video_path, tracked_positions, output_path):
        print('run here c1')
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        print(ret)
        if not ret:
            print("Failed to read video")
            return

        heatmap = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)

        for fish_id, positions in tracked_positions.items():
            for x, y in positions:
                heatmap[y, x] += 1  

        cap.release()

        heatmap = cv2.GaussianBlur(heatmap, (25, 25), 0)  # Smooth heatmap
        heatmap = (heatmap / heatmap.max()) * 255  
        heatmap = np.uint8(heatmap)

        plt.figure(figsize=(10, 6))
        plt.imshow(heatmap, cmap='jet', interpolation='nearest')
        plt.colorbar(label="Fish Presence Intensity")
        plt.title("Fish Movement Heatmap")
        plt.axis("off")

        heatmap_image_path = output_path + "/heatmap.png"
        plt.savefig(heatmap_image_path)
        plt.close()

        print(f"Heatmap saved at: {heatmap_image_path}")
        return heatmap_image_path
    
    async def plot_3d_trajectories(self, tracked_positions, output_path):
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')

        for fish_id, positions in tracked_positions.items():
            if len(positions) > 1:
                xs, ys, zs = zip(*[(x, y, t) for t, (x, y) in enumerate(positions)])
                ax.plot(xs, ys, zs, label=f'Fish {fish_id}')

        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_zlabel("Time (Frames)")
        ax.set_title("3D Fish Trajectories")
        ax.legend()

        trajectory_image_path = output_path + "/3d_trajectory.png"
        plt.savefig(trajectory_image_path)
        plt.close()

        print(f"3D trajectory plot saved at: {trajectory_image_path}")
        return trajectory_image_path