import os
import cv2
import math
import numpy as np
from fastapi import HTTPException
from api.baseManager import BaseMamager
from scipy.spatial.distance import mahalanobis
from scipy.optimize import linear_sum_assignment
from api.preprocessing.cropAndSaveVideo import CropSaveVideo
from api.preprocessing.visualizeData import VisualizeData
from filterpy.kalman import KalmanFilter

class KalmanTracker:
    def __init__(self, initial_position=None):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                [0, 1, 0, 1],
                                                [0, 0, 1, 0],
                                                [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1  
        self.kalman.errorCovPost = np.eye(4, dtype=np.float32) * 1  

        # Initialize state with the first detected position
        if initial_position is not None:
            self.kalman.statePre = np.array([[initial_position[0]], [initial_position[1]], [0], [0]], dtype=np.float32)
            self.kalman.statePost = np.array([[initial_position[0]], [initial_position[1]], [0], [0]], dtype=np.float32)
        else:
            self.kalman.statePre = np.zeros((4, 1), dtype=np.float32)
            self.kalman.statePost = np.zeros((4, 1), dtype=np.float32)

        self.predicted = None

    def predict(self):
        self.predicted = self.kalman.predict()
        return self.predicted[:2].flatten()

    def update(self, centroid):
        measurement = np.array([[np.float32(centroid[0])], [np.float32(centroid[1])]])
        self.kalman.correct(measurement)

class PreprocessingSteps(BaseMamager):
    def __init__(self, directory_path: str, output_path: str):
        self.directory_path = directory_path
        self.output_path = output_path
        self.previous_boxes = []
        self.next_fish_id = 1
        self.crop_save_instance = CropSaveVideo(output_path)
        self.visualizeData = VisualizeData()

    async def getVideoPaths(self):
        if not os.path.isdir(self.directory_path):
            raise HTTPException(status_code=400, detail="Invalid directory path")

        video_paths = [
            os.path.join(self.directory_path, f)
            for f in os.listdir(self.directory_path)
            if f.endswith(('.mp4', '.avi', '.mov'))
        ]

        if not video_paths:
            raise HTTPException(status_code=404, detail="No video files found")

        return video_paths

    async def addFilters(self, video_paths, filter_name: str):
        return video_paths   
    
    async def getVideoFrames(self, video_path):
        print(f"Original video path: {video_path}")
        # Call the method from the cropSaveVideo instance
        cropped_video_path = await self.crop_save_instance.crop_and_save_video(video_path)
        print(f"Cropped video path: {cropped_video_path}")
        cap = cv2.VideoCapture(cropped_video_path)
        backSub = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=300, detectShadows=True)
        kernel_size = (3, 3)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)

        output_filename = os.path.join(self.output_path, f"processed_{os.path.basename(cropped_video_path)}")
        bg_subtracted_filename = os.path.join(self.output_path, f"bg_subtracted_{os.path.basename(cropped_video_path)}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))
        out_bg_subtracted = cv2.VideoWriter(bg_subtracted_filename, fourcc, fps, (frame_width, frame_height), isColor=False)
        
        trackers = [KalmanTracker() for _ in range(5)]  
        fish_ids = [i + 1 for i in range(5)] 
        assignments = {}

        all_tracked_positions = {fish_id: [] for fish_id in fish_ids}

        tank_x_min, tank_x_max = 50, frame_width - 50  
        tank_y_min, tank_y_max = 50, frame_height - 50

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            fgMask = backSub.apply(frame)
            fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel, iterations=4)

            out_bg_subtracted.write(fgMask)

            contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5] 

            centroids = []
            for contour in contours:
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    if tank_x_min <= cx <= tank_x_max and tank_y_min <= cy <= tank_y_max:
                        centroids.append((cx, cy))
                        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

            predicted_positions = [tracker.predict() for tracker in trackers]

            # Hungarian Algorithm for data association
            if centroids:
                cost_matrix = np.zeros((len(predicted_positions), len(centroids)))
                for i, predicted in enumerate(predicted_positions):
                    for j, centroid in enumerate(centroids):
                        cost_matrix[i, j] = np.linalg.norm(predicted - np.array(centroid))

                row_ind, col_ind = linear_sum_assignment(cost_matrix)

                for row, col in zip(row_ind, col_ind):
                    assignments[row] = centroids[col]
                    trackers[row].update(centroids[col])

                for fish_id, tracker in zip(fish_ids, trackers):
                    prediction = tracker.predicted if tracker.predicted is not None else (0, 0)

                    # Only update tracked positions if the prediction is within the tank bounds
                    if tank_x_min <= prediction[0] <= tank_x_max and tank_y_min <= prediction[1] <= tank_y_max:
                        all_tracked_positions[fish_id].append((int(prediction[0]), int(prediction[1])))

                        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]  
                        color = colors[fish_id % len(colors)]  

                        if len(all_tracked_positions[fish_id]) > 1:
                            for i in range(1, len(all_tracked_positions[fish_id])):
                                cv2.line(frame, all_tracked_positions[fish_id][i - 1], all_tracked_positions[fish_id][i], color, 2)  

                        cv2.circle(frame, (int(prediction[0]), int(prediction[1])), 15, color, -1) 

            out.write(frame)

            cv2.imshow("Tracking", frame)
            if cv2.waitKey(30) & 0xFF == 27:
                break

        cap.release()
        out.release()
        out_bg_subtracted.release()
        cv2.destroyAllWindows()
        print(cropped_video_path, all_tracked_positions, self.output_path)

        #  heatmap generation
        heatmap_path = await self.visualizeData.generate_heatmap(cropped_video_path, all_tracked_positions, self.output_path)
        print(f"Heatmap generated at: {heatmap_path}")

        #  3D Trajectory Visualization
        trajectory_path = await self.visualizeData.plot_3d_trajectories(all_tracked_positions, self.output_path)
        print(f"Processed video saved at: {output_filename}")
        return {
            "processed_video": output_filename,
            "bg_subtracted_video": bg_subtracted_filename,
            "heatmap": heatmap_path,
            "trajectory_plot": trajectory_path
        }
