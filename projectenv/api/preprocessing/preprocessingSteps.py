import os
import cv2
import math
import numpy as np
from fastapi import HTTPException
from api.baseManager import BaseMamager
from scipy.spatial.distance import mahalanobis
from scipy.optimize import linear_sum_assignment
from api.preprocessing.cropAndSaveVideo import CropSaveVideo
from filterpy.kalman import KalmanFilter

class KalmanTracker:
    def __init__(self):
        # creates a Kalman filter with 4 state variables (representing position and velocity in 2D) and 2 measurement variables (the observed positions).
        self.kalman = cv2.KalmanFilter(4, 2)
        # This matrix converts the state to the measurement space. Here, it's set to extract only the position (x and y coordinates) from the state vector
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                  [0, 1, 0, 0]], np.float32)
        #This matrix defines how the state evolves from one time step to the next, considering the system's dynamics (e.g., constant velocity model). It models the movement, predicting the next state based on the current state. The matrix assumes that the object's position changes based on its velocity
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                 [0, 1, 0, 1],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kalman.statePre = np.random.rand(4, 1).astype(np.float32)
        self.kalman.statePost = np.random.rand(4, 1).astype(np.float32)
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
        backSub = cv2.createBackgroundSubtractorKNN(detectShadows=True)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

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

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            fgMask = backSub.apply(frame)
            fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel, iterations=4)

            contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]  # Top 5 contours (fish)

            centroids = []
            for contour in contours:
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    centroids.append((cx, cy))
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

            # Get predicted positions from Kalman filters
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

                fish_paths = {fish_id: [] for fish_id in fish_ids}  

                for fish_id, tracker in zip(fish_ids, trackers):
                    prediction = tracker.predicted if tracker.predicted is not None else (0, 0)

                    if tracker.predicted is not None:
                        fish_paths[fish_id].append((int(prediction[0]), int(prediction[1])))

                    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]  
                    color = colors[fish_id % len(colors)]  

                    if len(fish_paths[fish_id]) > 1:
                        for i in range(1, len(fish_paths[fish_id])):
                            cv2.line(frame, fish_paths[fish_id][i - 1], fish_paths[fish_id][i], color, 2)  

                    cv2.circle(frame, (int(prediction[0]), int(prediction[1])), 15, color, -1) 

            out.write(frame)

            cv2.imshow("Tracking", frame)
            if cv2.waitKey(30) & 0xFF == 27:
                break

        cap.release()
        out.release()
        out_bg_subtracted.release()
        cv2.destroyAllWindows()
        print(f"Processed video saved at: {output_filename}")
        return {
            "processed_video": output_filename,
            "bg_subtracted_video": bg_subtracted_filename
        }


