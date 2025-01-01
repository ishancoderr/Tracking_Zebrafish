import os
import cv2
import math
import numpy as np
from fastapi import HTTPException
from api.baseManager import BaseMamager
from scipy.spatial.distance import mahalanobis
from scipy.optimize import linear_sum_assignment

class Preprocessing(BaseMamager):
    def __init__(self, directory_path: str, output_path: str):
        self.directory_path = directory_path
        self.output_path = output_path
        self.previous_boxes = []
        self.next_fish_id = 1
        self.lost_fish = {}

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

    async def backgroundSubtraction(self, video_path):
        cap = cv2.VideoCapture(video_path)
        backSub = cv2.createBackgroundSubtractorKNN(detectShadows=True)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))

        output_filename = os.path.join(self.output_path, f"processed_{os.path.basename(video_path)}")
        bg_subtracted_filename = os.path.join(self.output_path, f"bg_subtracted_{os.path.basename(video_path)}")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))
        out_bg_subtracted = cv2.VideoWriter(bg_subtracted_filename, fourcc, fps, (frame_width, frame_height), isColor=False)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            fgMask = backSub.apply(frame)
            fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel, iterations=4)

            contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)

            centroids = []

            for contour in contours[:5]:  
                area = cv2.contourArea(contour)
                if area > 140:
                    x, y, w, h = cv2.boundingRect(contour)
                    centroid_x = x + w // 2
                    centroid_y = y + h // 2

                    roi_frame = frame[y:y + h, x:x + w]
                    roi_gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
                    brightness = np.mean(roi_gray)
                    contrast = np.std(roi_gray)

                    centroid_details = {
                        'centroid_x': centroid_x,
                        'centroid_y': centroid_y,
                        'area': area,
                        'brightness': brightness,
                        'contrast': contrast,
                        'width': w,
                        'height': h
                    }

                    centroids.append(centroid_details)

                    cv2.circle(frame, (centroid_x, centroid_y), 5, (0, 255, 0), -1)
                    cv2.putText(frame, f"B: {brightness:.2f}", (centroid_x, centroid_y + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.putText(frame, f"C: {contrast:.2f}", (centroid_x, centroid_y + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            self.previous_boxes = []
            self.next_fish_id = 1
            centroids = await self.assign_fish_numbers(self.previous_boxes, centroids)

            for i in range(min(5, len(centroids))):
                centroid = centroids[i]
                centroid_x, centroid_y = centroid['centroid_x'], centroid['centroid_y']
                fish_id = centroid['fish_id']
                cv2.putText(frame, fish_id, (centroid_x, centroid_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            out.write(frame)
            out_bg_subtracted.write(fgMask)

        cap.release()
        out.release()
        out_bg_subtracted.release()

        return {
            "processed_video": output_filename,
            "bg_subtracted_video": bg_subtracted_filename
        }
    '''
    async def match_bounding_boxes(self, bounding_boxes):
        if len(bounding_boxes) <= 1:
            return bounding_boxes

        features = np.array([
            [box['area'], box['contrast'], box['brightness']]
            for box in bounding_boxes
        ])

        filtered_boxes = [box for i, box in enumerate(bounding_boxes) if features[i][0] >= 600]
        if not filtered_boxes:
            return []

        filtered_features = np.array([
            [ box['contrast'], box['brightness']]
            for box in filtered_boxes
        ])

        filtered_centroids = np.array([
        [(box['x'] + box['w'] / 2), (box['y'] + box['h'] / 2)]
        for box in filtered_boxes
        ])

        z_features = (filtered_features - filtered_features.mean(axis=0)) / filtered_features.std(axis=0)

        inv_cov_matrix = np.linalg.inv(np.cov(z_features.T))
        mean_vector = z_features.mean(axis=0)

        distances = [
            mahalanobis(row, mean_vector, inv_cov_matrix)
            for row in z_features
        ]

        for i, box in enumerate(filtered_boxes):
            box['score'] = -distances[i]  

        final_boxes = [box for box in filtered_boxes if box['contrast'] > 25]

        final_boxes.sort(key=lambda x: x['score'], reverse=True)

        return final_boxes
    '''
    async def assign_fish_numbers(self, previous_centroids, current_centroids):
        if not previous_centroids:
            for centroid in current_centroids:
                centroid['fish_id'] = f"F{self.next_fish_id}"
                self.next_fish_id += 1
            self.previous_boxes = current_centroids
            return current_centroids

        distance_threshold = 120
        brightness_threshold = 15
        contrast_threshold = 10

        cost_matrix = await self.calculate_cost_matrix(previous_centroids, current_centroids, brightness_threshold, contrast_threshold)

        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        unmatched_previous, unmatched_current = set(range(len(previous_centroids))), set(range(len(current_centroids)))
        assigned_ids = set()

        for row, col in zip(row_indices, col_indices):
            if cost_matrix[row, col] < distance_threshold:
                current_centroids[col]['fish_id'] = previous_centroids[row]['fish_id']
                assigned_ids.add(previous_centroids[row]['fish_id'])
                unmatched_previous.discard(row)
                unmatched_current.discard(col)
            else:
                self.lost_fish[previous_centroids[row]['fish_id']] = previous_centroids[row]

        await self.handle_unmatched_current(unmatched_current, current_centroids, assigned_ids)

        await self.remove_lost_fish(assigned_ids)

        self.previous_boxes = current_centroids
        return current_centroids

    async def calculate_cost_matrix(self, previous_centroids, current_centroids, brightness_threshold, contrast_threshold):
        cost_matrix = np.zeros((len(previous_centroids), len(current_centroids)))

        for i, prev in enumerate(previous_centroids):
            prev_point = np.array([prev['x'], prev['y']])
            prev_brightness, prev_contrast = prev['brightness'], prev['contrast']

            for j, curr in enumerate(current_centroids):
                curr_point = np.array([curr['x'], curr['y']])
                curr_brightness, curr_contrast = curr['brightness'], curr['contrast']

                distance = np.linalg.norm(prev_point - curr_point)
                brightness_diff = abs(prev_brightness - curr_brightness)
                contrast_diff = abs(prev_contrast - curr_contrast)

                if brightness_diff > brightness_threshold and contrast_diff > contrast_threshold:
                    cost_matrix[i, j] = float('inf')
                else:
                    cost_matrix[i, j] = (distance * 0.6) + (brightness_diff * 0.2) + (contrast_diff * 0.2)

        return cost_matrix

    async def handle_unmatched_current(self, unmatched_current, current_centroids, assigned_ids):
        for col in unmatched_current:
            centroid = current_centroids[col]
            closest_lost_id, min_cost = await self.find_closest_lost_fish(centroid)

            if closest_lost_id:
                centroid['fish_id'] = closest_lost_id
                del self.lost_fish[closest_lost_id]
            else:
                centroid['fish_id'] = f"F{self.next_fish_id}"
                self.next_fish_id += 1

    async def find_closest_lost_fish(self, centroid):
        closest_lost_id, min_cost = None, float('inf')

        for lost_id, lost_centroid in self.lost_fish.items():
            distance, brightness_diff, contrast_diff = await self.calculate_centroid_diff(centroid, lost_centroid)
            cost = (distance * 0.6) + (brightness_diff * 0.2) + (contrast_diff * 0.2)
            if cost < min_cost:
                closest_lost_id = lost_id
                min_cost = cost

        return closest_lost_id, min_cost

    async def calculate_centroid_diff(self, centroid, lost_centroid):
        lost_point = np.array([lost_centroid['x'], lost_centroid['y']])
        lost_brightness, lost_contrast = lost_centroid['brightness'], lost_centroid['contrast']
        curr_point = np.array([centroid['x'], centroid['y']])
        curr_brightness, curr_contrast = centroid['brightness'], centroid['contrast']

        distance = np.linalg.norm(lost_point - curr_point)
        brightness_diff = abs(lost_brightness - curr_brightness)
        contrast_diff = abs(lost_contrast - curr_contrast)

        return distance, brightness_diff, contrast_diff

    async def remove_lost_fish(self, assigned_ids):
        for lost_id in list(self.lost_fish.keys()):
            if lost_id not in assigned_ids:
                del self.lost_fish[lost_id]


