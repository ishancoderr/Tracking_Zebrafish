import cv2
import os

class CropSaveVideo:
    def __init__(self, output_path):
        self.output_path = output_path
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    async def crop_and_save_video(self, video_path):
        print("Processing video for cropping...")
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}.")
            return

        ret, first_frame = cap.read()
        if not ret:
            print("Error: Could not read the first frame of the video.")
            cap.release()
            return

        scale_percent = 50  
        resized_frame = cv2.resize(
            first_frame, 
            (int(first_frame.shape[1] * scale_percent / 100), 
             int(first_frame.shape[0] * scale_percent / 100))
        )

        print("Select the region for the bounding box and press ENTER.")
        bounding_box = cv2.selectROI("Select Bounding Box", resized_frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select Bounding Box")
        print('Bounding box:', bounding_box)

        x, y, w, h = [int(coord / (scale_percent / 100)) for coord in bounding_box]

        cropped_video_path = os.path.join(self.output_path, f"cropped_{os.path.basename(video_path)}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        cropped_width = w
        cropped_height = h
        out_cropped = cv2.VideoWriter(cropped_video_path, fourcc, fps, (cropped_width, cropped_height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            cropped_frame = frame[y:y+h, x:x+w]
            out_cropped.write(cropped_frame)

        cap.release()
        out_cropped.release()

        print(f"Cropped video saved at: {cropped_video_path}")
        return cropped_video_path