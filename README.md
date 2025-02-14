# Tracking_Zebrafish

## Animal Welfare - Tracking Zebrafish in Laboratory Conditions

This project provides a robust API for tracking zebrafish in laboratory conditions using advanced video processing techniques. It is designed to assist researchers in monitoring zebrafish behavior, ensuring animal welfare, and analyzing movement patterns. The API offers two versions of tracking:

### Version 1: Basic Tracking
Basic tracking without a Kalman filter, suitable for detecting shadows and simpler tracking scenarios.

### Version 2: Advanced Tracking
Advanced tracking with a Kalman filter, heatmap generation, and 3D trajectory visualization for more accurate and detailed analysis.

---

## Table of Contents
1. [Overview](#overview)
2. [Getting Started](#getting-started)
   - [Prerequisites](#prerequisites)
   - [Installation](#installation)
3. [API Endpoints](#api-endpoints)
   - [Version 1: Basic Tracking](#version-1-basic-tracking)
   - [Version 2: Advanced Tracking](#version-2-advanced-tracking)
4. [How It Works](#how-it-works)
   - [Version 1: Workflow](#version-1-workflow)
   - [Version 2: Workflow](#version-2-workflow)
5. [Examples](#examples)
   - [Example Requests](#example-requests)
   - [Example Responses](#example-responses)
6. [Troubleshooting](#troubleshooting)
7. [Contributing](#contributing)
8. [License](#license)

---

## Overview

The **Tracking Zebrafish API** processes video files to detect and track zebrafish in laboratory environments. It uses computer vision techniques such as background subtraction, contour detection, and Kalman filtering to achieve accurate tracking. The API provides two versions:

- **Version 1**: A lightweight solution for basic tracking and shadow detection.
- **Version 2**: A more advanced solution with additional features like heatmaps and 3D trajectory plots.

---

## Getting Started

### Prerequisites
Before using the API, ensure you have the following installed:
- Python 3.8 or higher
- OpenCV (`opencv-python`)
- NumPy (`numpy`)
- FastAPI (`fastapi`)
- Uvicorn (`uvicorn`) for running the server
- Other dependencies listed in `requirements.txt`

### Installation
```bash
# Clone the repository
git clone https://github.com/your-repo/Tracking_Zebrafish.git
cd Tracking_Zebrafish

# Install the required dependencies
pip install -r requirements.txt

# Start the FastAPI server
uvicorn main:app --reload
```
The API will be available at [http://localhost:8000](http://localhost:8000).

---

## API Endpoints

### Version 1: Basic Tracking
- **Endpoint**: `/get_detectFish_v1`
- **Method**: `POST`
- **Description**: Detects and tracks fish in videos using background subtraction and contour detection. This version is lightweight and suitable for detecting shadows.

#### Input:
```json
{
  "directory_path": "path/to/input/videos",
  "output_path": "path/to/save/results"
}
```
#### Output:
```json
{
  "processed_videos": ["path/to/processed_video1.mp4"],
  "background_sub_videos": ["path/to/bg_subtracted_video1.mp4"]
}
```

### Version 2: Advanced Tracking
- **Endpoint**: `/get_detectFish_v2`
- **Method**: `POST`
- **Description**: Detects and tracks fish using a Kalman filter for improved accuracy. It also generates heatmaps and 3D trajectory plots.

#### Input:
```json
{
  "directory_path": "path/to/input/videos",
  "output_path": "path/to/save/results"
}
```
#### Output:
```json
{
  "processed_videos": ["path/to/processed_video1.mp4"],
  "background_sub_videos": ["path/to/bg_subtracted_video1.mp4"],
  "heatmaps": ["path/to/heatmap.png"],
  "trajectory_plots": ["path/to/3d_trajectory.png"]
}
```

---

## How It Works

### Version 1: Workflow
1. **Background Subtraction**
   - Uses OpenCV's KNN background subtractor to isolate moving objects (fish) from the background.
   - Applies morphological operations to clean up the foreground mask.
2. **Contour Detection**
   - Detects contours in the foreground mask and filters them based on area.
   - Extracts features like centroid, brightness, and contrast for each detected fish.
3. **Fish Tracking**
   - Assigns unique IDs to fish based on their centroids and features.
   - Tracks fish across frames using a simple matching algorithm.

### Version 2: Workflow
1. **Background Subtraction** (Similar to Version 1 but with additional filtering and smoothing.)
2. **Kalman Filter** (Uses a Kalman filter to predict and update fish positions, improving tracking accuracy.)
3. **Data Association** (Uses the Hungarian algorithm to match predicted positions with detected centroids.)
4. **Heatmap Generation** (Generates a heatmap showing the intensity of fish movement over time.)
5. **3D Trajectory Visualization** (Plots the 3D trajectories of fish over time.)

---

## Examples

### Example Requests
**Version 1:**
```bash
curl -X POST "http://localhost:8000/get_detectFish_v1" -H "Content-Type: application/json" -d '{"directory_path": "videos/input", "output_path": "videos/output"}'
```
**Version 2:**
```bash
curl -X POST "http://localhost:8000/get_detectFish_v2" -H "Content-Type: application/json" -d '{"directory_path": "videos/input", "output_path": "videos/output"}'
```

### Example Responses
**Version 1:**
```json
{
  "processed_videos": ["videos/output/processed_video1.mp4"],
  "background_sub_videos": ["videos/output/bg_subtracted_video1.mp4"]
}
```
**Version 2:**
```json
{
  "processed_videos": ["videos/output/processed_video1.mp4"],
  "background_sub_videos": ["videos/output/bg_subtracted_video1.mp4"],
  "heatmaps": ["videos/output/heatmap.png"],
  "trajectory_plots": ["videos/output/3d_trajectory.png"]
}
```

---

## Troubleshooting
- **No videos found**: Ensure the `directory_path` contains video files with supported formats (.mp4, .avi, .mov).
- **Invalid paths**: Verify that `directory_path` and `output_path` are correct and accessible.
- **Server errors**: Check the server logs for detailed error messages.

---

## Contributing
We welcome contributions! To contribute:
1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Submit a pull request with a detailed description of your changes.

---

## License
This project is licensed under the MIT License. See the LICENSE file for details.

For further assistance, please open an issue on the GitHub repository or contact the maintainers.

Happy tracking! 🐟


