import cv2
import numpy as np
import pandas as pd
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Callable
from utils import ensure_dir, save_stats_json
import logging
import traceback

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple, Callable
import logging

def process_video_stream(
    video_path: str,
    output_folder: str,
    cancel_flag: Callable[[], bool],
    progress_callback: Callable[[float], None]
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Process a video stream and collect data for all frames.

    Args:
        video_path (str): Path to the input video file.
        output_folder (str): Path to the output folder for saved frames.
        cancel_flag (Callable[[], bool]): Function to check if processing should be canceled.
        progress_callback (Callable[[float], None]): Function to report progress.

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: DataFrame with all frame data and dictionary of processing statistics.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Failed to open video file: {video_path}")
        return None, None

    frame_data = []
    frame_number = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    analyzer = FrameAnalyzer()

    while cap.isOpened():
        if cancel_flag():
            logging.info("Video processing canceled by user.")
            break

        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1
        metrics = {
            'frame_number': frame_number,
            'sharpness': analyzer.calculate_sharpness(frame),
            'contrast': analyzer.calculate_contrast(frame),
            'exposure': analyzer.calculate_exposure(frame),
            'feature_density': analyzer.calculate_feature_density(frame)
        }

        if frame_number > 1:
            metrics.update({
                'motion': analyzer.calculate_motion(prev_frame, frame),
                'feature_matches': analyzer.calculate_feature_matches(prev_frame, frame),
                'camera_motion': analyzer.calculate_camera_motion(prev_frame, frame)
            })
        else:
            metrics.update({
                'motion': 0,
                'feature_matches': 0,
                'camera_motion': 0
            })

        frame_data.append(metrics)
        prev_frame = frame

        if frame_number % 10 == 0 or frame_number == total_frames:
            progress_callback(frame_number / total_frames)

    cap.release()
    progress_callback(1.0)

    all_frame_data = pd.DataFrame(frame_data)

    stats = {
        "processing_parameters": {
            "total_frames_processed": total_frames,
        }
    }

    return all_frame_data, stats

# Make sure the FrameAnalyzer class is defined above this function
class FrameAnalyzer:
    @staticmethod
    def calculate_sharpness(frame: np.ndarray) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    @staticmethod
    def calculate_motion(prev_frame: np.ndarray, curr_frame: np.ndarray) -> float:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev_gray, curr_gray)
        return np.sum(diff) / (diff.shape[0] * diff.shape[1])

    @staticmethod
    def calculate_contrast(frame: np.ndarray) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return np.max(gray) - np.min(gray)

    @staticmethod
    def calculate_exposure(frame: np.ndarray) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        histogram = cv2.calcHist([gray], [0], None, [256], [0, 256])
        return np.mean(histogram)

    @staticmethod
    def calculate_feature_density(frame: np.ndarray) -> int:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        keypoints = sift.detect(gray, None)
        return len(keypoints)

    @staticmethod
    def calculate_feature_matches(prev_frame: np.ndarray, curr_frame: np.ndarray) -> int:
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY), None)
        kp2, des2 = sift.detectAndCompute(cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY), None)
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
        return len(good_matches)

    @staticmethod
    def calculate_camera_motion(prev_frame: np.ndarray, curr_frame: np.ndarray) -> float:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        return np.mean(np.abs(flow))