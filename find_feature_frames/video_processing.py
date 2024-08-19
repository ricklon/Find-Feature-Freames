import cv2
import numpy as np
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Callable
from utils import ensure_dir, save_stats_json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FrameAnalyzer:
    """Class for analyzing video frames and calculating various metrics."""

    @staticmethod
    def calculate_sharpness(frame: np.ndarray) -> float:
        """Calculate the sharpness of a frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    @staticmethod
    def calculate_motion(prev_frame: np.ndarray, curr_frame: np.ndarray) -> float:
        """Calculate the motion between two consecutive frames."""
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev_gray, curr_gray)
        return np.sum(diff) / (diff.shape[0] * diff.shape[1])

    @staticmethod
    def calculate_contrast(frame: np.ndarray) -> float:
        """Calculate the contrast of a frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return np.max(gray) - np.min(gray)

    @staticmethod
    def calculate_exposure(frame: np.ndarray) -> float:
        """Calculate the exposure of a frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        histogram = cv2.calcHist([gray], [0], None, [256], [0, 256])
        return np.mean(histogram)

    @staticmethod
    def calculate_feature_density(frame: np.ndarray) -> int:
        """Calculate the feature density of a frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        keypoints = sift.detect(gray, None)
        return len(keypoints)

    @staticmethod
    def calculate_feature_matches(prev_frame: np.ndarray, curr_frame: np.ndarray) -> int:
        """Calculate the number of feature matches between two consecutive frames."""
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY), None)
        kp2, des2 = sift.detectAndCompute(cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY), None)
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
        return len(good_matches)

    @staticmethod
    def calculate_camera_motion(prev_frame: np.ndarray, curr_frame: np.ndarray) -> float:
        """Calculate the camera motion between two consecutive frames."""
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        return np.mean(np.abs(flow))

def process_video_stream(
    video_path: str,
    output_folder: str,
    sharpness_threshold: float,
    motion_threshold: float,
    contrast_range: Tuple[float, float],
    exposure_range: Tuple[float, float],
    feature_density_range: Tuple[float, float],
    feature_matches_range: Tuple[float, float],
    camera_motion_range: Tuple[float, float],
    cancel_flag: Callable[[], bool],
    progress_callback: Callable[[float], None]
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Process a video stream and extract frames based on specified criteria.

    Args:
        video_path (str): Path to the input video file.
        output_folder (str): Path to the output folder for saved frames.
        sharpness_threshold (float): Minimum sharpness threshold.
        motion_threshold (float): Maximum motion threshold.
        contrast_range (Tuple[float, float]): Range of acceptable contrast values.
        exposure_range (Tuple[float, float]): Range of acceptable exposure values.
        feature_density_range (Tuple[float, float]): Range of acceptable feature density values.
        feature_matches_range (Tuple[float, float]): Range of acceptable feature matches values.
        camera_motion_range (Tuple[float, float]): Range of acceptable camera motion values.
        cancel_flag (Callable[[], bool]): Function to check if processing should be canceled.
        progress_callback (Callable[[float], None]): Function to report progress.

    Returns:
        Tuple[List[str], Dict[str, Any]]: List of saved frame paths and dictionary of processing statistics.
    """
    cap = cv2.VideoCapture(video_path)
    prev_frame = None
    saved_frames = []
    frame_number = 0
    
    metrics = {
        "sharpness": [], "motion": [], "contrast": [], "exposure": [],
        "feature_density": [], "feature_matches": [], "camera_motion": []
    }
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ensure_dir(Path(output_folder))
    
    analyzer = FrameAnalyzer()
    
    while cap.isOpened():
        if cancel_flag():
            logging.info("Video processing canceled by user.")
            break
        
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_number += 1
        metrics["sharpness"].append(analyzer.calculate_sharpness(frame))
        metrics["contrast"].append(analyzer.calculate_contrast(frame))
        metrics["exposure"].append(analyzer.calculate_exposure(frame))
        metrics["feature_density"].append(analyzer.calculate_feature_density(frame))
        
        if prev_frame is not None:
            metrics["motion"].append(analyzer.calculate_motion(prev_frame, frame))
            metrics["feature_matches"].append(analyzer.calculate_feature_matches(prev_frame, frame))
            metrics["camera_motion"].append(analyzer.calculate_camera_motion(prev_frame, frame))
        else:
            metrics["motion"].append(0)
            metrics["feature_matches"].append(0)
            metrics["camera_motion"].append(0)

        # Check all criteria for frame selection
        if (metrics["sharpness"][-1] >= sharpness_threshold and
            metrics["motion"][-1] <= motion_threshold and
            contrast_range[0] <= metrics["contrast"][-1] <= contrast_range[1] and
            exposure_range[0] <= metrics["exposure"][-1] <= exposure_range[1] and
            feature_density_range[0] <= metrics["feature_density"][-1] <= feature_density_range[1] and
            feature_matches_range[0] <= metrics["feature_matches"][-1] <= feature_matches_range[1] and
            camera_motion_range[0] <= metrics["camera_motion"][-1] <= camera_motion_range[1]):
            
            filename = f"frame_{frame_number:05d}_sharp_{metrics['sharpness'][-1]:.2f}_motion_{metrics['motion'][-1]:.2f}.jpg"
            filepath = os.path.join(output_folder, filename)
            cv2.imwrite(filepath, frame)
            saved_frames.append(filepath)
            logging.info(f"Saved frame: {filename}")

        prev_frame = frame

        if frame_number % 10 == 0 or frame_number == total_frames:
            progress_callback(frame_number / total_frames)
    
    cap.release()
    progress_callback(1.0)

    # Handle the case where no frames meet the criteria
    stats = {}
    for key, value in metrics.items():
        if value:  # Check if the list is not empty
            stats[key] = {
                "Max": np.max(value),
                "Min": np.min(value),
                "Avg": np.mean(value)
            }
        else:
            stats[key] = {
                "Max": None,
                "Min": None,
                "Avg": None
            }

    stats["processing_parameters"] = {
        "sharpness_threshold": sharpness_threshold,
        "motion_threshold": motion_threshold,
        "contrast_range": contrast_range,
        "exposure_range": exposure_range,
        "feature_density_range": feature_density_range,
        "feature_matches_range": feature_matches_range,
        "camera_motion_range": camera_motion_range,
        "total_frames_processed": total_frames,
        "frames_extracted": len(saved_frames)
    }
    
    save_stats_json(stats, Path(output_folder))
    logging.info(f"Video processing completed. Extracted {len(saved_frames)} frames.")
    
    return saved_frames, stats