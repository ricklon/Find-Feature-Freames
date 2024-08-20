import cv2
import numpy as np
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Callable
from utils import ensure_dir, save_stats_json
import logging
import traceback

def process_video_stream(
    video_path: str,
    output_folder: str,
    filter_settings: Dict[str, Tuple[float, float]],
    cancel_flag: Callable[[], bool],
    progress_callback: Callable[[float], None]
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Process a video stream and extract frames based on specified criteria.

    Args:
        video_path (str): Path to the input video file.
        output_folder (str): Path to the output folder for saved frames.
        filter_settings (Dict[str, Tuple[float, float]]): Dictionary of filter settings for each metric.
        cancel_flag (Callable[[], bool]): Function to check if processing should be canceled.
        progress_callback (Callable[[float], None]): Function to report progress.

    Returns:
        Tuple[List[str], Dict[str, Any]]: List of saved frame paths and dictionary of processing statistics.
    """
    logging.info(f"Starting video processing for {video_path}")
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"Failed to open video file: {video_path}")
            return [], {}

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
            if all(filter_settings[metric][0] <= metrics[metric][-1] <= filter_settings[metric][1] 
                   for metric in filter_settings):
                
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
                    "Avg": np.mean(value),
                    "Values": value
                }
            else:
                stats[key] = {
                    "Max": None,
                    "Min": None,
                    "Avg": None,
                    "Values": []
                }

        stats["processing_parameters"] = {
            "filter_settings": filter_settings,
            "total_frames_processed": total_frames,
            "frames_extracted": len(saved_frames)
        }
        
        save_stats_json(stats, Path(output_folder))
        logging.info(f"Video processing completed. Extracted {len(saved_frames)} frames.")
        logging.debug(f"Stats keys: {stats.keys()}")
        logging.debug(f"Sharpness stats: {stats.get('sharpness', 'Not found')}")
        
        return saved_frames, stats
    
    except Exception as e:
        logging.error(f"Error in process_video_stream: {str(e)}")
        logging.error(f"Error details: {traceback.format_exc()}")
        return [], {}  # Return empty lists in case of an error

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