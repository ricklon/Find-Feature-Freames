import cv2
import numpy as np
import os
from utils import ensure_dir, save_stats_json

def calculate_sharpness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def calculate_motion(prev_frame, curr_frame):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(prev_gray, curr_gray)
    return np.sum(diff) / (diff.shape[0] * diff.shape[1])

def calculate_contrast(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return np.max(gray) - np.min(gray)

def calculate_exposure(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    histogram = cv2.calcHist([gray], [0], None, [256], [0, 256])
    return np.mean(histogram)

def calculate_feature_density(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints = sift.detect(gray, None)
    return len(keypoints)

def calculate_feature_matches(prev_frame, curr_frame):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY), None)
    kp2, des2 = sift.detectAndCompute(cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY), None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
    return len(good_matches)

def calculate_camera_motion(prev_frame, curr_frame):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return np.mean(np.abs(flow))

def process_video_stream(video_path, output_folder, sharpness_threshold, motion_threshold, cancel_flag, progress_callback):
    cap = cv2.VideoCapture(video_path)
    prev_frame = None
    saved_frames = []
    frame_number = 0
    
    metrics = {
        "sharpness": [], "motion": [], "contrast": [], "exposure": [],
        "feature_density": [], "feature_matches": [], "camera_motion": []
    }
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ensure_dir(output_folder)
    
    while cap.isOpened():
        if cancel_flag():
            break
        
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_number += 1
        metrics["sharpness"].append(calculate_sharpness(frame))
        metrics["contrast"].append(calculate_contrast(frame))
        metrics["exposure"].append(calculate_exposure(frame))
        metrics["feature_density"].append(calculate_feature_density(frame))
        
        if prev_frame is not None:
            metrics["motion"].append(calculate_motion(prev_frame, frame))
            metrics["feature_matches"].append(calculate_feature_matches(prev_frame, frame))
            metrics["camera_motion"].append(calculate_camera_motion(prev_frame, frame))
        else:
            metrics["motion"].append(0)
            metrics["feature_matches"].append(0)
            metrics["camera_motion"].append(0)

        if metrics["sharpness"][-1] >= sharpness_threshold and metrics["motion"][-1] <= motion_threshold:
            filename = f"frame_{frame_number:05d}_sharpness_{metrics['sharpness'][-1]:.2f}_motion_{metrics['motion'][-1]:.2f}.jpg"
            filepath = os.path.join(output_folder, filename)
            cv2.imwrite(filepath, frame)
            saved_frames.append(filepath)

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
        "total_frames_processed": total_frames,
        "frames_extracted": len(saved_frames)
    }
    
    save_stats_json(stats, output_folder)
    
    return saved_frames, stats