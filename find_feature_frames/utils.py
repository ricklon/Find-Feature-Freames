import cv2
import os
import json
import zipfile
from pathlib import Path
from io import BytesIO
import numpy as np

# Constants
MAX_UPLOAD_SIZE = 200 * 1024 * 1024  # 200MB in bytes
TEMP_DIR = "temp"
OUTPUT_DIR = "output"

def ensure_dir(directory):
    Path(directory).mkdir(parents=True, exist_ok=True)

def create_zip(saved_frames):
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        for frame_path in saved_frames:
            zip_file.write(frame_path, os.path.basename(frame_path))
    zip_buffer.seek(0)
    return zip_buffer

def save_stats_json(stats, output_folder):
    stats_file = os.path.join(output_folder, "stats.json")
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=4, cls=NumpyEncoder)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def get_video_stats(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"Error": "Failed to open video file."}
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps and fps > 0 else "Unknown (Invalid FPS)"
    
    cap.release()
    return {
        "Total Frames": total_frames,
        "FPS": fps if fps and fps > 0 else "Unknown",
        "Resolution": f"{width}x{height}",
        "Duration (seconds)": duration
    }

def handle_upload(uploaded_file):
    ensure_dir(TEMP_DIR)
    video_path = os.path.join(TEMP_DIR, uploaded_file.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return video_path

def clear_files_and_reset(st):
    if st.session_state.video_path:
        os.remove(st.session_state.video_path)
    st.session_state.video_path = None
    st.session_state.output_folder = None
    st.session_state.saved_frames = None
    st.session_state.stats = None
    st.session_state.cancel_processing = False
    if 'zip_buffer' in st.session_state:
        del st.session_state.zip_buffer
    if 'stats_file' in st.session_state:
        del st.session_state.stats_file
    st.success("All files and state have been cleared. You can upload a new video.")

def create_download_zip(output_folder):
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        for root, dirs, files in os.walk(output_folder):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, output_folder)
                zip_file.write(file_path, arcname)
    zip_buffer.seek(0)
    return zip_buffer