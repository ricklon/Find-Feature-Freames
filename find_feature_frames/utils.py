import cv2
import os
import json
import zipfile
from pathlib import Path
from io import BytesIO
import numpy as np
from typing import Dict, Any, List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
MAX_UPLOAD_SIZE = 200 * 1024 * 1024  # 200MB in bytes
TEMP_DIR = Path("temp")
OUTPUT_DIR = Path("output")

# Ensure OUTPUT_DIR exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def ensure_dir(directory: Path) -> None:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        directory (Path): The directory path to ensure.
    """
    directory.mkdir(parents=True, exist_ok=True)

def create_zip(saved_frames: List[str]) -> BytesIO:
    """
    Create a zip file containing the saved frames.
    
    Args:
        saved_frames (List[str]): List of paths to saved frames.
    
    Returns:
        BytesIO: A buffer containing the zip file.
    """
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        for frame_path in saved_frames:
            zip_file.write(frame_path, Path(frame_path).name)
    zip_buffer.seek(0)
    return zip_buffer

def save_stats_json(stats: Dict[str, Any], output_folder: Path) -> None:
    """
    Save statistics to a JSON file.
    
    Args:
        stats (Dict[str, Any]): Dictionary of statistics to save.
        output_folder (Path): Path to the output folder.
    """
    stats_file = output_folder / "stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=4, cls=NumpyEncoder)

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types."""
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def get_video_stats(video_path: str) -> Dict[str, Any]:
    """
    Get statistics for a video file.
    
    Args:
        video_path (str): Path to the video file.
    
    Returns:
        Dict[str, Any]: Dictionary of video statistics.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Failed to open video file: {video_path}")
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

def handle_upload(uploaded_file: Any) -> Path:
    """
    Handle the upload of a video file.
    
    Args:
        uploaded_file (Any): The uploaded file object.
    
    Returns:
        Path: Path to the saved video file.
    """
    ensure_dir(TEMP_DIR)
    video_path = TEMP_DIR / uploaded_file.name
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    logging.info(f"File saved to {video_path}")
    return video_path

def clear_files_and_reset(st: Any) -> None:
    """
    Clear temporary files and reset the Streamlit session state.
    
    Args:
        st (Any): The Streamlit session state object.
    """
    if st.session_state.video_path:
        try:
            os.remove(st.session_state.video_path)
            logging.info(f"Removed file: {st.session_state.video_path}")
        except OSError as e:
            logging.error(f"Error removing file {st.session_state.video_path}: {e}")
    
    st.session_state.video_path = None
    st.session_state.output_folder = None
    st.session_state.saved_frames = None
    st.session_state.stats = None
    st.session_state.cancel_processing = False
    if 'zip_buffer' in st.session_state:
        del st.session_state.zip_buffer
    if 'stats_file' in st.session_state:
        del st.session_state.stats_file
    logging.info("Session state reset completed")
    st.success("All files and state have been cleared. You can upload a new video.")

def create_download_zip(output_folder: Path) -> BytesIO:
    """
    Create a zip file containing all files in the output folder.
    
    Args:
        output_folder (Path): Path to the output folder.
    
    Returns:
        BytesIO: A buffer containing the zip file.
    """
    if not output_folder.exists():
        raise FileNotFoundError(f"Output folder not found: {output_folder}")

    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        for file_path in output_folder.rglob('*'):
            if file_path.is_file():
                zip_file.write(file_path, file_path.relative_to(output_folder))
    zip_buffer.seek(0)
    logging.info(f"Created zip file for output folder: {output_folder}")
    return zip_buffer