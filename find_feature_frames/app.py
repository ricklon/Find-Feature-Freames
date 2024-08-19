import streamlit as st
import cv2
import os
import uuid
import zipfile
from pathlib import Path
from io import BytesIO
import numpy as np

# Existing functions for sharpness and motion
def calculate_sharpness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var

def calculate_motion(prev_frame, curr_frame):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(prev_gray, curr_gray)
    motion = np.sum(diff) / (diff.shape[0] * diff.shape[1])  # Normalize by the number of pixels
    return motion

# New functions for contrast, exposure, feature density, feature matches, and camera motion
def calculate_contrast(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    min_intensity = np.min(gray)
    max_intensity = np.max(gray)
    contrast = max_intensity - min_intensity
    return contrast

def calculate_exposure(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    histogram = cv2.calcHist([gray], [0], None, [256], [0, 256])
    exposure = np.mean(histogram)
    return exposure

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
    curr_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    avg_flow = np.mean(np.abs(flow))
    return avg_flow

# Video statistics function
def get_video_stats(video_path):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return {"Error": "Failed to open video file."}
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if fps and fps > 0:
        duration = total_frames / fps
    else:
        duration = "Unknown (Invalid FPS)"
    
    cap.release()
    
    return {
        "Total Frames": total_frames,
        "FPS": fps if fps and fps > 0 else "Unknown",
        "Resolution": f"{width}x{height}",
        "Duration (seconds)": duration
    }

# Main processing function
def process_video_stream(video_path, output_folder, sharpness_threshold, motion_threshold, cancel_flag):
    cap = cv2.VideoCapture(video_path)
    prev_frame = None
    saved_frames = []
    frame_number = 0
    
    sharpness_values = []
    motion_values = []
    contrast_values = []
    exposure_values = []
    feature_density_values = []
    feature_match_values = []
    camera_motion_values = []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    progress_bar = st.progress(0)
    
    while cap.isOpened():
        if cancel_flag():
            st.write("Processing canceled.")
            break
        
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_number += 1
        sharpness = calculate_sharpness(frame)
        contrast = calculate_contrast(frame)
        exposure = calculate_exposure(frame)
        feature_density = calculate_feature_density(frame)
        
        sharpness_values.append(sharpness)
        contrast_values.append(contrast)
        exposure_values.append(exposure)
        feature_density_values.append(feature_density)
        
        if prev_frame is not None:
            motion = calculate_motion(prev_frame, frame)
            feature_matches = calculate_feature_matches(prev_frame, frame)
            camera_motion = calculate_camera_motion(prev_frame, frame)
        else:
            motion = 0
            feature_matches = 0
            camera_motion = 0
        
        motion_values.append(motion)
        feature_match_values.append(feature_matches)
        camera_motion_values.append(camera_motion)

        # Select the frame if it meets the criteria
        if sharpness >= sharpness_threshold and motion <= motion_threshold:
            filename = f"frame_{frame_number:05d}_sharpness_{sharpness:.2f}_motion_{motion:.2f}.jpg"
            filepath = os.path.join(output_folder, filename)
            cv2.imwrite(filepath, frame)
            saved_frames.append(filepath)

        prev_frame = frame

        # Update the progress bar every 10 frames to avoid slowing down processing
        if frame_number % 10 == 0 or frame_number == total_frames:
            progress_bar.progress(frame_number / total_frames)
    
    cap.release()
    progress_bar.progress(1.0)  # Ensure the progress bar is complete

    # Calculate statistics
    sharpness_stats = {"Max": np.max(sharpness_values), "Min": np.min(sharpness_values), "Avg": np.mean(sharpness_values)}
    motion_stats = {"Max": np.max(motion_values), "Min": np.min(motion_values), "Avg": np.mean(motion_values)}
    contrast_stats = {"Max": np.max(contrast_values), "Min": np.min(contrast_values), "Avg": np.mean(contrast_values)}
    exposure_stats = {"Max": np.max(exposure_values), "Min": np.min(exposure_values), "Avg": np.mean(exposure_values)}
    feature_density_stats = {"Max": np.max(feature_density_values), "Min": np.min(feature_density_values), "Avg": np.mean(feature_density_values)}
    feature_match_stats = {"Max": np.max(feature_match_values), "Min": np.min(feature_match_values), "Avg": np.mean(feature_match_values)}
    camera_motion_stats = {"Max": np.max(camera_motion_values), "Min": np.min(camera_motion_values), "Avg": np.mean(camera_motion_values)}
    
    return saved_frames, sharpness_stats, motion_stats, contrast_stats, exposure_stats, feature_density_stats, feature_match_stats, camera_motion_stats

# Function to create a ZIP file of saved frames
def create_zip(saved_frames):
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        for frame_path in saved_frames:
            zip_file.write(frame_path, os.path.basename(frame_path))
    zip_buffer.seek(0)
    return zip_buffer

# Streamlit Interface
st.title("Video Frame Extraction")
st.write("Upload a video, and we will extract the best frames based on sharpness and motion criteria.")

# Set a custom file size limit for upload (e.g., 200MB)
max_upload_size = 200 * 1024 * 1024  # 200MB in bytes
st.write(f"Maximum file size for upload: {max_upload_size / (1024 * 1024)} MB")

# Upload video file with file size limit check
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])

# Store the cancel flag in session state
if 'cancel_processing' not in st.session_state:
    st.session_state.cancel_processing = False

# Reset the cancel flag when a new file is uploaded
if uploaded_file is not None:
    st.session_state.cancel_processing = False

    # Save the uploaded file to a temporary location
    video_path = os.path.join("temp", uploaded_file.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display video statistics
    video_stats = get_video_stats(video_path)
    st.write("### Video Statistics")
    st.write(video_stats)

# User inputs for sharpness and motion thresholds
sharpness_threshold = st.slider("Sharpness Threshold", min_value=0.0, max_value=1000.0, value=150.0)
motion_threshold = st.slider("Motion Threshold", min_value=0.0, max_value=10000.0, value=200.0)

# Run button to start processing
if st.button("Run"):
    if uploaded_file is not None and uploaded_file.size <= max_upload_size:
        # Create a unique output folder
        output_folder = os.path.join("output", str(uuid.uuid4()))
        
        # Process video and get best frames
        with st.spinner("Processing video..."):
            saved_frames, sharpness_stats, motion_stats, contrast_stats, exposure_stats, feature_density_stats, feature_match_stats, camera_motion_stats = process_video_stream(
                video_path, output_folder, sharpness_threshold, motion_threshold, 
                lambda: st.session_state.cancel_processing
            )

        # Display saved frames
        st.success(f"Extracted {len(saved_frames)} frames that meet the criteria.")
        
        # Show statistics
        st.write("### Sharpness Statistics")
        st.write(sharpness_stats)
        
        st.write("### Motion Statistics")
        st.write(motion_stats)
        
        st.write("### Contrast Statistics")
        st.write(contrast_stats)

        st.write("### Exposure Statistics")
        st.write(exposure_stats)

        st.write("### Feature Density Statistics")
        st.write(feature_density_stats)

        st.write("### Feature Match Statistics")
        st.write(feature_match_stats)

        st.write("### Camera Motion Statistics")
        st.write(camera_motion_stats)
        
        # Provide a download link for the ZIP file
        if saved_frames:
            zip_buffer = create_zip(saved_frames)
            st.download_button(
                label="Download ZIP",
                data=zip_buffer,
                file_name="extracted_frames.zip",
                mime="application/zip"
            )
        
        # Display images in a 4-column gallery layout
        cols = st.columns(4)
        for i, frame_path in enumerate(saved_frames):
            with cols[i % 4]:
                st.image(frame_path, caption=os.path.basename(frame_path), use_column_width=True)
        
    else:
        if uploaded_file is not None and uploaded_file.size > max_upload_size:
            st.error(f"The uploaded file exceeds the maximum allowed size of {max_upload_size / (1024 * 1024)} MB. Please upload a smaller file.")

# Cancel button to stop processing
if st.button("Cancel"):
    st.session_state.cancel_processing = True
    st.warning("Canceling processing...")

# Clear files and reset
if st.button("Clear Files and Reset"):
    if 'video_path' in st.session_state and st.session_state.video_path:
        os.remove(st.session_state.video_path)
        st.session_state.video_path = None
    st.session_state.cancel_processing = False
    st.success("Temporary files deleted. You can upload a new video.")
