import streamlit as st
import cv2
import os
from pathlib import Path
import uuid
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
from utils import MAX_UPLOAD_SIZE, OUTPUT_DIR, create_zip, get_video_stats, handle_upload, clear_files_and_reset, create_download_zip
from video_processing import process_video_stream
from visualization import create_visualizations, suggest_filter_adjustments, create_filter_comparison
import logging
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Type aliases
FilterSettings = Dict[str, Tuple[float, float]]
Stats = Dict[str, Any]

def init_session_state() -> None:
    """Initialize session state variables."""
    if 'video_path' not in st.session_state:
        st.session_state.video_path = None
    if 'output_folder' not in st.session_state:
        st.session_state.output_folder = None
    if 'saved_frames' not in st.session_state:
        st.session_state.saved_frames = None
    if 'stats' not in st.session_state:
        st.session_state.stats = None
    if 'cancel_processing' not in st.session_state:
        st.session_state.cancel_processing = False
    if 'original_video_path' not in st.session_state:
        st.session_state.original_video_path = None
    if 'filter_settings' not in st.session_state:
        st.session_state.filter_settings = {
            'sharpness': (350.0, float('inf')),
            'motion': (0.0, 200.0),
            'contrast': (100, 255),
            'exposure': (100, 7000),
            'feature_density': (500, 6000),
            'feature_matches': (0, 4000),
            'camera_motion': (0.0, 5.0)
        }

def extract_and_save_frames(video_path: str, frame_numbers: List[int], output_folder: str) -> List[str]:
    cap = cv2.VideoCapture(video_path)
    saved_frames = []
    for frame_number in frame_numbers:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
        ret, frame = cap.read()
        if ret:
            filename = f"frame_{frame_number:05d}.jpg"
            filepath = os.path.join(output_folder, filename)
            cv2.imwrite(filepath, frame)
            saved_frames.append(filepath)
    cap.release()
    return saved_frames

def refine_filters_and_extract_frames(stats: Dict[str, Any], filter_settings: Dict[str, Tuple[float, float]], video_path: str, output_folder: str) -> Tuple[List[str], Dict[str, Any]]:
    df = pd.DataFrame({metric: stats[metric]['Values'] for metric in stats if isinstance(stats[metric], dict) and 'Values' in stats[metric]})
    
    # Apply filters
    mask = pd.Series(True, index=df.index)
    for metric, (min_val, max_val) in filter_settings.items():
        if metric in df.columns:
            mask &= (df[metric] >= min_val) & (df[metric] <= max_val)
    
    selected_frames = df[mask].index.tolist()
    
    # Extract and save selected frames
    saved_frames = extract_and_save_frames(video_path, selected_frames, output_folder)
    
    # Update stats
    new_stats = {metric: {key: value for key, value in stat.items() if key != 'Values'} for metric, stat in stats.items()}
    for metric in new_stats:
        if isinstance(new_stats[metric], dict) and 'Values' in stats[metric]:
            new_stats[metric]['Values'] = [val for i, val in enumerate(stats[metric]['Values']) if i in selected_frames]
    
    new_stats['processing_parameters'] = stats['processing_parameters']
    new_stats['processing_parameters']['frames_extracted'] = len(saved_frames)
    
    return saved_frames, new_stats

def render_ui() -> str:
    """Render the main UI elements and handle file upload."""
    st.write("Upload a video, and we will extract the best frames based on various criteria.")
    st.write(f"Maximum file size for upload: {MAX_UPLOAD_SIZE / (1024 * 1024)} MB")

    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])
    
    if uploaded_file is not None:
        logging.info("File uploaded")
        video_path = handle_upload(uploaded_file)
        st.session_state.video_path = video_path
        st.session_state.new_upload = True
        logging.info(f"Video path set to {video_path}")
        display_video_stats(video_path)
    
    return st.session_state.video_path

def display_video_stats(video_path: str) -> None:
    """Display statistics for the uploaded video."""
    video_stats = get_video_stats(video_path)
    st.write("### Video Statistics")
    st.write(video_stats)

@st.cache_data
def apply_suggested_settings(current_settings: FilterSettings, suggestions: Dict[str, Dict[str, float]]) -> FilterSettings:
    """Apply suggested filter settings."""
    new_settings = current_settings.copy()
    for metric, suggestion in suggestions.items():
        if metric in ['sharpness', 'motion', 'camera_motion']:
            new_settings[metric] = (float(suggestion['suggested_min']), float(suggestion['suggested_max']))
        else:
            new_settings[metric] = (int(suggestion['suggested_min']), int(suggestion['suggested_max']))
    return new_settings

def render_filter_settings(filter_settings: FilterSettings) -> FilterSettings:
    """Render UI for filter settings and return updated settings."""
    st.write("### Filter Settings")

    # Display the user guide
    # Create a collapsible section for the user guide
    with st.expander("How to Use the Filters", expanded=False):
        st.markdown("""
        These filters help you select the best frames from your video based on various criteria:

        1. **Sharpness**: Higher values mean sharper images. Increase this to keep only the sharpest frames.

        2. **Motion**: Lower values mean less motion between frames. Decrease this to capture more stable scenes.

        3. **Contrast**: Adjust the range to select frames with desired contrast levels.

        4. **Exposure**: Set the range to capture frames with proper exposure (not too dark or bright).

        5. **Feature Density**: Higher values mean more detailed frames. Adjust to preference.

        6. **Feature Matches**: Higher values indicate more similarity between consecutive frames.

        7. **Camera Motion**: Lower values mean less camera movement. Adjust based on desired stability.

        Tip: Start with default settings and adjust gradually. You can always refine your selection after the initial processing.
        """)

    new_settings: FilterSettings = {}
    
    new_settings['sharpness'] = (
        st.slider("Sharpness Threshold (Higher is sharper)", 
                  min_value=0.0, 
                  max_value=10000.0,
                  value=float(filter_settings['sharpness'][0]), 
                  step=1.0),
        float('inf')
    )
    
    new_settings['motion'] = (0.0, st.slider("Motion Threshold (Lower means less motion)", 
                                             min_value=0.0, 
                                             max_value=10000.0, 
                                             value=float(filter_settings['motion'][1]), 
                                             step=0.1))
  
    
    new_settings['contrast'] = tuple(st.slider("Contrast Range", 
                                               min_value=0, 
                                               max_value=255, 
                                               value=(int(filter_settings['contrast'][0]), int(filter_settings['contrast'][1]))))
    
    new_settings['exposure'] = tuple(st.slider("Exposure Range (average pixel value)", 
                                               min_value=0, 
                                               max_value=10000, 
                                               value=(int(filter_settings['exposure'][0]), int(filter_settings['exposure'][1]))))
    
    new_settings['feature_density'] = tuple(st.slider("Feature Density Range (features per frame)", 
                                                      min_value=0, 
                                                      max_value=10000, 
                                                      value=(int(filter_settings['feature_density'][0]), int(filter_settings['feature_density'][1]))))
    
    new_settings['feature_matches'] = tuple(st.slider("Feature Matches Range (matches between consecutive frames)", 
                                                      min_value=0, 
                                                      max_value=5000, 
                                                      value=(int(filter_settings['feature_matches'][0]), int(filter_settings['feature_matches'][1]))))
    
    new_settings['camera_motion'] = tuple(st.slider("Camera Motion Range (average pixel displacement)", 
                                                    min_value=0.0, 
                                                    max_value=10.0, 
                                                    value=(float(filter_settings['camera_motion'][0]), float(filter_settings['camera_motion'][1])), 
                                                    step=0.1))
    
    return new_settings

def display_results() -> None:
    if st.session_state.saved_frames is None or st.session_state.filtered_stats is None:
        st.info("No results to display. Please run the processing first.")
        return

    saved_frames = st.session_state.saved_frames
    stats = st.session_state.filtered_stats
    output_folder = st.session_state.output_folder
    filter_settings = st.session_state.filter_settings

    if len(saved_frames) == 0:
        st.warning("No frames met the specified criteria. Try adjusting the thresholds.")
    else:
        st.success(f"Extracted {len(saved_frames)} frames that meet the criteria.")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Statistics and Management", "Visualizations", "Extracted Frames", "Filter Analysis"])
    
    with tab1:
        display_statistics_and_management(stats, output_folder)
    
    with tab2:
        display_visualizations(stats, saved_frames, filter_settings)
    
    with tab3:
        display_extracted_frames(saved_frames)
    
    with tab4:
        display_filter_analysis(stats, filter_settings)
    




def display_visualizations(stats: Stats, saved_frames: List[str], filter_settings: FilterSettings) -> None:
    """Display visualizations of processing results."""
    st.write("### Visualizations and Summary Statistics")
    logging.debug(f"Stats keys before visualization: {stats.keys()}")
    logging.debug(f"Sharpness data in stats: {stats.get('sharpness', 'Not found')}")
    if "processing_parameters" in stats and any(stats[key]["Avg"] is not None for key in stats if key != "processing_parameters"):
        try:
            fig_timeline, fig1, fig2, summary_stats, filter_comparison, filter_suggestions = create_visualizations(stats, saved_frames, filter_settings)
            
            st.plotly_chart(fig_timeline, use_container_width=True)
            st.plotly_chart(fig1, use_container_width=True)
            st.plotly_chart(fig2, use_container_width=True)
            
            st.write("### Summary Statistics for Metrics")
            for metric, values in summary_stats.items():
                st.write(f"**{metric.capitalize()}**")
                for stat, value in values.items():
                    st.write(f"- {stat}: {value:.2f}")
                st.write("")
            
        except Exception as e:
            logging.error(f"An error occurred while creating visualizations: {str(e)}")
            st.error("An error occurred while creating visualizations. Please check the logs for more details.")
    else:
        st.info("No data available for visualization. This may be because no frames met the criteria.")

def display_extracted_frames(saved_frames: List[str]) -> None:
    """Display extracted frames in a grid layout."""
    if saved_frames:
        st.write("### Extracted Frames")
        cols = st.columns(4)
        for i, frame_path in enumerate(saved_frames):
            with cols[i % 4]:
                st.image(frame_path, caption=os.path.basename(frame_path), use_column_width=True)
    else:
        st.info("No frames to display. Try adjusting the thresholds.")

def display_filter_analysis(stats: Stats, filter_settings: FilterSettings) -> None:
    """Display filter analysis and adjustment suggestions."""
    st.write("### Filter Analysis")
    filter_comparison = create_filter_comparison(stats, filter_settings)
    for metric, data in filter_comparison.items():
        st.write(f"**{metric.capitalize()}**")
        
        # Display the current filter range
        if metric in ['sharpness', 'motion', 'camera_motion']:
            st.write(f"Current range: [{filter_settings[metric][0]:.2f}, {filter_settings[metric][1]:.2f}]")
        else:
            st.write(f"Current range: [{filter_settings[metric][0]}, {filter_settings[metric][1]}]")
        
        st.write(f"- Frames within range: {data['frames_within_range']} out of {data['total_frames']} ({data['percentage_within_range']:.2f}%)")
        st.write(f"- Closest value below range: {data['closest_below']:.2f}" if data['closest_below'] is not None else "- No values below range")
        st.write(f"- Closest value above range: {data['closest_above']:.2f}" if data['closest_above'] is not None else "- No values above range")
        st.write("")

    st.write("### Filter Adjustment Suggestions")
    filter_suggestions = suggest_filter_adjustments(filter_comparison, filter_settings)
    if filter_suggestions:
        for metric, suggestion in filter_suggestions.items():
            st.write(f"**{metric.capitalize()}**")
            st.write(f"- Current range: [{suggestion['current_min']:.2f}, {suggestion['current_max']:.2f}]")
            st.write(f"- Suggested range: [{suggestion['suggested_min']:.2f}, {suggestion['suggested_max']:.2f}]")
            st.write("")
        
        if st.button("Apply Suggested Settings"):
            new_settings = apply_suggested_settings(filter_settings, filter_suggestions)
            st.session_state.filter_settings = new_settings
            st.success("Suggested settings applied. Please run the processing again with the new settings.")
            st.rerun()
    else:
        st.write("No filter adjustments suggested. Current settings appear to be optimal.")

def process_video(video_path: str) -> None:
    """Process the uploaded video and update session state with results."""
    output_folder = Path(OUTPUT_DIR) / str(uuid.uuid4())
    
    with st.spinner("Processing video..."):
        progress_bar = st.progress(0)
        filter_settings = st.session_state.filter_settings
        try:
            result = process_video_stream(
                video_path=video_path,
                output_folder=str(output_folder),
                filter_settings=filter_settings,
                cancel_flag=lambda: st.session_state.cancel_processing,
                progress_callback=lambda progress: progress_bar.progress(progress)
            )
            if result is None:
                st.error("Video processing failed. Please check the logs for more information.")
                return
            saved_frames, stats = result
            st.session_state.saved_frames = saved_frames
            st.session_state.stats = stats
            st.session_state.output_folder = output_folder
            st.session_state.original_video_path = video_path  # Store the original video path
            st.success("Video processing completed!")
            
            # Log the stats for debugging
            logging.debug(f"Stats after processing: {stats}")
            
        except Exception as e:
            st.error(f"An error occurred during video processing: {str(e)}")
            logging.error(f"Video processing error: {str(e)}")
            return
    st.rerun()

def apply_filters(all_frame_data: pd.DataFrame, filter_settings: Dict[str, Tuple[float, float]], video_path: str, output_folder: str) -> Tuple[List[str], Dict[str, Any]]:
    # Apply filters
    mask = pd.Series(True, index=all_frame_data.index)
    for metric, (min_val, max_val) in filter_settings.items():
        if metric in all_frame_data.columns:
            if np.isinf(max_val):  # Handle infinity for upper bound
                mask &= (all_frame_data[metric] >= min_val)
            else:
                mask &= (all_frame_data[metric] >= min_val) & (all_frame_data[metric] <= max_val)
    
    filtered_data = all_frame_data[mask]
    selected_frames = filtered_data['frame_number'].tolist()
    
    # Extract and save selected frames
    saved_frames = extract_and_save_frames(video_path, selected_frames, output_folder)
    
    # Create filtered stats
    filtered_stats = {}
    for metric in all_frame_data.columns:
        if metric != 'frame_number':
            filtered_stats[metric] = {
                'Max': filtered_data[metric].max(),
                'Min': filtered_data[metric].min(),
                'Avg': filtered_data[metric].mean(),
                'Values': filtered_data[metric].tolist()
            }
    
    filtered_stats['processing_parameters'] = {
        'filter_settings': filter_settings,
        'total_frames_processed': len(all_frame_data),
        'frames_extracted': len(saved_frames)
    }
    
    return saved_frames, filtered_stats

def display_statistics_and_management(stats: Stats, output_folder: Path) -> None:
    """Display processing statistics and file management options."""
    st.write("### Processing Statistics")
    st.write("Full statistics have been saved in stats.json in the output folder.")
    st.write(f"Output folder: {output_folder}")
    
    st.write("Summary of key statistics:")
    for stat_name, stat_values in stats.items():
        if stat_name != "processing_parameters":
            if stat_values["Avg"] is not None:
                st.write(f"- {stat_name.capitalize()}: Avg = {stat_values['Avg']:.2f}")
            else:
                st.write(f"- {stat_name.capitalize()}: No data available")
    
    st.write("### Processing Parameters")
    for param, value in stats["processing_parameters"].items():
        st.write(f"- {param.replace('_', ' ').capitalize()}: {value}")
    
    zip_buffer = create_download_zip(output_folder)
    st.download_button(
        label="Download Results",
        data=zip_buffer,
        file_name="video_analysis_results.zip",
        mime="application/zip",
        key=f"download_button_{output_folder}"  # Add a unique key based on the output folder
    )

def main() -> None:
    st.title("Video Frame Extraction and Analysis")

    # Initialize session state
    init_session_state()

    video_path = render_ui()

    # Render filter settings and get the new settings
    new_filter_settings = render_filter_settings(st.session_state.filter_settings)

    # Check if there are unapplied changes
    unapplied_changes = new_filter_settings != st.session_state.filter_settings
    if unapplied_changes:
        st.warning("You have unapplied filter changes. Click 'Run' to apply them.")

    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("Run"):
            if video_path:
                if os.path.getsize(video_path) <= MAX_UPLOAD_SIZE:
                    # Apply the new filter settings
                    st.session_state.filter_settings = new_filter_settings
                    if 'all_frame_data' not in st.session_state or st.session_state.original_video_path != video_path:
                        # New video uploaded or first run, process it fully
                        process_video(video_path)
                    else:
                        # Apply filters to existing data
                        saved_frames, filtered_stats = apply_filters(
                            st.session_state.all_frame_data,
                            st.session_state.filter_settings,
                            st.session_state.original_video_path,
                            str(st.session_state.output_folder)
                        )
                        st.session_state.saved_frames = saved_frames
                        st.session_state.filtered_stats = filtered_stats
                        st.success("Filter application completed!")
                    # Force a rerun to update the visualizations
                    st.rerun()
                else:
                    st.error(f"The uploaded file exceeds the maximum allowed size of {MAX_UPLOAD_SIZE / (1024 * 1024)} MB. Please upload a smaller file.")
            else:
                st.error("Please upload a video file before processing.")
    with col2:
        if st.button("Cancel"):
            st.session_state.cancel_processing = True
            st.warning("Canceling processing...")
            st.rerun()

    with col3:
        if st.button("Clear Results"):
            clear_files_and_reset(st)
            st.rerun()

import streamlit as st
import cv2
import os
from pathlib import Path
import uuid
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
from utils import MAX_UPLOAD_SIZE, OUTPUT_DIR, create_zip, get_video_stats, handle_upload, clear_files_and_reset, create_download_zip
from video_processing import process_video_stream
from visualization import create_visualizations, suggest_filter_adjustments, create_filter_comparison
import logging
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Type aliases
FilterSettings = Dict[str, Tuple[float, float]]
Stats = Dict[str, Any]

def init_session_state() -> None:
    """Initialize session state variables."""
    if 'video_path' not in st.session_state:
        st.session_state.video_path = None
    if 'output_folder' not in st.session_state:
        st.session_state.output_folder = None
    if 'saved_frames' not in st.session_state:
        st.session_state.saved_frames = None
    if 'stats' not in st.session_state:
        st.session_state.stats = None
    if 'cancel_processing' not in st.session_state:
        st.session_state.cancel_processing = False
    if 'original_video_path' not in st.session_state:
        st.session_state.original_video_path = None
    if 'filter_settings' not in st.session_state:
        st.session_state.filter_settings = {
            'sharpness': (350.0, float('inf')),
            'motion': (0.0, 200.0),
            'contrast': (100, 255),
            'exposure': (100, 7000),
            'feature_density': (500, 6000),
            'feature_matches': (0, 4000),
            'camera_motion': (0.0, 5.0)
        }

def extract_and_save_frames(video_path: str, frame_numbers: List[int], output_folder: str) -> List[str]:
    cap = cv2.VideoCapture(video_path)
    saved_frames = []
    for frame_number in frame_numbers:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
        ret, frame = cap.read()
        if ret:
            filename = f"frame_{frame_number:05d}.jpg"
            filepath = os.path.join(output_folder, filename)
            cv2.imwrite(filepath, frame)
            saved_frames.append(filepath)
    cap.release()
    return saved_frames

def refine_filters_and_extract_frames(stats: Dict[str, Any], filter_settings: Dict[str, Tuple[float, float]], video_path: str, output_folder: str) -> Tuple[List[str], Dict[str, Any]]:
    df = pd.DataFrame({metric: stats[metric]['Values'] for metric in stats if isinstance(stats[metric], dict) and 'Values' in stats[metric]})
    
    # Apply filters
    mask = pd.Series(True, index=df.index)
    for metric, (min_val, max_val) in filter_settings.items():
        if metric in df.columns:
            mask &= (df[metric] >= min_val) & (df[metric] <= max_val)
    
    selected_frames = df[mask].index.tolist()
    
    # Extract and save selected frames
    saved_frames = extract_and_save_frames(video_path, selected_frames, output_folder)
    
    # Update stats
    new_stats = {metric: {key: value for key, value in stat.items() if key != 'Values'} for metric, stat in stats.items()}
    for metric in new_stats:
        if isinstance(new_stats[metric], dict) and 'Values' in stats[metric]:
            new_stats[metric]['Values'] = [val for i, val in enumerate(stats[metric]['Values']) if i in selected_frames]
    
    new_stats['processing_parameters'] = stats['processing_parameters']
    new_stats['processing_parameters']['frames_extracted'] = len(saved_frames)
    
    return saved_frames, new_stats

def render_ui() -> str:
    """Render the main UI elements and handle file upload."""
    st.write("Upload a video, and we will extract the best frames based on various criteria.")
    st.write(f"Maximum file size for upload: {MAX_UPLOAD_SIZE / (1024 * 1024)} MB")

    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])
    
    if uploaded_file is not None:
        logging.info("File uploaded")
        video_path = handle_upload(uploaded_file)
        st.session_state.video_path = video_path
        st.session_state.new_upload = True
        logging.info(f"Video path set to {video_path}")
        display_video_stats(video_path)
    
    return st.session_state.video_path

def display_video_stats(video_path: str) -> None:
    """Display statistics for the uploaded video."""
    video_stats = get_video_stats(video_path)
    st.write("### Video Statistics")
    st.write(video_stats)

@st.cache_data
def apply_suggested_settings(current_settings: FilterSettings, suggestions: Dict[str, Dict[str, float]]) -> FilterSettings:
    """Apply suggested filter settings."""
    new_settings = current_settings.copy()
    for metric, suggestion in suggestions.items():
        if metric in ['sharpness', 'motion', 'camera_motion']:
            new_settings[metric] = (float(suggestion['suggested_min']), float(suggestion['suggested_max']))
        else:
            new_settings[metric] = (int(suggestion['suggested_min']), int(suggestion['suggested_max']))
    return new_settings

def render_filter_settings(filter_settings: FilterSettings) -> FilterSettings:
    """Render UI for filter settings and return updated settings."""
    st.write("### Filter Settings")

    # Display the user guide
    # Create a collapsible section for the user guide
    with st.expander("How to Use the Filters", expanded=False):
        st.markdown("""
        These filters help you select the best frames from your video based on various criteria:

        1. **Sharpness**: Higher values mean sharper images. Increase this to keep only the sharpest frames.

        2. **Motion**: Lower values mean less motion between frames. Decrease this to capture more stable scenes.

        3. **Contrast**: Adjust the range to select frames with desired contrast levels.

        4. **Exposure**: Set the range to capture frames with proper exposure (not too dark or bright).

        5. **Feature Density**: Higher values mean more detailed frames. Adjust to preference.

        6. **Feature Matches**: Higher values indicate more similarity between consecutive frames.

        7. **Camera Motion**: Lower values mean less camera movement. Adjust based on desired stability.

        Tip: Start with default settings and adjust gradually. You can always refine your selection after the initial processing.
        """)

    new_settings: FilterSettings = {}
    
    new_settings['sharpness'] = (
        st.slider("Sharpness Threshold (Higher is sharper)", 
                  min_value=0.0, 
                  max_value=10000.0,
                  value=float(filter_settings['sharpness'][0]), 
                  step=1.0),
        float('inf')
    )
    
    new_settings['motion'] = (0.0, st.slider("Motion Threshold (Lower means less motion)", 
                                             min_value=0.0, 
                                             max_value=10000.0, 
                                             value=float(filter_settings['motion'][1]), 
                                             step=0.1))
  
    
    new_settings['contrast'] = tuple(st.slider("Contrast Range", 
                                               min_value=0, 
                                               max_value=255, 
                                               value=(int(filter_settings['contrast'][0]), int(filter_settings['contrast'][1]))))
    
    new_settings['exposure'] = tuple(st.slider("Exposure Range (average pixel value)", 
                                               min_value=0, 
                                               max_value=10000, 
                                               value=(int(filter_settings['exposure'][0]), int(filter_settings['exposure'][1]))))
    
    new_settings['feature_density'] = tuple(st.slider("Feature Density Range (features per frame)", 
                                                      min_value=0, 
                                                      max_value=10000, 
                                                      value=(int(filter_settings['feature_density'][0]), int(filter_settings['feature_density'][1]))))
    
    new_settings['feature_matches'] = tuple(st.slider("Feature Matches Range (matches between consecutive frames)", 
                                                      min_value=0, 
                                                      max_value=5000, 
                                                      value=(int(filter_settings['feature_matches'][0]), int(filter_settings['feature_matches'][1]))))
    
    new_settings['camera_motion'] = tuple(st.slider("Camera Motion Range (average pixel displacement)", 
                                                    min_value=0.0, 
                                                    max_value=10.0, 
                                                    value=(float(filter_settings['camera_motion'][0]), float(filter_settings['camera_motion'][1])), 
                                                    step=0.1))
    
    return new_settings

def display_results() -> None:
    saved_frames = st.session_state.saved_frames
    stats = st.session_state.filtered_stats
    output_folder = st.session_state.output_folder
    filter_settings = st.session_state.filter_settings

    if len(saved_frames) == 0:
        st.warning("No frames met the specified criteria. Try adjusting the thresholds.")
    else:
        st.success(f"Extracted {len(saved_frames)} frames that meet the criteria.")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Statistics and Management", "Visualizations", "Extracted Frames", "Filter Analysis"])
    
    with tab1:
        display_statistics_and_management(stats, output_folder)
    
    with tab2:
        display_visualizations(stats, saved_frames, filter_settings)
    
    with tab3:
        display_extracted_frames(saved_frames)
    
    with tab4:
        display_filter_analysis(stats, filter_settings)
    




def display_visualizations(stats: Stats, saved_frames: List[str], filter_settings: FilterSettings) -> None:
    """Display visualizations of processing results."""
    st.write("### Visualizations and Summary Statistics")
    logging.debug(f"Stats keys before visualization: {stats.keys()}")
    logging.debug(f"Sharpness data in stats: {stats.get('sharpness', 'Not found')}")
    if "processing_parameters" in stats and any(stats[key]["Avg"] is not None for key in stats if key != "processing_parameters"):
        try:
            fig_timeline, fig1, fig2, summary_stats, filter_comparison, filter_suggestions = create_visualizations(stats, saved_frames, filter_settings)
            
            st.plotly_chart(fig_timeline, use_container_width=True)
            st.plotly_chart(fig1, use_container_width=True)
            st.plotly_chart(fig2, use_container_width=True)
            
            st.write("### Summary Statistics for Metrics")
            for metric, values in summary_stats.items():
                st.write(f"**{metric.capitalize()}**")
                for stat, value in values.items():
                    st.write(f"- {stat}: {value:.2f}")
                st.write("")
            
        except Exception as e:
            logging.error(f"An error occurred while creating visualizations: {str(e)}")
            st.error("An error occurred while creating visualizations. Please check the logs for more details.")
    else:
        st.info("No data available for visualization. This may be because no frames met the criteria.")
        
def display_extracted_frames(saved_frames: List[str]) -> None:
    """Display extracted frames in a grid layout."""
    if saved_frames:
        st.write("### Extracted Frames")
        cols = st.columns(4)
        for i, frame_path in enumerate(saved_frames):
            with cols[i % 4]:
                st.image(frame_path, caption=os.path.basename(frame_path), use_column_width=True)
    else:
        st.info("No frames to display. Try adjusting the thresholds.")

def display_filter_analysis(stats: Stats, filter_settings: FilterSettings) -> None:
    """Display filter analysis and adjustment suggestions."""
    st.write("### Filter Analysis")
    filter_comparison = create_filter_comparison(stats, filter_settings)
    for metric, data in filter_comparison.items():
        st.write(f"**{metric.capitalize()}**")
        
        # Display the current filter range
        if metric in ['sharpness', 'motion', 'camera_motion']:
            st.write(f"Current range: [{filter_settings[metric][0]:.2f}, {filter_settings[metric][1]:.2f}]")
        else:
            st.write(f"Current range: [{filter_settings[metric][0]}, {filter_settings[metric][1]}]")
        
        st.write(f"- Frames within range: {data['frames_within_range']} out of {data['total_frames']} ({data['percentage_within_range']:.2f}%)")
        st.write(f"- Closest value below range: {data['closest_below']:.2f}" if data['closest_below'] is not None else "- No values below range")
        st.write(f"- Closest value above range: {data['closest_above']:.2f}" if data['closest_above'] is not None else "- No values above range")
        st.write("")

    st.write("### Filter Adjustment Suggestions")
    filter_suggestions = suggest_filter_adjustments(filter_comparison, filter_settings)
    if filter_suggestions:
        for metric, suggestion in filter_suggestions.items():
            st.write(f"**{metric.capitalize()}**")
            st.write(f"- Current range: [{suggestion['current_min']:.2f}, {suggestion['current_max']:.2f}]")
            st.write(f"- Suggested range: [{suggestion['suggested_min']:.2f}, {suggestion['suggested_max']:.2f}]")
            st.write("")
        
        if st.button("Apply Suggested Settings"):
            new_settings = apply_suggested_settings(filter_settings, filter_suggestions)
            st.session_state.filter_settings = new_settings
            st.success("Suggested settings applied. Please run the processing again with the new settings.")
            st.rerun()
    else:
        st.write("No filter adjustments suggested. Current settings appear to be optimal.")

def process_video(video_path: str) -> None:
    """Process the uploaded video and update session state with results."""
    output_folder = Path(OUTPUT_DIR) / str(uuid.uuid4())
    
    with st.spinner("Processing video..."):
        progress_bar = st.progress(0)
        try:
            all_frame_data, stats = process_video_stream(
                video_path=video_path,
                output_folder=str(output_folder),
                cancel_flag=lambda: st.session_state.cancel_processing,
                progress_callback=lambda progress: progress_bar.progress(progress)
            )
            if all_frame_data is None or stats is None:
                st.error("Video processing failed. Please check the logs for more information.")
                return
            
            st.session_state.all_frame_data = all_frame_data
            st.session_state.stats = stats
            st.session_state.output_folder = output_folder
            st.session_state.original_video_path = video_path
            
            # Apply initial filter
            saved_frames, filtered_stats = apply_filters(all_frame_data, st.session_state.filter_settings, video_path, str(output_folder))
            st.session_state.saved_frames = saved_frames
            st.session_state.filtered_stats = filtered_stats
            
            st.success("Video processing completed!")
        except Exception as e:
            st.error(f"An error occurred during video processing: {str(e)}")
            logging.error(f"Video processing error: {str(e)}")
            logging.error(traceback.format_exc())
            return
    st.rerun()

def apply_filters(all_frame_data: pd.DataFrame, filter_settings: Dict[str, Tuple[float, float]], video_path: str, output_folder: str) -> Tuple[List[str], Dict[str, Any]]:
    # Apply filters
    mask = pd.Series(True, index=all_frame_data.index)
    for metric, (min_val, max_val) in filter_settings.items():
        if metric in all_frame_data.columns:
            if np.isinf(max_val):  # Handle infinity for upper bound
                mask &= (all_frame_data[metric] >= min_val)
            else:
                mask &= (all_frame_data[metric] >= min_val) & (all_frame_data[metric] <= max_val)
    
    filtered_data = all_frame_data[mask]
    selected_frames = filtered_data['frame_number'].tolist()
    
    # Extract and save selected frames
    saved_frames = extract_and_save_frames(video_path, selected_frames, output_folder)
    
    # Create filtered stats
    filtered_stats = {}
    for metric in all_frame_data.columns:
        if metric != 'frame_number':
            filtered_stats[metric] = {
                'Max': filtered_data[metric].max(),
                'Min': filtered_data[metric].min(),
                'Avg': filtered_data[metric].mean(),
                'Values': filtered_data[metric].tolist()
            }
    
    filtered_stats['processing_parameters'] = {
        'filter_settings': filter_settings,
        'total_frames_processed': len(all_frame_data),
        'frames_extracted': len(saved_frames)
    }
    
    return saved_frames, filtered_stats

def display_statistics_and_management(stats: Stats, output_folder: Path) -> None:
    """Display processing statistics and file management options."""
    st.write("### Processing Statistics")
    st.write("Full statistics have been saved in stats.json in the output folder.")
    st.write(f"Output folder: {output_folder}")
    
    st.write("Summary of key statistics:")
    for stat_name, stat_values in stats.items():
        if stat_name != "processing_parameters":
            if stat_values["Avg"] is not None:
                st.write(f"- {stat_name.capitalize()}: Avg = {stat_values['Avg']:.2f}")
            else:
                st.write(f"- {stat_name.capitalize()}: No data available")
    
    st.write("### Processing Parameters")
    for param, value in stats["processing_parameters"].items():
        st.write(f"- {param.replace('_', ' ').capitalize()}: {value}")
    
    zip_buffer = create_download_zip(output_folder)
    st.download_button(
        label="Download Results",
        data=zip_buffer,
        file_name="video_analysis_results.zip",
        mime="application/zip",
        key=f"download_button_{output_folder}"  # Add a unique key based on the output folder
    )

def main() -> None:
    st.title("Video Frame Extraction and Analysis")

    # Initialize session state
    init_session_state()

    video_path = render_ui()

    # Render filter settings and get the new settings
    new_filter_settings = render_filter_settings(st.session_state.filter_settings)

    # Check if there are unapplied changes
    unapplied_changes = new_filter_settings != st.session_state.filter_settings
    if unapplied_changes:
        st.warning("You have unapplied filter changes. Click 'Run' to apply them.")

    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("Run"):
            if video_path:
                if os.path.getsize(video_path) <= MAX_UPLOAD_SIZE:
                    # Apply the new filter settings
                    st.session_state.filter_settings = new_filter_settings
                    if 'all_frame_data' not in st.session_state or st.session_state.original_video_path != video_path:
                        # New video uploaded or first run, process it fully
                        process_video(video_path)
                    else:
                        # Apply filters to existing data
                        saved_frames, filtered_stats = apply_filters(
                            st.session_state.all_frame_data,
                            st.session_state.filter_settings,
                            st.session_state.original_video_path,
                            str(st.session_state.output_folder)
                        )
                        st.session_state.saved_frames = saved_frames
                        st.session_state.filtered_stats = filtered_stats
                        st.success("Filter application completed!")
                    # Force a rerun to update the visualizations
                    st.rerun()
                else:
                    st.error(f"The uploaded file exceeds the maximum allowed size of {MAX_UPLOAD_SIZE / (1024 * 1024)} MB. Please upload a smaller file.")
            else:
                st.error("Please upload a video file before processing.")
    with col2:
        if st.button("Cancel"):
            st.session_state.cancel_processing = True
            st.warning("Canceling processing...")
            st.rerun()

    with col3:
        if st.button("Clear Results"):
            clear_files_and_reset(st)
            st.rerun()

    with col4:
        if st.button("Reset Settings"):
            st.session_state.filter_settings = {
                'sharpness': (350.0, float('inf')),
                'motion': (0.0, 200.0),
                'contrast': (100, 255),
                'exposure': (100, 7000),
                'feature_density': (500, 6000),
                'feature_matches': (0, 4000),
                'camera_motion': (0.0, 5.0)
            }
            st.success("Filter settings have been reset to default values.")
            st.rerun()

    # Display results only if we have processed data
    if 'filtered_stats' in st.session_state and st.session_state.filtered_stats is not None:
        display_results()
    else:
        st.info("No results to display. Please run the processing first.")

    # Reset new_upload state
    if 'new_upload' in st.session_state:
        st.session_state.new_upload = False

if __name__ == "__main__":
    main()