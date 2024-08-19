import streamlit as st
import os
import uuid
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional
from utils import MAX_UPLOAD_SIZE, OUTPUT_DIR, create_zip, get_video_stats, handle_upload, clear_files_and_reset, create_download_zip
from video_processing import process_video_stream
from visualization import create_visualizations, suggest_filter_adjustments, create_filter_comparison
import logging

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
    if 'new_upload' not in st.session_state:
        st.session_state.new_upload = False
    if 'filter_settings' not in st.session_state:
        st.session_state.filter_settings = {
            'sharpness': (350.0, 1000.0),
            'motion': (0.0, 200.0),
            'contrast': (100, 255),
            'exposure': (100, 7000),
            'feature_density': (500, 6000),
            'feature_matches': (0, 4000),
            'camera_motion': (0.0, 5.0)
        }

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

def render_filter_settings(filter_settings):
    st.write("### Filter Settings")
    new_settings = {}
    
    new_settings['sharpness'] = (
        st.slider("Sharpness Threshold", 
                  min_value=0.0, 
                  max_value=1000.0,
                  value=float(filter_settings['sharpness'][0]), 
                  step=0.1),
        1000.0
    )
    
    new_settings['motion'] = (0.0, st.slider("Motion Threshold", 
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
    """Display processing results and visualizations."""
    if st.session_state.saved_frames is None or st.session_state.stats is None:
        st.info("No results to display. Please run the processing first.")
        return

    saved_frames = st.session_state.saved_frames
    stats = st.session_state.stats
    output_folder = st.session_state.output_folder
    filter_settings = st.session_state.filter_settings

    if len(saved_frames) == 0:
        st.warning("No frames met the specified criteria. Try adjusting the thresholds.")
    else:
        st.success(f"Extracted {len(saved_frames)} frames that meet the criteria.")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Statistics and Management", "Visualizations", "Extracted Frames", "Filter Analysis"])
    
    with tab1:
        if output_folder:
            display_statistics_and_management(stats, output_folder)
        else:
            st.error("Output folder not found. Please run the processing first.")
    
    with tab2:
        display_visualizations(stats, saved_frames, filter_settings)
    
    with tab3:
        display_extracted_frames(saved_frames)
    
    with tab4:
        display_filter_analysis(stats, filter_settings)

def display_statistics_and_management(stats: Stats, output_folder: Optional[str]) -> None:
    """Display processing statistics and file management options."""
    st.write("### Processing Statistics")
    st.write("Full statistics have been saved in stats.json in the output folder.")
    
    if output_folder:
        st.write(f"Output folder: {output_folder}")
        
        st.write("Summary of key statistics:")
        for stat_name, stat_values in stats.items():
            if stat_name != "processing_parameters":
                if "Avg" in stat_values and stat_values["Avg"] is not None:
                    st.write(f"- {stat_name.capitalize()}: Avg = {stat_values['Avg']:.2f}")
                else:
                    st.write(f"- {stat_name.capitalize()}: No data available")
        
        if "processing_parameters" in stats:
            st.write("### Processing Parameters")
            for param, value in stats["processing_parameters"].items():
                st.write(f"- {param.replace('_', ' ').capitalize()}: {value}")
        
        try:
            zip_buffer = create_download_zip(output_folder)
            st.download_button(
                label="Download Results",
                data=zip_buffer,
                file_name="video_analysis_results.zip",
                mime="application/zip"
            )
        except FileNotFoundError:
            st.error("Output folder not found. The results may have been deleted or moved.")
        except Exception as e:
            logging.error(f"Error creating download zip: {str(e)}")
            st.error("An error occurred while preparing the download. Please try again later.")
    else:
        st.error("Output folder not specified. Results may not be available.")


def display_visualizations(stats: Stats, saved_frames: List[str], filter_settings: FilterSettings) -> None:
    """Display visualizations of processing results."""
    st.write("### Visualizations and Summary Statistics")
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
    filter_comparison = create_filter_comparison(pd.DataFrame(stats), filter_settings)
    for metric, data in filter_comparison.items():
        st.write(f"**{metric.capitalize()}**")
        
        # Display the current filter range
        if metric == 'sharpness':
            st.write(f"Current range: [{filter_settings[metric][0]:.2f}, ∞)")
        elif metric == 'motion':
            st.write(f"Current range: [0, {filter_settings[metric][1]:.2f}]")
        else:
            st.write(f"Current range: [{filter_settings[metric][0]:.2f}, {filter_settings[metric][1]:.2f}]")
        
        st.write(f"- Frames within range: {data['frames_within_range']} out of {data['total_frames']} ({data['percentage_within_range']:.2f}%)")
        st.write(f"- Closest value below range: {data['closest_below']:.2f}" if data['closest_below'] is not None else "- No values below range")
        st.write(f"- Closest value above range: {data['closest_above']:.2f}" if data['closest_above'] is not None else "- No values above range")
        st.write("")

    st.write("### Filter Adjustment Suggestions")
    filter_suggestions = suggest_filter_adjustments(filter_comparison, filter_settings)
    if filter_suggestions:
        for metric, suggestion in filter_suggestions.items():
            st.write(f"**{metric.capitalize()}**")
            if metric == 'sharpness':
                st.write(f"- Current range: [{suggestion['current_min']:.2f}, ∞)")
            elif metric == 'motion':
                st.write(f"- Current range: [0, {suggestion['current_max']:.2f}]")
            else:
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
    output_folder = os.path.join(OUTPUT_DIR, str(uuid.uuid4()))
    
    with st.spinner("Processing video..."):
        progress_bar = st.progress(0)
        try:
            saved_frames, stats = process_video_stream(
                video_path, output_folder, 
                st.session_state.filter_settings['sharpness'][0],
                st.session_state.filter_settings['motion'][1],
                st.session_state.filter_settings['contrast'],
                st.session_state.filter_settings['exposure'],
                st.session_state.filter_settings['feature_density'],
                st.session_state.filter_settings['feature_matches'],
                st.session_state.filter_settings['camera_motion'],
                lambda: st.session_state.cancel_processing,
                lambda progress: progress_bar.progress(progress)
            )
            st.session_state.saved_frames = saved_frames
            st.session_state.stats = stats
            st.session_state.output_folder = output_folder
            st.success("Video processing completed!")
        except Exception as e:
            logging.error(f"Error during video processing: {str(e)}")
            st.error("An error occurred during video processing. Please try again.")
    st.rerun()


def main() -> None:
    """Main function to run the Streamlit app."""
    st.title("Video Frame Extraction and Analysis")

    # Initialize session state
    init_session_state()

    video_path = render_ui()

    # Render filter settings
    st.session_state.filter_settings = render_filter_settings(st.session_state.filter_settings)

    if st.button("Run"):
        logging.info("Run button clicked")
        if video_path:
            logging.info(f"Video path is {video_path}")
            if os.path.getsize(video_path) <= MAX_UPLOAD_SIZE:
                process_video(video_path)
            else:
                st.error(f"The uploaded file exceeds the maximum allowed size of {MAX_UPLOAD_SIZE / (1024 * 1024)} MB. Please upload a smaller file.")
        else:
            st.error("Please upload a video file before processing.")

    if st.button("Cancel"):
        st.session_state.cancel_processing = True
        st.warning("Canceling processing...")
        st.rerun()

    display_results()

    # Reset new_upload state
    if 'new_upload' in st.session_state:
        st.session_state.new_upload = False

if __name__ == "__main__":
    main()