import streamlit as st
import os
import uuid
from utils import MAX_UPLOAD_SIZE, OUTPUT_DIR, create_zip, get_video_stats, handle_upload, clear_files_and_reset
from video_processing import process_video_stream
from visualization import create_visualizations

# Initialize session state variables
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

def display_results():
    if st.session_state.saved_frames is None or st.session_state.stats is None:
        st.info("No results to display. Please run the processing first.")
        return

    saved_frames = st.session_state.saved_frames
    stats = st.session_state.stats
    output_folder = st.session_state.output_folder

    if len(saved_frames) == 0:
        st.warning("No frames met the specified criteria. Try adjusting the thresholds.")
    else:
        st.success(f"Extracted {len(saved_frames)} frames that meet the criteria.")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Statistics and Management", "Visualizations", "Extracted Frames"])
    
    with tab1:
        st.write("### Processing Statistics")
        st.write("Full statistics have been saved in stats.json in the output folder.")
        st.write(f"Output folder: {output_folder}")
        
        # Display a summary of the stats
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
        
        # ... [rest of the tab1 content remains the same] ...
    
    with tab2:
        st.write("### Visualizations")
        if "processing_parameters" in stats and any(stats[key]["Avg"] is not None for key in stats if key != "processing_parameters"):
            fig1, fig2, fig3, fig4 = create_visualizations(stats)
            st.plotly_chart(fig1)
            st.plotly_chart(fig2)
            st.plotly_chart(fig3)
            st.plotly_chart(fig4)
        else:
            st.info("No data available for visualization. This may be because no frames met the criteria.")
    
    with tab3:
        if saved_frames:
            st.write("### Extracted Frames")
            cols = st.columns(4)
            for i, frame_path in enumerate(saved_frames):
                with cols[i % 4]:
                    st.image(frame_path, caption=os.path.basename(frame_path), use_column_width=True)
        else:
            st.info("No frames to display. Try adjusting the thresholds.")
def render_ui():
    st.write("Upload a video, and we will extract the best frames based on sharpness and motion criteria.")
    st.write(f"Maximum file size for upload: {MAX_UPLOAD_SIZE / (1024 * 1024)} MB")

    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])
    
    if uploaded_file is not None and st.session_state.video_path is None:
        st.session_state.cancel_processing = False
        video_path = handle_upload(uploaded_file)
        st.session_state.video_path = video_path
        st.rerun()
    
    if st.session_state.video_path:
        display_video_stats(st.session_state.video_path)
        return st.session_state.video_path
    return None

def display_video_stats(video_path):
    video_stats = get_video_stats(video_path)
    st.write("### Video Statistics")
    st.write(video_stats)

def main():
    st.title("Video Frame Extraction and Analysis")

    video_path = render_ui()

    sharpness_threshold = st.slider("Sharpness Threshold", min_value=0.0, max_value=1000.0, value=150.0)
    motion_threshold = st.slider("Motion Threshold", min_value=0.0, max_value=10000.0, value=200.0)

    if st.button("Run"):
        if video_path and os.path.getsize(video_path) <= MAX_UPLOAD_SIZE:
            output_folder = os.path.join(OUTPUT_DIR, str(uuid.uuid4()))
            
            with st.spinner("Processing video..."):
                progress_bar = st.progress(0)
                saved_frames, stats = process_video_stream(
                    video_path, output_folder, sharpness_threshold, motion_threshold, 
                    lambda: st.session_state.cancel_processing,
                    lambda progress: progress_bar.progress(progress)
                )
                st.session_state.saved_frames = saved_frames
                st.session_state.stats = stats
                st.session_state.output_folder = output_folder
            st.rerun()
        elif video_path:
            st.error(f"The uploaded file exceeds the maximum allowed size of {MAX_UPLOAD_SIZE / (1024 * 1024)} MB. Please upload a smaller file.")
        else:
            st.error("Please upload a video file before processing.")

    if st.button("Cancel"):
        st.session_state.cancel_processing = True
        st.warning("Canceling processing...")
        st.rerun()

    display_results()

if __name__ == "__main__":
    main()