import os
from typing import List, Dict, Tuple, Any
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_frame_numbers(saved_frames: List[str]) -> List[int]:
    """
    Extract frame numbers from saved frame filenames.
    
    Args:
        saved_frames (List[str]): List of saved frame file paths.
    
    Returns:
        List[int]: List of extracted frame numbers.
    """
    frame_numbers = []
    for frame in saved_frames:
        match = re.search(r'frame_(\d+)', os.path.basename(frame))
        if match:
            frame_numbers.append(int(match.group(1)))
        else:
            match = re.search(r'(\d+)', os.path.basename(frame))
            if match:
                frame_numbers.append(int(match.group(1)))
    return frame_numbers

def create_timeline_graph(df: pd.DataFrame, selected_frames: List[int], metric: str) -> go.Figure:
    """
    Create a timeline graph for a single metric.
    
    Args:
        df (pd.DataFrame): DataFrame containing the metric data.
        selected_frames (List[int]): List of frame numbers for selected frames.
        metric (str): Name of the metric to plot.
    
    Returns:
        go.Figure: Plotly figure object for the timeline graph.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['frame'], y=df[metric], mode='lines', name=metric))
    
    selected_y = df[df['frame'].isin(selected_frames)][metric]
    fig.add_trace(go.Scatter(
        x=selected_frames,
        y=selected_y,
        mode='markers',
        marker=dict(size=10, symbol='star', color='red'),
        name=f'Selected Frames ({metric})'
    ))
    
    fig.update_layout(
        title=f"{metric.capitalize()} Over Video Timeline",
        xaxis_title="Frame Number",
        yaxis_title=metric.capitalize()
    )
    
    return fig

def create_timeline_graphs(df: pd.DataFrame, selected_frames: List[int], metrics: List[str]) -> go.Figure:
    """
    Create timeline graphs for multiple metrics.
    
    Args:
        df (pd.DataFrame): DataFrame containing the metrics data.
        selected_frames (List[int]): List of frame numbers for selected frames.
        metrics (List[str]): List of metric names to plot.
    
    Returns:
        go.Figure: Plotly figure object containing subplots for each metric.
    """
    fig = make_subplots(rows=len(metrics), cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        subplot_titles=metrics)

    for i, metric in enumerate(metrics, start=1):
        metric_fig = create_timeline_graph(df, selected_frames, metric)
        for trace in metric_fig.data:
            fig.add_trace(trace, row=i, col=1)
        
        fig.update_yaxes(title_text=metric.capitalize(), row=i, col=1)

    fig.update_layout(height=300*len(metrics), title_text="Key Metrics Over Video Timeline", showlegend=False)
    fig.update_xaxes(title_text="Frame Number", row=len(metrics), col=1)

    return fig

def create_summary_stats(df: pd.DataFrame, metrics: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Create summary statistics for given metrics.
    
    Args:
        df (pd.DataFrame): DataFrame containing the metrics data.
        metrics (List[str]): List of metric names to summarize.
    
    Returns:
        Dict[str, Dict[str, float]]: Dictionary of summary statistics for each metric.
    """
    summary = {}
    for metric in metrics:
        summary[metric] = {
            'Max': df[metric].max(),
            'Min': df[metric].min(),
            'Average': df[metric].mean(),
            'Median': df[metric].median()
        }
    return summary

def get_slider_ranges(df: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    """
    Calculate appropriate slider ranges for each metric.
    
    Args:
        df (pd.DataFrame): DataFrame containing the metrics data.
    
    Returns:
        Dict[str, Tuple[float, float]]: Dictionary of slider ranges for each metric.
    """
    ranges = {}
    for column in df.columns:
        if column != 'frame':
            min_val = df[column].min()
            max_val = df[column].max()
            range_val = max_val - min_val
            
            if range_val > 1000:
                # Use logarithmic scale for large ranges
                ranges[column] = (np.log10(max(1, min_val)), np.log10(max_val))
            else:
                # Add some padding to the range
                padding = range_val * 0.1
                ranges[column] = (max(0, min_val - padding), max_val + padding)
    
    return ranges

def create_filter_comparison(df: pd.DataFrame, filter_settings: Dict[str, Tuple[float, float]]) -> Dict[str, Dict[str, Any]]:
    """
    Create a comparison of filter settings against the data.
    
    Args:
        df (pd.DataFrame): DataFrame containing the metrics data.
        filter_settings (Dict[str, Tuple[float, float]]): Dictionary of filter settings for each metric.
    
    Returns:
        Dict[str, Dict[str, Any]]: Dictionary of filter comparison results for each metric.
    """
    comparison = {}
    for metric, (min_val, max_val) in filter_settings.items():
        if metric in df.columns:
            within_range = df[(df[metric] >= min_val) & (df[metric] <= max_val)]
            comparison[metric] = {
                'total_frames': len(df),
                'frames_within_range': len(within_range),
                'percentage_within_range': (len(within_range) / len(df)) * 100,
                'closest_below': df[df[metric] < min_val][metric].max() if len(df[df[metric] < min_val]) > 0 else None,
                'closest_above': df[df[metric] > max_val][metric].min() if len(df[df[metric] > max_val]) > 0 else None,
            }
    return comparison

def suggest_filter_adjustments(comparison: Dict[str, Dict[str, Any]], current_settings: Dict[str, Tuple[float, float]]) -> Dict[str, Dict[str, float]]:
    """
    Suggest adjustments to filter settings based on the comparison results.
    
    Args:
        comparison (Dict[str, Dict[str, Any]]): Dictionary of filter comparison results.
        current_settings (Dict[str, Tuple[float, float]]): Dictionary of current filter settings.
    
    Returns:
        Dict[str, Dict[str, float]]: Dictionary of suggested filter adjustments.
    """
    suggestions = {}
    for metric, data in comparison.items():
        if data['percentage_within_range'] < 1:  # Less than 1% of frames within range
            if data['closest_below'] is not None:
                suggestions[metric] = {
                    'current_min': current_settings[metric][0],
                    'suggested_min': max(data['closest_below'], current_settings[metric][0] * 0.9),
                    'current_max': current_settings[metric][1],
                    'suggested_max': current_settings[metric][1],
                }
            elif data['closest_above'] is not None:
                suggestions[metric] = {
                    'current_min': current_settings[metric][0],
                    'suggested_min': current_settings[metric][0],
                    'current_max': current_settings[metric][1],
                    'suggested_max': min(data['closest_above'], current_settings[metric][1] * 1.1),
                }
    return suggestions

def create_visualizations(stats: Dict[str, Any], saved_frames: List[str], filter_settings: Dict[str, Tuple[float, float]]) -> Tuple[go.Figure, go.Figure, go.Figure, Dict[str, Dict[str, float]], Dict[str, Dict[str, Any]], Dict[str, Dict[str, float]]]:
    """
    Create visualizations and analysis based on the video processing results.
    
    Args:
        stats (Dict[str, Any]): Dictionary of video processing statistics.
        saved_frames (List[str]): List of saved frame file paths.
        filter_settings (Dict[str, Tuple[float, float]]): Dictionary of filter settings for each metric.
    
    Returns:
        Tuple[go.Figure, go.Figure, go.Figure, Dict[str, Dict[str, float]], Dict[str, Dict[str, Any]], Dict[str, Dict[str, float]]]:
            - Timeline graph figure
            - Distribution plot figure 1
            - Distribution plot figure 2
            - Summary statistics
            - Filter comparison results
            - Filter adjustment suggestions
    """
    try:
        # Create a DataFrame from the stats
        df = pd.DataFrame({key: value for key, value in stats.items() if key != "processing_parameters"})
        df['frame'] = df.index

        # Extract frame numbers from saved_frames
        selected_frames = extract_frame_numbers(saved_frames)

        # List of metrics to visualize
        metrics = ['sharpness', 'motion', 'contrast', 'exposure', 'feature_density', 'feature_matches', 'camera_motion']

        # Create timeline graphs
        fig_timeline = create_timeline_graphs(df, selected_frames, metrics)

        # Create distribution plots
        fig1 = make_subplots(rows=2, cols=2, subplot_titles=metrics[:4])
        fig2 = make_subplots(rows=2, cols=2, subplot_titles=metrics[4:] + [''])

        for i, metric in enumerate(metrics[:4], start=1):
            fig1.add_trace(go.Histogram(x=df[metric], name=metric), row=(i-1)//2+1, col=(i-1)%2+1)
            if metric in filter_settings:
                min_val, max_val = filter_settings[metric]
                fig1.add_vline(x=min_val, line_dash="dash", line_color="red", row=(i-1)//2+1, col=(i-1)%2+1)
                fig1.add_vline(x=max_val, line_dash="dash", line_color="red", row=(i-1)//2+1, col=(i-1)%2+1)

        for i, metric in enumerate(metrics[4:], start=1):
            fig2.add_trace(go.Histogram(x=df[metric], name=metric), row=(i-1)//2+1, col=(i-1)%2+1)
            if metric in filter_settings:
                min_val, max_val = filter_settings[metric]
                fig2.add_vline(x=min_val, line_dash="dash", line_color="red", row=(i-1)//2+1, col=(i-1)%2+1)
                fig2.add_vline(x=max_val, line_dash="dash", line_color="red", row=(i-1)//2+1, col=(i-1)%2+1)

        fig1.update_layout(height=600, title_text="Distribution of Metrics (Part 1)")
        fig2.update_layout(height=600, title_text="Distribution of Metrics (Part 2)")

        # Create summary statistics
        summary_stats = create_summary_stats(df, metrics)

        # Create filter comparison
        filter_comparison = create_filter_comparison(df, filter_settings)

        # Suggest filter adjustments
        filter_suggestions = suggest_filter_adjustments(filter_comparison, filter_settings)

        return fig_timeline, fig1, fig2, summary_stats, filter_comparison, filter_suggestions
    
    except Exception as e:
        logging.error(f"Error in create_visualizations: {str(e)}")
        raise