import os
from typing import List, Dict, Tuple, Any, Union
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import numpy as np
import logging
import traceback

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
        fig.add_trace(go.Scatter(x=df['frame_number'], y=df[metric], mode='lines', name=metric), row=i, col=1)
        
        selected_y = df[df['frame_number'].isin(selected_frames)][metric]
        fig.add_trace(go.Scatter(
            x=selected_frames,
            y=selected_y,
            mode='markers',
            marker=dict(size=10, symbol='star', color='red'),
            name=f'Selected Frames ({metric})'
        ), row=i, col=1)
        
        fig.update_yaxes(title_text=metric, row=i, col=1)

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
        if column != 'frame_number':
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

def create_filter_comparison(stats: Dict[str, Dict[str, Any]], filter_settings: Dict[str, Tuple[float, float]]) -> Dict[str, Dict[str, Any]]:
    """
    Create a comparison of filter settings against the data.
    
    Args:
        stats (Dict[str, Dict[str, Any]]): Dictionary of statistics for each metric.
        filter_settings (Dict[str, Tuple[float, float]]): Dictionary of filter settings for each metric.
    
    Returns:
        Dict[str, Dict[str, Any]]: Dictionary of filter comparison results for each metric.
    """
    comparison = {}
    for metric, (min_val, max_val) in filter_settings.items():
        if metric in stats and "Values" in stats[metric]:
            values = stats[metric]["Values"]
            within_range = [v for v in values if min_val <= v <= max_val]
            comparison[metric] = {
                'total_frames': len(values),
                'frames_within_range': len(within_range),
                'percentage_within_range': (len(within_range) / len(values)) * 100 if values else 0,
                'closest_below': max([v for v in values if v < min_val], default=None),
                'closest_above': min([v for v in values if v > max_val], default=None),
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

def create_visualizations(stats: Dict[str, Union[List[float], Dict[str, Any]]], saved_frames: List[str], filter_settings: Dict[str, Tuple[float, float]]) -> Tuple[go.Figure, go.Figure, go.Figure, Dict[str, Dict[str, float]], Dict[str, Dict[str, Any]], Dict[str, Dict[str, float]]]:
    try:
        logging.debug(f"Stats keys in create_visualizations: {stats.keys()}")
        logging.debug(f"Sharpness data in stats: {stats.get('sharpness', 'Not found')}")
       
        # Create a DataFrame from the stats
        metrics = ['sharpness', 'motion', 'contrast', 'exposure', 'feature_density', 'feature_matches', 'camera_motion']
        
        df = pd.DataFrame()
        for metric in metrics:
            if metric in stats and isinstance(stats[metric], dict) and 'Values' in stats[metric]:
                df[metric] = stats[metric]['Values']
            elif metric in stats and isinstance(stats[metric], list):
                df[metric] = stats[metric]
            else:
                logging.warning(f"Metric '{metric}' not found in stats or has unexpected format")
        
        logging.debug(f"DataFrame columns: {df.columns}")
        logging.debug(f"DataFrame head: {df.head()}")
        
        # Ensure 'frame_number' column exists
        if 'frame_number' not in df.columns:
            df['frame_number'] = range(1, len(df) + 1)
        
        # Check if 'sharpness' is in the DataFrame
        if 'sharpness' in df.columns:
            logging.debug(f"Sharpness values: {df['sharpness'].tolist()[:10]}...")  # Show first 10 values
        else:
            logging.warning("'sharpness' column is missing from the DataFrame")

        # Extract frame numbers from saved_frames
        selected_frames = extract_frame_numbers(saved_frames)

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
        logging.error(f"Error details: {traceback.format_exc()}")
        raise