import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import numpy as np

def extract_frame_numbers(saved_frames):
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

def create_timeline_graphs(df, selected_frames, metrics):
    fig = make_subplots(rows=len(metrics), cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        subplot_titles=metrics)

    for i, metric in enumerate(metrics, start=1):
        fig.add_trace(go.Scatter(x=df['frame'], y=df[metric], mode='lines', name=metric), row=i, col=1)
        
        selected_y = df[df['frame'].isin(selected_frames)][metric]
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

def create_summary_stats(df, metrics):
    summary = {}
    for metric in metrics:
        summary[metric] = {
            'Max': df[metric].max(),
            'Min': df[metric].min(),
            'Average': df[metric].mean(),
            'Median': df[metric].median()
        }
    return summary

def get_slider_ranges(df):
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

def create_filter_comparison(df, filter_settings):
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

def suggest_filter_adjustments(comparison, current_settings):
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

def create_visualizations(stats, saved_frames, filter_settings):
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