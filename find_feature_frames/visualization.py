import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def create_visualizations(stats):
    # Convert stats to a format suitable for visualization
    df = pd.DataFrame({key: values for key, values in stats.items() if key != "processing_parameters"})
    
    # Line plot of all metrics over frames
    fig1 = px.line(df, title="Metrics Over Frames")
    
    # Box plot of all metrics
    fig2 = px.box(df, title="Distribution of Metrics")
    
    # Correlation heatmap
    corr = df.corr()
    fig3 = go.Figure(data=go.Heatmap(z=corr.values, x=corr.index, y=corr.columns))
    fig3.update_layout(title="Correlation Heatmap of Metrics")
    
    # Histogram of sharpness and motion
    fig4 = go.Figure()
    fig4.add_trace(go.Histogram(x=df['sharpness'], name='Sharpness'))
    fig4.add_trace(go.Histogram(x=df['motion'], name='Motion'))
    fig4.update_layout(title="Distribution of Sharpness and Motion", barmode='overlay')
    
    return fig1, fig2, fig3, fig4
