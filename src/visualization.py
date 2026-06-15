import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# Premium Color Palette
COLORS = {
    'ops': '#0284c7',        # Ocean Blue
    'cranes': '#f97316',     # Safety Orange
    'reefers': '#10b981',    # Emerald (Cooling)
    'base': '#64748b',      # Slate Gray
    'peaks': '#ef4444',      # Red Alert
    'solar': '#eab308',      # Sun Yellow
    'wind': '#06b6d4',       # Wind Cyan
    'battery': '#8b5cf6',    # Battery Violet
    'grid': '#3b82f6',       # Grid Blue
    'grid_opt': '#10b981',   # Optimized Grid Green
    'actual': '#0f172a',     # Dark slate
    'forecast': '#ec4899',   # Pink Forecast
}

# Dark theme layout adjustments for Plotly
DARK_TEMPLATE = dict(
    layout=go.Layout(
        plot_bgcolor='rgba(15, 23, 42, 0.05)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        font=dict(color='#f1f5f9', family='Inter, Roboto, sans-serif'),
        xaxis=dict(
            gridcolor='#334155',
            linecolor='#475569',
            zerolinecolor='#475569',
            showgrid=True,
        ),
        yaxis=dict(
            gridcolor='#334155',
            linecolor='#475569',
            zerolinecolor='#475569',
            showgrid=True,
        ),
        hoverlabel=dict(
            bgcolor='#1e293b',
            font_size=13,
            font_family='Inter, sans-serif'
        ),
        legend=dict(
            bgcolor='rgba(15, 23, 42, 0.6)',
            bordercolor='#334155',
            borderwidth=1
        )
    )
)

def apply_layout(fig, title, xaxis_title, yaxis_title, height=450):
    fig.update_layout(
        title={
            'text': title,
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 18, 'weight': 'bold'}
        },
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        height=height,
        margin=dict(l=40, r=40, t=60, b=40),
        template=DARK_TEMPLATE
    )
    return fig

def plot_demand_components(df):
    """
    Plots a stacked area chart showing the breakdown of simulated port electrical demand.
    """
    fig = go.Figure()
    
    components = [
        ('Base_Demand_kW', 'Base Infrastructure', COLORS['base']),
        ('Reefer_Demand_kW', 'Reefer Containers', COLORS['reefers']),
        ('Crane_Demand_kW', 'Cargo Handling Cranes', COLORS['cranes']),
        ('OPS_Demand_kW', 'Vessel Shore Power (OPS)', COLORS['ops']),
        ('Peak_Demand_kW', 'Peak Operations', COLORS['peaks'])
    ]
    
    for col, label, color in components:
        fig.add_trace(go.Scatter(
            x=df['Hour'],
            y=df[col],
            mode='lines',
            line=dict(width=0.5, color=color),
            stackgroup='one',
            name=label,
            hoverinfo='x+y+name'
        ))
        
    fig.add_trace(go.Scatter(
        x=df['Hour'],
        y=df['Total_Demand_kW'],
        mode='lines',
        line=dict(color='#ffffff', width=1.5, dash='dash'),
        name='Total Port Load',
        hoverinfo='x+y+name'
    ))
    
    return apply_layout(fig, "Digital Twin: Port Electricity Demand Breakdown", "Simulation Hour", "Power Demand (kW)")

def plot_renewables_generation(df):
    """
    Plots solar and wind generation curves with wind speed overlay.
    """
    fig = go.Figure()
    
    # Solar Generation Area
    fig.add_trace(go.Scatter(
        x=df['Hour'],
        y=df['Solar_Gen_kW'],
        mode='lines',
        line=dict(width=0, color=COLORS['solar']),
        fill='tozeroy',
        name='Solar PV Output (kW)',
        yaxis='y1'
    ))
    
    # Wind Generation Area
    fig.add_trace(go.Scatter(
        x=df['Hour'],
        y=df['Wind_Gen_kW'],
        mode='lines',
        line=dict(width=0, color=COLORS['wind']),
        fill='tozeroy',
        name='Wind Turbine Output (kW)',
        yaxis='y1'
    ))
    
    # Wind Speed line on secondary axis
    fig.add_trace(go.Scatter(
        x=df['Hour'],
        y=df['Wind_Speed_ms'],
        mode='lines',
        line=dict(color='#f43f5e', width=1.5, dash='dot'),
        name='Wind Speed (m/s)',
        yaxis='y2'
    ))
    
    fig.update_layout(
        yaxis2=dict(
            title='Wind Speed (m/s)',
            overlaying='y',
            side='right',
            showgrid=False,
            font=dict(color='#f43f5e')
        ),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    return apply_layout(fig, "Renewable Generation Profiles & Environmental Input", "Simulation Hour", "Power Output (kW)")

def plot_forecast_vs_actual(df):
    """
    Plots Forecast vs Actual Demand with residuals underneath.
    """
    fig = go.Figure()
    
    # Actual demand
    fig.add_trace(go.Scatter(
        x=df['Hour'],
        y=df['Total_Demand_kW'],
        mode='lines',
        line=dict(color='#3b82f6', width=2),
        name='Actual Load'
    ))
    
    # Forecast demand
    fig.add_trace(go.Scatter(
        x=df['Hour'],
        y=df['Forecast_Demand_kW'],
        mode='lines',
        line=dict(color=COLORS['forecast'], width=2, dash='dash'),
        name='ML Forecast (GBR)'
    ))
    
    # Error bands (confidence bounds simulated for display)
    error = np.abs(df['Total_Demand_kW'] - df['Forecast_Demand_kW'])
    std_err = np.std(error)
    
    fig.add_trace(go.Scatter(
        x=pd.concat([df['Hour'], df['Hour'].iloc[::-1]]),
        y=pd.concat([df['Forecast_Demand_kW'] + 1.96 * std_err, (df['Forecast_Demand_kW'] - 1.96 * std_err).iloc[::-1]]),
        fill='toself',
        fillcolor='rgba(236, 72, 153, 0.15)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        name='95% Prediction Interval'
    ))
    
    return apply_layout(fig, "Machine Learning Validation: Forecast vs. Actual Port Demand", "Simulation Hour", "Power (kW)")

def plot_battery_soc(df, opt_results, battery_capacity):
    """
    Plots Battery State of Charge (SoC) and battery charge/discharge actions.
    """
    fig = go.Figure()
    
    # SoC Curve (%)
    soc_pct = (opt_results['battery_soc'] / battery_capacity) * 100
    fig.add_trace(go.Scatter(
        x=df['Hour'],
        y=soc_pct,
        mode='lines',
        line=dict(color=COLORS['battery'], width=2.5),
        name='Battery SoC (%)',
        yaxis='y1'
    ))
    
    # Charge/Discharge columns
    fig.add_trace(go.Bar(
        x=df['Hour'],
        y=opt_results['battery_charge'],
        marker_color='#10b981',
        opacity=0.6,
        name='Charging Power (kW)',
        yaxis='y2'
    ))
    
    fig.add_trace(go.Bar(
        x=df['Hour'],
        y=-opt_results['battery_discharge'],
        marker_color='#ef4444',
        opacity=0.6,
        name='Discharging Power (kW)',
        yaxis='y2'
    ))
    
    fig.update_layout(
        yaxis2=dict(
            title='Battery Power Action (kW)',
            overlaying='y',
            side='right',
            showgrid=False
        ),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    return apply_layout(fig, "BESS Operations: State of Charge (SoC) and Charging Decisions", "Simulation Hour", "SoC (%)")

def plot_dispatch_comparison(df, opt_results):
    """
    Compares the Grid Import before and after MILP dispatch optimization.
    """
    fig = go.Figure()
    
    grid_before = np.maximum(0, df['Total_Demand_kW'] - df['Total_Renewables_kW'])
    grid_after = opt_results['grid_import']
    
    fig.add_trace(go.Scatter(
        x=df['Hour'],
        y=grid_before,
        mode='lines',
        line=dict(color='#f97316', width=1.5),
        name='Baseline Grid Import (Unoptimized)'
    ))
    
    fig.add_trace(go.Scatter(
        x=df['Hour'],
        y=grid_after,
        mode='lines',
        line=dict(color=COLORS['grid_opt'], width=2),
        fill='tozeroy',
        fillcolor='rgba(16, 185, 129, 0.1)',
        name='Optimized Grid Import (MILP)'
    ))
    
    # Highlight Peak pricing hours in background
    for day in range(df['Day'].max()):
        # Peak prices are between 17:00 and 21:00
        start_peak = day * 24 + 17
        end_peak = day * 24 + 21
        fig.add_vrect(
            x0=start_peak, x1=end_peak,
            fillcolor="#ef4444", opacity=0.08,
            layer="below", line_width=0,
            annotation_text="Peak Rate", 
            annotation_position="top left",
            annotation_font=dict(color="#ef4444", size=10)
        )
        
    return apply_layout(fig, "MILP Impact: Grid Import Shaving & Peak Cost Avoidance", "Simulation Hour", "Grid Import Power (kW)")

def plot_feature_importance(importance_df):
    """
    Plots the ML feature importance bar chart.
    """
    fig = px.bar(
        importance_df,
        x='Importance',
        y='Feature',
        orientation='h',
        color='Importance',
        color_continuous_scale='tealgrn'
    )
    
    fig.update_layout(
        coloraxis_showscale=False,
        yaxis=dict(autorange="reversed")
    )
    
    return apply_layout(fig, "Gradient Boosting Regressor: Feature Importance Drivers", "Relative Feature Weight", "Feature Name")


def get_css():
    """
    Returns custom CSS for the Streamlit dashboard to create a modern, academic layout.
    """
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Inter:wght@300;400;500;700&display=swap');
    
    /* Global Overrides */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1250px;
    }
    
    h1, h2, h3, h4 {
        font-family: 'Outfit', sans-serif !important;
        color: #0f172a;
        font-weight: 700;
    }
    
    body {
        font-family: 'Inter', sans-serif;
    }
    
    /* Glassmorphism Metric Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(226, 232, 240, 0.8);
        border-radius: 16px;
        padding: 1.25rem;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.05), 0 2px 4px -2px rgb(0 0 0 / 0.05);
        transition: transform 0.2s, box-shadow 0.2s;
        margin-bottom: 1rem;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgb(0 0 0 / 0.07), 0 4px 6px -4px rgb(0 0 0 / 0.07);
    }
    
    .metric-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .metric-val {
        font-size: 1.8rem;
        font-weight: 800;
        color: #0f172a;
        font-family: 'Outfit', sans-serif;
        line-height: 1;
        margin-bottom: 0.25rem;
    }
    
    .metric-delta {
        font-size: 0.85rem;
        font-weight: 600;
        display: flex;
        align-items: center;
    }
    
    .delta-up {
        color: #ef4444;
    }
    
    .delta-down {
        color: #10b981;
    }
    
    /* Research Box Info */
    .research-box {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
        border-radius: 16px;
        padding: 1.75rem;
        border: 1px solid #334155;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.3);
        margin-bottom: 2rem;
    }
    
    .research-box h2, .research-box h3 {
        color: #38bdf8 !important;
    }
    
    .research-box p {
        color: #cbd5e1;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    /* Custom Tab styling hint */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f1f5f9;
        border: 1px solid #e2e8f0;
        border-radius: 8px 8px 0px 0px;
        padding: 8px 16px;
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        color: #475569;
        transition: all 0.2s;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e2e8f0;
        color: #0f172a;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #0f172a !important;
        color: #ffffff !important;
        border-color: #0f172a !important;
    }
    </style>
    """
