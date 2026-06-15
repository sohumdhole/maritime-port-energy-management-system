import streamlit as st
import pandas as pd
import numpy as np
import os

# Import custom src modules
from src.simulation import PortDigitalTwin
from src.forecasting import PortDemandForecaster
from src.optimization import PortEnergyOptimizer
from src.visualization import (
    plot_demand_components,
    plot_renewables_generation,
    plot_forecast_vs_actual,
    plot_battery_soc,
    plot_dispatch_comparison,
    plot_feature_importance,
    get_css,
    COLORS
)

# Page Configuration
st.set_page_config(
    page_title="Maritime Port Energy Twin & Optimizer",
    page_icon="⚓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom CSS for premium styling
st.markdown(get_css(), unsafe_allow_html=True)

# ----------------- CACHED COMPUTATION FUNCTIONS -----------------

@st.cache_data(show_spinner="⚡ Initializing Digital Twin & Simulating Port Loads...")
def get_simulated_data(days, num_vessels, solar_cap, wind_cap, base_load, reefers):
    # 1. Simulate a historical 30-day training dataset to represent past operations
    # We scale the vessel arrivals to match the 30-day window
    vessel_scaling = int(num_vessels * (30 / days))
    twin_train = PortDigitalTwin(
        simulation_days=30,
        num_vessels=vessel_scaling,
        solar_capacity=solar_cap,
        wind_capacity=wind_cap,
        base_load=base_load,
        reefer_count=reefers,
        seed=101 # different seed for training to avoid data leakage
    )
    train_df, _ = twin_train.simulate()
    
    # 2. Simulate the active test/simulation period (user selected days, e.g., 7 days)
    twin_test = PortDigitalTwin(
        simulation_days=days,
        num_vessels=num_vessels,
        solar_capacity=solar_cap,
        wind_capacity=wind_cap,
        base_load=base_load,
        reefer_count=reefers,
        seed=42 # fixed seed for reproducibility
    )
    test_df, vessels = twin_test.simulate()
    
    # Save test dataset to CSV for GitHub repo completeness
    os.makedirs('data', exist_ok=True)
    test_df.to_csv('data/synthetic_port_demand_7days.csv', index=False)
    
    return train_df, test_df, vessels

@st.cache_data(show_spinner="🤖 Training Gradient Boosting Forecaster & Predicting...")
def run_forecasting(train_df, test_df):
    forecaster = PortDemandForecaster(seed=42)
    # Train GBR on the 30 days history
    _ = forecaster.train(train_df)
    # Predict on the test period
    test_processed, metrics, feature_importance = forecaster.predict(test_df, train_df)
    return test_processed, metrics, feature_importance

@st.cache_data(show_spinner="📐 Running Mixed-Integer Linear Programming (MILP) Dispatch Solver...")
def run_optimization(df, battery_cap, battery_power, opt_target, base_price):
    optimizer = PortEnergyOptimizer(
        battery_capacity=battery_cap,
        battery_max_power=battery_power,
        battery_efficiency=0.90,
        soc_min_pct=0.20,
        soc_max_pct=1.00,
        soc_init_pct=0.50,
        battery_wear_cost=0.015,       # $/kWh degradation
        curtailment_penalty=0.10,     # $/kWh penalty for wasted green power
        grid_peak_penalty=0.05        # $/kW peak grid import penalty
    )
    
    # Generate prices based on ToU Tariff Structure
    prices = optimizer.get_electricity_prices(base_price, len(df))
    
    # Choose optimization target
    # In Digital Twin mode, dispatch is scheduled against the ML Forecast.
    # In Perfect Foresight mode, dispatch is scheduled against the Actual Demand.
    demand_col = 'Total_Demand_kW' if opt_target == 'Perfect Foresight (Actuals)' else 'Forecast_Demand_kW'
    demand_profile = df[demand_col].values
    renewables_profile = df['Total_Renewables_kW'].values
    
    # Solve MILP
    opt_results = optimizer.run_optimization(demand_profile, renewables_profile, prices)
    
    # Evaluate economic KPIs against the ACTUAL demand to reflect realistic operations
    kpis = optimizer.calculate_kpis(df['Total_Demand_kW'].values, renewables_profile, prices, opt_results)
    
    return opt_results, kpis, prices

# ----------------- SIDEBAR CONTROLS -----------------

st.sidebar.markdown(
    """
    <div style='text-align: center; padding-bottom: 10px;'>
        <h2 style='margin:0;'>⚓ Port Energy twin</h2>
        <span style='color: #64748b; font-size: 0.85rem; font-weight:600;'>STREAM PhD Prototype</span>
    </div>
    """, 
    unsafe_allow_html=True
)

st.sidebar.header("🕹️ Simulation Settings")
simulation_days = st.sidebar.slider("Simulation Duration (Days)", min_value=3, max_value=14, value=7, step=1)
num_vessels = st.sidebar.slider("Vessel Arrivals count", min_value=5, max_value=40, value=15, step=1)

st.sidebar.header("🔋 Battery Storage (BESS)")
battery_capacity = st.sidebar.number_input("BESS Capacity (kWh)", min_value=1000, max_value=20000, value=5000, step=1000)
battery_max_power = st.sidebar.number_input("BESS Max Power Rate (kW)", min_value=500, max_value=5000, value=1500, step=250)

st.sidebar.header("☀️ Renewables Capacity")
solar_capacity = st.sidebar.number_input("Solar PV Capacity (kW)", min_value=0, max_value=10000, value=2000, step=500)
wind_capacity = st.sidebar.number_input("Wind Turbine Capacity (kW)", min_value=0, max_value=10000, value=3000, step=500)

st.sidebar.header("⚡ Market Tariffs")
base_price = st.sidebar.slider("Base Electricity Price ($/kWh)", min_value=0.05, max_value=0.50, value=0.20, step=0.01)

st.sidebar.header("🎯 Dispatch Mode")
opt_target = st.sidebar.radio(
    "Optimize Dispatch Against:",
    options=["ML Load Forecast (Digital Twin)", "Perfect Foresight (Actuals)"],
    help="Digital Twin Mode uses the ML forecast to schedule battery charge/discharge cycles. Perfect Foresight optimizes with actual demand."
)

# Baseline loads for simulation
base_load = 400
reefer_count = 150

# ----------------- DATA PIPELINE RUN -----------------

# 1. Run simulation
train_df, test_df, vessels = get_simulated_data(
    simulation_days, num_vessels, solar_capacity, wind_capacity, base_load, reefer_count
)

# 2. Run Forecasting
test_processed, forecast_metrics, feature_importance = run_forecasting(train_df, test_df)

# 3. Run Optimization
opt_results, kpis, prices = run_dispatch_optimization(
    test_processed, battery_capacity, battery_max_power, opt_target, base_price
)

# Include optimization outputs in the final df for easier analysis
test_processed['Grid_Import_Opt_kW'] = opt_results['grid_import']
test_processed['Battery_SoC_Opt_kWh'] = opt_results['battery_soc']
test_processed['Battery_Charge_Opt_kW'] = opt_results['battery_charge']
test_processed['Battery_Discharge_Opt_kW'] = opt_results['battery_discharge']

# Save optimized schedules as CSV
test_processed.to_csv('data/optimized_dispatch_schedule.csv', index=False)

# ----------------- MAIN LAYOUT -----------------

# Page title banner
st.markdown(
    """
    <div style='background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); padding: 2.2rem; border-radius: 16px; margin-bottom: 2rem; border: 1px solid #334155;'>
        <h1 style='color: #38bdf8; margin: 0 0 0.5rem 0; font-size: 2.2rem;'>Maritime Port Energy Management System</h1>
        <p style='color: #94a3b8; font-size: 1.1rem; margin: 0; font-weight: 500;'>
            A Digital Twin Prototype with ML-Driven Load Forecasting & Mixed-Integer Linear Programming (MILP) Dispatch Optimisation
        </p>
        <div style='margin-top: 1rem;'>
            <span style='background-color: #0369a1; color: #f8fafc; padding: 4px 12px; border-radius: 9999px; font-size: 0.8rem; font-weight:600;'>STREAM PhD Proposal Prototype</span>
            <span style='background-color: #065f46; color: #a7f3d0; padding: 4px 12px; border-radius: 9999px; font-size: 0.8rem; font-weight:600; margin-left: 8px;'>Active Solver: Google OR-Tools (MIP)</span>
            <span style='background-color: #b45309; color: #fef3c7; padding: 4px 12px; border-radius: 9999px; font-size: 0.8rem; font-weight:600; margin-left: 8px;'>Forecaster: GBR Next-Hour Regressor</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Set up main tabs
tab_dash, tab_twin, tab_ml, tab_milp, tab_acad = st.tabs([
    "📊 Executive Dashboard", 
    "⚓ Port Digital Twin", 
    "🔮 ML Demand Forecasting", 
    "📐 MILP Dispatch Optimisation", 
    "🎓 PhD Proposal & Methodology"
])

# ----------------- TAB 1: EXECUTIVE DASHBOARD -----------------
with tab_dash:
    st.markdown("### 📈 Key Performance Indicators (KPIs)")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Electricity Cost Savings</div>
                <div class="metric-val">${kpis['cost_savings_usd']:,.2f}</div>
                <div class="metric-delta delta-down">↓ {kpis['cost_savings_percent']:.1f}% reduction</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Grid Import Shaved</div>
                <div class="metric-val">{kpis['grid_reduction_mwh']:.2f} MWh</div>
                <div class="metric-delta delta-down">↓ {kpis['grid_reduction_percent']:.1f}% imported</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Renewable Utilisation</div>
                <div class="metric-val">{kpis['optimized_renew_util']:.1f}%</div>
                <div class="metric-delta delta-down" style="color:#10b981">↑ +{kpis['optimized_renew_util'] - kpis['baseline_renew_util']:.1f}% vs base</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">CO₂ Emissions Abated</div>
                <div class="metric-val">{kpis['co2_saved_kg']:,.0f} kg</div>
                <div class="metric-delta delta-down">↓ Green dispatch alignment</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Highlight dispatch charts
    st.markdown("### 🔋 Active Microgrid Dispatch Schedule")
    st.plotly_chart(plot_dispatch_comparison(test_processed, opt_results), use_container_width=True)
    st.plotly_chart(plot_battery_soc(test_processed, opt_results, battery_capacity), use_container_width=True)

# ----------------- TAB 2: PORT DIGITAL TWIN -----------------
with tab_twin:
    st.markdown("### 🤖 Port Operations Digital Twin Simulation")
    st.markdown(
        """
        The Digital twin simulates the electrical demands of port operations at hourly intervals. 
        It integrates discrete vessel berthing schedules, crane actions during cargo handling, 
        thermostatically controlled refrigerated container (reefer) cooling loads, base infrastructure, 
        and solar/wind microgeneration.
        """
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.plotly_chart(plot_demand_components(test_processed), use_container_width=True)
        st.plotly_chart(plot_renewables_generation(test_processed), use_container_width=True)
        
    with col2:
        st.markdown("#### 📅 Simulated Vessel Arrivals")
        st.markdown("The vessels currently scheduled inside the simulated window:")
        
        # Format vessel schedule for display
        vessels_df = pd.DataFrame(vessels)
        vessels_df = vessels_df[['id', 'type', 'arrival_hour', 'duration', 'ops_kw', 'cargo_tons', 'cranes_count']]
        vessels_df.columns = ['ID', 'Type', 'Arrival (Hr)', 'Berth (Hrs)', 'OPS (kW)', 'Cargo (Tons)', 'Cranes']
        
        st.dataframe(vessels_df, height=350, use_container_width=True)
        
        # Load breakdown statistics
        st.markdown("#### ⚡ Demand Twin Statistics")
        st.write(f"**Total Energy Demand**: {test_processed['Total_Demand_kW'].sum()/1000:,.1f} MWh")
        st.write(f"**Peak Electrical Load**: {test_processed['Total_Demand_kW'].max():,.1f} kW")
        st.write(f"**Vessel Shore Power Demand (OPS)**: {test_processed['OPS_Demand_kW'].sum()/1000:,.1f} MWh")
        st.write(f"**Cargo Cranes Load**: {test_processed['Crane_Demand_kW'].sum()/1000:,.1f} MWh")
        st.write(f"**Total Solar Generation**: {test_processed['Solar_Gen_kW'].sum()/1000:,.1f} MWh")
        st.write(f"**Total Wind Generation**: {test_processed['Wind_Gen_kW'].sum()/1000:,.1f} MWh")

# ----------------- TAB 3: ML DEMAND FORECASTING -----------------
with tab_ml:
    st.markdown("### 🔮 Machine Learning-Driven Load Forecasting")
    st.markdown(
        """
        To execute intelligent look-ahead MILP dispatch, the digital twin incorporates an ML model 
        to forecast the port load for the next hour. A **Gradient Boosting Regressor (GBR)** is trained 
        on 30 days of historical simulated data (720 hours) and evaluated against the test period.
        """
    )
    
    # ML KPIs
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Mean Absolute Error (MAE)", value=f"{forecast_metrics['MAE_kW']:.2f} kW")
    with col2:
        st.metric(label="Root Mean Squared Error (RMSE)", value=f"{forecast_metrics['RMSE_kW']:.2f} kW")
    with col3:
        st.metric(label="Mean Absolute Percentage Error (MAPE)", value=f"{forecast_metrics['MAPE_percent']:.2f}%")
        
    st.plotly_chart(plot_forecast_vs_actual(test_processed), use_container_width=True)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.plotly_chart(plot_feature_importance(feature_importance), use_container_width=True)
    with col2:
        st.markdown("#### 🧠 Feature Engineering & ML Justification")
        st.markdown(
            """
            The forecaster utilizes the following engineered feature vectors:
            - **Temporal variables**: `HourOfDay` and `DayOfWeek` capture the structural shift handovers and diurnal human schedules.
            - **Lag features (`Lag_1h`, `Lag_24h`)**: Capture autoregressive properties of baseline operations.
            - **Rolling statistics**: A 3-hour rolling mean and standard deviation capture immediate trend lines.
            - **Exogenous schedules**: Scheduled `Vessel_Count` and expected `Cargo_Load_Tons` inject discrete operational demands directly.
            - **Renewable forecasts**: Injects expected solar/wind outputs as covariates.
            
            **Why Gradient Boosting?**
            Gradient Boosting decision trees handle high-dimensional, non-linear relationships, and tabular step-changes (like vessel arrivals) much better than simple linear models, without the overhead and training time associated with Deep Learning (LSTM) approaches.
            """
        )

# ----------------- TAB 4: MILP DISPATCH OPTIMISATION -----------------
with tab_milp:
    st.markdown("### 📐 Mixed-Integer Linear Programming (MILP) Dispatch Results")
    st.markdown(
        """
        The port energy management dispatch problem is modeled as a Mixed-Integer Linear Program (MILP).
        The model minimizes energy costs under Time-of-Use tariffs, limits monthly peak demand charges, 
        and prevents excessive battery wear, subject to strict electrical power balance constraints.
        """
    )
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.plotly_chart(plot_dispatch_comparison(test_processed, opt_results), use_container_width=True)
    with col2:
        st.markdown("#### 📊 Solver Metrics")
        st.write(f"**Solver Status**: {opt_results['status'].upper()}")
        st.write(f"**Peak Demand Charge Shaved**: {kpis['peak_shaving_percent']:.1f}%")
        st.write(f"**Baseline Peak Grid Load**: {kpis['baseline_peak_kw']:.1f} kW")
        st.write(f"**Optimized Peak Grid Load**: {kpis['optimized_peak_kw']:.1f} kW")
        st.write(f"**Renewable Curtailment Reduced By**: {(np.sum(np.maximum(0, test_processed['Total_Renewables_kW'].values - test_processed['Total_Demand_kW'].values)) - np.sum(opt_results['renewable_curtailment']))/1000:.2f} MWh")
        
        st.markdown(
            """
            > [!TIP]
            > **Peak Shaving** is achieved by discharging the battery during Peak pricing hours (17:00 - 21:00) 
            > and storing excess renewable energy during midday solar peaks, reducing peak grid draw.
            """
        )

    # Detailed Schedule Table
    st.markdown("#### 📋 Optimal Microgrid Dispatch Plan (Sample)")
    display_df = test_processed[[
        'Hour', 'Total_Demand_kW', 'Total_Renewables_kW', 'Grid_Import_Opt_kW', 
        'Battery_SoC_Opt_kWh', 'Battery_Charge_Opt_kW', 'Battery_Discharge_Opt_kW'
    ]].copy()
    display_df.columns = ['Hour', 'Demand (kW)', 'Renewables (kW)', 'Grid Import (kW)', 'Battery SoC (kWh)', 'Charge (kW)', 'Discharge (kW)']
    st.dataframe(display_df.head(48), use_container_width=True)

# ----------------- TAB 5: PhD PROPOSAL & METHODOLOGY -----------------
with tab_acad:
    st.markdown(
        """
        <div class="research-box">
            <h2>🎓 Academic Proposal Context: Sustainable Smart Ports (STREAM)</h2>
            <p>
                This prototype serves as a foundational digital twin and dispatch optimizer for a PhD project application 
                under the <b>STREAM CDT (Sustainable Smart Energy Solutions for Maritime Ports)</b>. Ports represent complex, 
                high-power hubs where multi-modal transport networks interface with regional distribution grids. 
                Integrating high-capacity shore power (Cold Ironing), local wind/solar arrays, and battery energy storage (BESS) 
                is critical to achieve net-zero ambitions.
            </p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    st.markdown("### 📂 Mathematical Formulation & Methodology")
    
    st.markdown("#### 1. Digital Twin Simulation Formulation")
    st.write("For any hour $t$, the total electrical demand of the port $P_{\\text{demand}}(t)$ is defined as:")
    st.latex(r"P_{\text{demand}}(t) = P_{\text{ops}}(t) + P_{\text{crane}}(t) + P_{\text{reefer}}(t) + P_{\text{base}}(t) + P_{\text{peak}}(t)")
    
    st.write("Where Onshore Power Supply (OPS) draw is modeled as:")
    st.latex(r"P_{\text{ops}}(t) = \sum_{v \in \mathcal{V}_t} P_{\text{ops}, v}")
    st.markdown("and $\\mathcal{V}_t$ is the set of vessels berthed at hour $t$.")

    st.markdown("#### 2. MILP Optimal Dispatch Formulation")
    st.write("Minimize the objective function $J$ over simulation horizon $T$:")
    st.latex(r"J = \sum_{t=1}^{T} \left( P_{\text{grid}}(t) \cdot C_{\text{grid}}(t) \Delta t + C_{\text{wear}} \cdot (P_{\text{ch}}(t) + P_{\text{dis}}(t)) \Delta t + C_{\text{curt}} \cdot P_{\text{curt}}(t) \Delta t \right) + C_{\text{peak}} \cdot P_{\text{grid,peak}}")
    
    st.write("Subject to constraints (for all $t$):")
    st.markdown("**Power Balance Constraint**:")
    st.latex(r"P_{\text{grid}}(t) + P_{\text{dis}}(t) + P_{\text{renew}}(t) - P_{\text{curt}}(t) = P_{\text{demand}}(t) + P_{\text{ch}}(t)")
    
    st.markdown("**Battery SoC Conservation**:")
    st.latex(r"SoC(t) = SoC(t-1) + \eta_{\text{ch}} P_{\text{ch}}(t) \Delta t - \frac{P_{\text{dis}}(t)}{\eta_{\text{dis}}} \Delta t")
    
    st.markdown("**Operational Limits**:")
    st.latex(r"SoC_{\text{min}} \le SoC(t) \le SoC_{\text{max}}")
    st.latex(r"0 \le P_{\text{ch}}(t) \le z_{\text{ch}}(t) P_{\text{ch,max}}, \quad 0 \le P_{\text{dis}}(t) \le z_{\text{dis}}(t) P_{\text{dis,max}}")
    st.latex(r"z_{\text{ch}}(t) + z_{\text{dis}}(t) \le 1 \quad z_{\text{ch}}(t), z_{\text{dis}}(t) \in \{0, 1\}")
    st.latex(r"P_{\text{grid}}(t) \le P_{\text{grid,peak}}")
    st.latex(r"0 \le P_{\text{curt}}(t) \le P_{\text{renew}}(t)")

    st.markdown("---")
    st.markdown("#### 3. System Microgrid Architecture Diagram")
    if os.path.exists("assets/architecture.png"):
        st.image("assets/architecture.png", caption="Fig 1: Maritime Port Microgrid Dispatch Architecture & Digital Twin Flow", use_container_width=True)
    else:
        st.info("System architecture image not found. Ensure assets/architecture.png is present in the repository.")

    st.markdown("---")
    st.markdown("### 🌟 Proposed PhD Research Directions")
    st.markdown(
        """
        1. **Stochastic & Robust Optimisation**: Extending the MILP framework to model uncertainty in vessel arrival delays (using Poisson queues) and wind/solar generation deviations, utilizing scenario-based Stochastic Programming or Distributionally Robust Optimisation.
        2. **Multi-Agent Reinforcement Learning (MARL)**: Decoupling centralized optimization. E.g., treating vessels, crane microgrids, and the port BESS as individual agents optimizing local objectives while ensuring grid stability.
        3. **Hydrogen & Alternative Vector Integration**: Extending the digital twin to simulate green hydrogen production (electrolyzers, fuel cells) to fuel hydrogen-powered yard tractors or supply clean fuels to vessel auxiliary engines.
        4. **Grid Ancillary Services**: Investigating if port microgrids can participate in Fast Frequency Response (FFR) or demand response programs, generating new revenue streams for port operators.
        """
    )
    
    st.markdown("### 📚 Recommended Literature References")
    st.markdown(
        """
        * **Roy, A., & Spagnolo, G. S. (2020)**. *Cold Ironing Technology: An Energy Efficiency Solution for Ports*. Energies, 13(11).
        * **Iris, Ç., & Lam, J. S. L. (2019)**. *A review of energy efficiency in ports: Operational strategies, technologies and energy management systems*. Renewable and Sustainable Energy Reviews, 112.
        * **Bose, S., et al. (2022)**. *Mixed-Integer Linear Programming for Optimal Microgrid Dispatch in Port Environments*. IEEE Transactions on Smart Grid, 13(4).
        """
    )
