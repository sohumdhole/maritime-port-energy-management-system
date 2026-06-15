import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    print("=== Maritime Port Energy Management System - Validation Script ===")
    print("-------------------------------------------------------------")
    
    # 1. Test imports
    print("[1/4] Testing module imports...")
    from src.simulation import PortDigitalTwin
    from src.forecasting import PortDemandForecaster
    from src.optimization import PortEnergyOptimizer
    print("-> Module imports successful.")

    # 2. Run 7-day simulation
    print("\n[2/5] Running 7-day port simulation...")
    twin = PortDigitalTwin(simulation_days=7, num_vessels=15, solar_capacity=2000, wind_capacity=3000, seed=42)
    df, vessels = twin.simulate()
    print(f"-> Simulation complete. Generated {len(df)} hourly steps.")
    print(f"-> Scheduled vessels count: {len(vessels)}")
    print(f"-> Peak Load simulated: {df['Total_Demand_kW'].max():.2f} kW")
    print(f"-> Peak Solar generation: {df['Solar_Gen_kW'].max():.2f} kW")

    # 3. Run forecasting on historical dataset
    print("\n[3/5] Training Gradient Boosting model and forecasting...")
    # Generate 30 days history for training
    twin_hist = PortDigitalTwin(simulation_days=30, num_vessels=int(15 * (30/7)), solar_capacity=2000, wind_capacity=3000, seed=101)
    hist_df, _ = twin_hist.simulate()
    
    forecaster = PortDemandForecaster(seed=42)
    forecaster.train(hist_df)
    test_processed, metrics, feat_importance = forecaster.predict(df, hist_df)
    print("-> Forecasting complete.")
    print(f"-> MAE: {metrics['MAE_kW']:.2f} kW")
    print(f"-> RMSE: {metrics['RMSE_kW']:.2f} kW")
    print(f"-> MAPE: {metrics['MAPE_percent']:.2f}%")

    # 4. Run optimization
    print("\n[4/5] Running MILP dispatch optimization...")
    optimizer = PortEnergyOptimizer(
        battery_capacity=5000,
        battery_max_power=1500,
        battery_efficiency=0.90
    )
    prices = optimizer.get_electricity_prices(0.20, len(test_processed))
    
    # Run optimization using the forecasted demand
    opt_results = optimizer.run_optimization(
        test_processed['Forecast_Demand_kW'].values, 
        test_processed['Total_Renewables_kW'].values, 
        prices
    )
    print(f"-> Optimization solver status: {opt_results['status']}")
    
    # Calculate KPIs
    kpis = optimizer.calculate_kpis(
        test_processed['Total_Demand_kW'].values, 
        test_processed['Total_Renewables_kW'].values, 
        prices, 
        opt_results
    )
    print(f"-> Optimized Cost Savings: ${kpis['cost_savings_usd']:.2f} ({kpis['cost_savings_percent']:.1f}%)")
    print(f"-> Grid Import Reduction: {kpis['grid_reduction_percent']:.1f}%")
    print(f"-> Optimized Renewable Utilisation: {kpis['optimized_renew_util']:.1f}%")
    
    # 5. Save synthetic CSV outputs
    print("\n[5/5] Saving synthetic CSV outputs to /data/...")
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/synthetic_port_demand_7days.csv', index=False)
    
    # Export full dispatch log
    test_processed['Grid_Import_Opt_kW'] = opt_results['grid_import']
    test_processed['Battery_SoC_Opt_kWh'] = opt_results['battery_soc']
    test_processed['Battery_Charge_Opt_kW'] = opt_results['battery_charge']
    test_processed['Battery_Discharge_Opt_kW'] = opt_results['battery_discharge']
    test_processed.to_csv('data/optimized_dispatch_schedule.csv', index=False)
    print("-> Data files saved successfully to /data/ folder.")
    
    print("\n=== ALL CORE PIPELINES ARE FULLY FUNCTIONAL AND VALIDATED! ===")
    
except Exception as e:
    print(f"\n[ERROR] Validation failed: {e}")
    sys.exit(1)
