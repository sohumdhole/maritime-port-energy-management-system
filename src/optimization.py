import numpy as np
import pandas as pd
import warnings

# Try importing OR-Tools
OR_TOOLS_AVAILABLE = False
try:
    from ortools.linear_solver import pywraplp
    OR_TOOLS_AVAILABLE = True
except ImportError:
    pass

# Try importing SciPy MILP
SCIPY_AVAILABLE = False
try:
    from scipy.optimize import milp, Bounds, LinearConstraint
    SCIPY_AVAILABLE = True
except ImportError:
    pass


class PortEnergyOptimizer:
    def __init__(self, 
                 battery_capacity=5000,     # kWh
                 battery_max_power=1000,    # kW
                 battery_efficiency=0.90,   # round-trip efficiency
                 soc_min_pct=0.20,          # min State of Charge %
                 soc_max_pct=1.00,          # max State of Charge %
                 soc_init_pct=0.50,         # initial State of Charge %
                 battery_wear_cost=0.015,   # $/kWh of battery throughput
                 curtailment_penalty=0.10,  # $/kWh of curtailed renewable energy
                 grid_peak_penalty=0.05):    # $/kW peak monthly demand charge
        self.battery_capacity = battery_capacity
        self.battery_max_power = battery_max_power
        self.battery_efficiency = battery_efficiency
        # Charge and discharge efficiencies are assumed equal (sqrt of round-trip efficiency)
        self.eta_ch = np.sqrt(battery_efficiency)
        self.eta_dis = np.sqrt(battery_efficiency)
        
        self.soc_min = battery_capacity * soc_min_pct
        self.soc_max = battery_capacity * soc_max_pct
        self.soc_init = battery_capacity * soc_init_pct
        
        self.battery_wear_cost = battery_wear_cost
        self.curtailment_penalty = curtailment_penalty
        self.grid_peak_penalty = grid_peak_penalty

    def get_electricity_prices(self, base_price, total_hours):
        """
        Generates a time-of-use (TOU) tariff profile.
        - Off-peak (23:00 - 07:00): 60% of base price
        - Mid-peak (07:00 - 17:00, 21:00 - 23:00): 100% of base price
        - Peak (17:00 - 21:00): 170% of base price
        """
        prices = np.zeros(total_hours)
        for h in range(total_hours):
            hr = h % 24
            if hr < 7 or hr >= 23:
                prices[h] = base_price * 0.60
            elif 17 <= hr < 21:
                prices[h] = base_price * 1.70
            else:
                prices[h] = base_price * 1.00
        return prices

    def optimize_ortools(self, demand, renewables, prices):
        """
        Solves the MILP dispatch optimization using Google OR-Tools.
        """
        T = len(demand)
        # Create the MIP solver with SCIP or CBC
        solver = pywraplp.Solver.CreateSolver('SCIP')
        if not solver:
            # Try CBC as fallback
            solver = pywraplp.Solver.CreateSolver('CBC')
            if not solver:
                raise RuntimeError("OR-Tools solver (SCIP/CBC) could not be initialized.")
        
        infinity = solver.infinity()
        
        # Decision Variables
        P_grid = [solver.NumVar(0.0, infinity, f'P_grid_{t}') for t in range(T)]
        P_ch = [solver.NumVar(0.0, self.battery_max_power, f'P_ch_{t}') for t in range(T)]
        P_dis = [solver.NumVar(0.0, self.battery_max_power, f'P_dis_{t}') for t in range(T)]
        P_curt = [solver.NumVar(0.0, infinity, f'P_curt_{t}') for t in range(T)]
        SoC = [solver.NumVar(self.soc_min, self.soc_max, f'SoC_{t}') for t in range(T)]
        
        # Binary variables to prevent simultaneous charging and discharging
        z_ch = [solver.BoolVar(f'z_ch_{t}') for t in range(T)]
        z_dis = [solver.BoolVar(f'z_dis_{t}') for t in range(T)]
        
        # Peak grid import variable for peak demand charge
        P_grid_peak = solver.NumVar(0.0, infinity, 'P_grid_peak')
        
        # Constraints
        for t in range(T):
            # 1. Power balance
            # Grid + Discharge + Renewables - Curtailment = Demand + Charge
            solver.Add(P_grid[t] + P_dis[t] + renewables[t] - P_curt[t] == demand[t] + P_ch[t])
            
            # 2. Battery Dynamics
            if t == 0:
                solver.Add(SoC[t] == self.soc_init + self.eta_ch * P_ch[t] - (P_dis[t] / self.eta_dis))
            else:
                solver.Add(SoC[t] == SoC[t-1] + self.eta_ch * P_ch[t] - (P_dis[t] / self.eta_dis))
                
            # 3. Charging/Discharging bounds coupled with binary variables
            solver.Add(P_ch[t] <= z_ch[t] * self.battery_max_power)
            solver.Add(P_dis[t] <= z_dis[t] * self.battery_max_power)
            solver.Add(z_ch[t] + z_dis[t] <= 1)
            
            # 4. Peak grid tracking
            solver.Add(P_grid_peak >= P_grid[t])
            
            # 5. Limit Curtailment to available renewables
            solver.Add(P_curt[t] <= renewables[t])
            
        # Objective Function
        objective = solver.Objective()
        for t in range(T):
            # Grid import cost
            objective.SetCoefficient(P_grid[t], float(prices[t]))
            # Battery degradation/wear cost
            objective.SetCoefficient(P_ch[t], float(self.battery_wear_cost))
            objective.SetCoefficient(P_dis[t], float(self.battery_wear_cost))
            # Curtailment penalty
            objective.SetCoefficient(P_curt[t], float(self.curtailment_penalty))
            
        # Monthly Peak demand charge penalty
        objective.SetCoefficient(P_grid_peak, float(self.grid_peak_penalty))
        objective.SetMinimization()
        
        # Solve
        status = solver.Solve()
        
        if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
            # Extract results
            res_grid = [P_grid[t].solution_value() for t in range(T)]
            res_ch = [P_ch[t].solution_value() for t in range(T)]
            res_dis = [P_dis[t].solution_value() for t in range(T)]
            res_curt = [P_curt[t].solution_value() for t in range(T)]
            res_soc = [SoC[t].solution_value() for t in range(T)]
            
            return {
                'status': 'optimal',
                'grid_import': np.array(res_grid),
                'battery_charge': np.array(res_ch),
                'battery_discharge': np.array(res_dis),
                'battery_soc': np.array(res_soc),
                'renewable_curtailment': np.array(res_curt)
            }
        else:
            raise RuntimeError("OR-Tools failed to find a feasible solution.")

    def optimize_heuristic(self, demand, renewables, prices):
        """
        Fallback Heuristic Dispatch Optimizer.
        Provides a smart rule-based scheduling logic:
        1. Peak periods: Maximize battery discharging to reduce grid imports and shave peak.
        2. Off-peak periods: Charge battery using cheap grid power if necessary, or excess renewables.
        3. Renewable-rich periods: Route excess renewables to charge battery first, curtail remaining excess.
        """
        T = len(demand)
        res_grid = np.zeros(T)
        res_ch = np.zeros(T)
        res_dis = np.zeros(T)
        res_curt = np.zeros(T)
        res_soc = np.zeros(T)
        
        current_soc = self.soc_init
        
        # Determine peak price threshold to decide discharge behavior
        price_median = np.median(prices)
        
        for t in range(T):
            dem = demand[t]
            ren = renewables[t]
            pr = prices[t]
            
            net_dem = dem - ren
            
            # Case 1: Excess Renewables (Net Demand < 0)
            if net_dem < 0:
                excess_ren = -net_dem
                # We can charge battery
                available_charge_capacity = (self.soc_max - current_soc) / self.eta_ch
                charge_power = min(excess_ren, self.battery_max_power, available_charge_capacity)
                
                res_ch[t] = charge_power
                res_dis[t] = 0.0
                current_soc += charge_power * self.eta_ch
                
                res_grid[t] = 0.0
                res_curt[t] = excess_ren - charge_power
                
            # Case 2: Deficit (Net Demand > 0)
            else:
                # Decide if we should discharge battery
                # Rule: Discharge if price is above median (expensive hour)
                if pr > price_median:
                    available_discharge_capacity = (current_soc - self.soc_min) * self.eta_dis
                    discharge_power = min(net_dem, self.battery_max_power, available_discharge_capacity)
                    
                    res_dis[t] = discharge_power
                    res_ch[t] = 0.0
                    current_soc -= discharge_power / self.eta_dis
                    
                    res_grid[t] = net_dem - discharge_power
                    res_curt[t] = 0.0
                # Rule: Cheap price hours (charge battery from grid up to some target if price is very low)
                elif pr < price_median * 0.8:
                    # Low price: import from grid to charge battery and meet demand
                    available_charge_capacity = (self.soc_max - current_soc) / self.eta_ch
                    charge_power = min(self.battery_max_power * 0.5, available_charge_capacity) # slow charge
                    
                    res_ch[t] = charge_power
                    res_dis[t] = 0.0
                    current_soc += charge_power * self.eta_ch
                    
                    res_grid[t] = net_dem + charge_power
                    res_curt[t] = 0.0
                else:
                    # Neutral price: satisfy demand from grid, battery idle
                    res_ch[t] = 0.0
                    res_dis[t] = 0.0
                    res_grid[t] = net_dem
                    res_curt[t] = 0.0
                    
            res_soc[t] = current_soc
            
        return {
            'status': 'heuristic_fallback',
            'grid_import': res_grid,
            'battery_charge': res_ch,
            'battery_discharge': res_dis,
            'battery_soc': res_soc,
            'renewable_curtailment': res_curt
        }

    def run_optimization(self, demand, renewables, prices):
        """
        Executes the optimization. Tries OR-Tools first, then falls back to Heuristic.
        """
        if OR_TOOLS_AVAILABLE:
            try:
                return self.optimize_ortools(demand, renewables, prices)
            except Exception as e:
                warnings.warn(f"OR-Tools solver failed: {e}. Falling back to heuristic solver.")
                return self.optimize_heuristic(demand, renewables, prices)
        else:
            warnings.warn("OR-Tools is not installed. Falling back to heuristic solver.")
            return self.optimize_heuristic(demand, renewables, prices)

    def calculate_kpis(self, demand, renewables, prices, opt_results):
        """
        Calculates KPIs comparing the optimized state against the unoptimized baseline.
        - Baseline: Grid import covers the entire net demand (no battery usage, excess renewables are curtailed).
        - Optimized: Battery charging and discharging schedules are applied to the ACTUAL demand.
                    Grid import and curtailment are recalculated based on actual power balance.
        """
        T = len(demand)
        
        # Extract scheduled battery actions
        ch_opt = opt_results['battery_charge']
        dis_opt = opt_results['battery_discharge']
        
        # Recalculate actual grid import and curtailment under the optimized BESS schedule
        # Grid + Discharge + Renewables - Curtailment = Demand + Charge
        # -> Grid - Curtailment = Demand + Charge - Discharge - Renewables
        net_load = demand + ch_opt - dis_opt - renewables
        
        grid_import_opt = np.maximum(0, net_load)
        curt_opt = np.maximum(0, -net_load)
        
        # Update opt_results in-place so downstream visualizations use actual realized profiles
        opt_results['grid_import'] = grid_import_opt
        opt_results['renewable_curtailment'] = curt_opt
        
        # Baseline: satisfy net demand from grid, excess renewables are curtailed
        grid_import_base = np.maximum(0, demand - renewables)
        curt_base = np.maximum(0, renewables - demand)
        
        # Costs (Financial only - do NOT include virtual curtailment penalties in financial KPIs!)
        cost_base_energy = np.sum(grid_import_base * prices)
        cost_opt_energy = np.sum(grid_import_opt * prices)
        
        # Battery wear cost (actual physical degradation cost)
        cost_opt_batt = np.sum((ch_opt + dis_opt) * self.battery_wear_cost)
        
        # Monthly Peak penalty
        peak_base = np.max(grid_import_base)
        peak_opt = np.max(grid_import_opt)
        cost_base_peak = peak_base * self.grid_peak_penalty
        cost_opt_peak = peak_opt * self.grid_peak_penalty
        
        total_cost_base = cost_base_energy + cost_base_peak
        total_cost_opt = cost_opt_energy + cost_opt_batt + cost_opt_peak
        
        # Savings
        cost_savings = total_cost_base - total_cost_opt
        cost_savings_pct = (cost_savings / total_cost_base) * 100 if total_cost_base > 0 else 0
        
        # Grid import reduction
        total_grid_base = np.sum(grid_import_base)
        total_grid_opt = np.sum(grid_import_opt)
        grid_reduction = total_grid_base - total_grid_opt
        grid_reduction_pct = (grid_reduction / total_grid_base) * 100 if total_grid_base > 0 else 0
        
        # Renewable utilization
        total_renew_gen = np.sum(renewables)
        renew_util_base = (1 - (np.sum(curt_base) / total_renew_gen)) * 100 if total_renew_gen > 0 else 100
        renew_util_opt = (1 - (np.sum(curt_opt) / total_renew_gen)) * 100 if total_renew_gen > 0 else 100
        
        # Carbon emissions reduction (assuming grid is carbon-intensive: 0.4 kg CO2 / kWh)
        co2_factor = 0.4 # kg CO2 / kWh
        co2_saved = (total_grid_base - total_grid_opt) * co2_factor
        
        return {
            'baseline_cost': total_cost_base,
            'optimized_cost': total_cost_opt,
            'cost_savings_usd': cost_savings,
            'cost_savings_percent': cost_savings_pct,
            'baseline_grid_mwh': total_grid_base / 1000.0,
            'optimized_grid_mwh': total_grid_opt / 1000.0,
            'grid_reduction_mwh': grid_reduction / 1000.0,
            'grid_reduction_percent': grid_reduction_pct,
            'baseline_renew_util': renew_util_base,
            'optimized_renew_util': renew_util_opt,
            'peak_shaving_percent': (1 - peak_opt / peak_base) * 100 if peak_base > 0 else 0,
            'co2_saved_kg': co2_saved,
            'baseline_peak_kw': peak_base,
            'optimized_peak_kw': peak_opt
        }
