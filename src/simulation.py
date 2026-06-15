import numpy as np
import pandas as pd

class PortDigitalTwin:
    def __init__(self, 
                 simulation_days=7, 
                 num_vessels=15, 
                 solar_capacity=2000,   # kW
                 wind_capacity=3000,    # kW
                 base_load=400,         # kW
                 reefer_count=150,      # number of reefers
                 seed=42):
        self.simulation_days = simulation_days
        self.total_hours = simulation_days * 24
        self.num_vessels = num_vessels
        self.solar_capacity = solar_capacity
        self.wind_capacity = wind_capacity
        self.base_load = base_load
        self.reefer_count = reefer_count
        self.seed = seed
        np.random.seed(seed)
        
    def generate_vessel_schedule(self):
        """
        Generates a synthetic schedule of vessel arrivals, mooring times, 
        OPS (shore power) demand, and cargo handling load.
        """
        vessel_types = ['Container', 'Cruise', 'Bulk Carrier']
        # Probabilities of vessel types
        vessel_probs = [0.6, 0.15, 0.25]
        
        # Load features per type (OPS kW, cargo tons, cranes count, berth duration hours)
        type_specs = {
            'Container': {'ops_kw': 1500, 'cargo_range': (2000, 8000), 'cranes': 3, 'berth_range': (18, 36)},
            'Cruise': {'ops_kw': 4000, 'cargo_range': (500, 1500), 'cranes': 1, 'berth_range': (10, 18)},
            'Bulk Carrier': {'ops_kw': 800, 'cargo_range': (5000, 15000), 'cranes': 2, 'berth_range': (24, 60)}
        }
        
        vessels = []
        # Distribute arrivals over the simulation duration
        arrival_hours = np.sort(np.random.randint(0, self.total_hours - 12, size=self.num_vessels))
        
        for i, arr_h in enumerate(arrival_hours):
            v_type = np.random.choice(vessel_types, p=vessel_probs)
            spec = type_specs[v_type]
            
            berth_dur = np.random.randint(spec['berth_range'][0], spec['berth_range'][1])
            dep_h = min(arr_h + berth_dur, self.total_hours)
            
            cargo = np.random.randint(spec['cargo_range'][0], spec['cargo_range'][1])
            
            vessels.append({
                'id': f'VES-{i+1:03d}',
                'type': v_type,
                'arrival_hour': int(arr_h),
                'departure_hour': int(dep_h),
                'duration': int(berth_dur),
                'ops_kw': spec['ops_kw'],
                'cargo_tons': cargo,
                'cranes_count': spec['cranes']
            })
            
        return vessels

    def simulate(self):
        """
        Runs the digital twin simulation step-by-step for the given duration,
        generating hourly demand profiles, environmental parameters, and renewable outputs.
        """
        np.random.seed(self.seed)
        vessels = self.generate_vessel_schedule()
        
        # Initialize hourly arrays
        hours = np.arange(self.total_hours)
        ops_demand = np.zeros(self.total_hours)
        crane_demand = np.zeros(self.total_hours)
        reefer_demand = np.zeros(self.total_hours)
        base_demand = np.zeros(self.total_hours)
        peak_demand = np.zeros(self.total_hours)
        vessel_counts = np.zeros(self.total_hours)
        cargo_load = np.zeros(self.total_hours)
        
        # 1. Simulate Vessel Demands (OPS + Cranes)
        for h in range(self.total_hours):
            active_vessels = [v for v in vessels if v['arrival_hour'] <= h < v['departure_hour']]
            vessel_counts[h] = len(active_vessels)
            
            for v in active_vessels:
                ops_demand[h] += v['ops_kw']
                
                # Cranes operate during the berthing period. 
                # Crane activity peaks in the middle 70% of the berthing duration.
                stay_pct = (h - v['arrival_hour']) / v['duration']
                if 0.15 <= stay_pct <= 0.85:
                    # Power: 250 kW per crane + variable load
                    crane_demand[h] += v['cranes_count'] * 250 * (0.8 + 0.4 * np.random.rand())
                    cargo_load[h] += v['cargo_tons'] / (v['duration'] * 0.7) # distributed cargo flow
        
        # 2. Simulate Reefer Demands
        # Reefers consume ~4 kW each. They cycle, and their aggregate load is temperature dependent.
        # Diurnal temperature cycle peaking at 15:00.
        temp_cycle = 15 + 8 * np.sin(np.pi * (hours % 24 - 8) / 12)  # 15C avg, range 7C to 23C
        reefer_base = self.reefer_count * 4.0 # baseline
        for h in range(self.total_hours):
            temp_effect = 1.0 + (temp_cycle[h] - 15) * 0.02  # 2% change per degree deviation from 15C
            reefer_demand[h] = reefer_base * temp_effect * (0.95 + 0.1 * np.random.rand())

        # 3. Simulate Base Infrastructure Demand
        # Base infrastructure demand is lower during night, peaks in day office hours (08:00 - 18:00)
        # and has a second minor peak for security lighting in the night (18:00 - 23:00)
        for h in range(self.total_hours):
            hr = h % 24
            if 8 <= hr <= 17:
                base_demand[h] = self.base_load * (1.2 + 0.2 * np.random.rand())
            elif 18 <= hr <= 22:
                base_demand[h] = self.base_load * (1.0 + 0.15 * np.random.rand())
            else:
                base_demand[h] = self.base_load * (0.7 + 0.1 * np.random.rand())

        # 4. Simulate Peak Operational Demands
        # Random heavy cargo movements, administrative spikes, grid events
        for h in range(self.total_hours):
            hr = h % 24
            # Morning shifts handover (07:00-09:00) and afternoon peaks
            if (7 <= hr <= 9) or (15 <= hr <= 17):
                peak_demand[h] = 300 * np.random.rand()
            else:
                peak_demand[h] = 50 * np.random.rand()
                
        # Total Demand before renewables
        total_demand = ops_demand + crane_demand + reefer_demand + base_demand + peak_demand
        
        # 5. Simulate Renewable Generation
        # Solar PV: Daily solar irradiance profile
        solar_gen = np.zeros(self.total_hours)
        for h in range(self.total_hours):
            hr = h % 24
            # Sun rises at 06:00, sets at 18:00
            if 6 <= hr <= 18:
                # Sine curve peaking at 12:00
                rad = np.sin(np.pi * (hr - 6) / 12)
                # Introduce cloud cover noise (simulated by random dropouts)
                cloud_noise = 0.8 + 0.2 * np.sin(2 * np.pi * h / 48) + 0.1 * np.random.randn()
                cloud_noise = np.clip(cloud_noise, 0.2, 1.0)
                solar_gen[h] = self.solar_capacity * rad * cloud_noise
            else:
                solar_gen[h] = 0.0

        # Wind Power: Wind speed following Weibull-like distribution + auto-regressive process
        wind_gen = np.zeros(self.total_hours)
        wind_speeds = np.zeros(self.total_hours)
        
        # Initial wind speed
        curr_wind = np.random.weibull(2.0) * 8.0 # typical Weibull
        for h in range(self.total_hours):
            # Auto-regressive process to simulate wind persistence
            curr_wind = 0.85 * curr_wind + 0.15 * (np.random.weibull(2.0) * 8.0) + 0.2 * np.random.randn()
            curr_wind = max(0.0, curr_wind)
            wind_speeds[h] = curr_wind
            
            # Wind Turbine Power Curve
            v_in = 3.0   # cut-in wind speed (m/s)
            v_r = 12.0   # rated wind speed (m/s)
            v_out = 25.0 # cut-out wind speed (m/s)
            
            if curr_wind < v_in or curr_wind > v_out:
                wind_gen[h] = 0.0
            elif curr_wind >= v_r:
                wind_gen[h] = self.wind_capacity
            else:
                # Cubic ramp up
                wind_gen[h] = self.wind_capacity * ((curr_wind**3 - v_in**3) / (v_r**3 - v_in**3))
                
        total_renewables = solar_gen + wind_gen
        net_demand = np.maximum(0, total_demand - total_renewables)
        
        # Package everything in a clean DataFrame
        df = pd.DataFrame({
            'Hour': hours,
            'Day': (hours // 24) + 1,
            'HourOfDay': hours % 24,
            'DayOfWeek': ((hours // 24) % 7) + 1,
            'OPS_Demand_kW': ops_demand,
            'Crane_Demand_kW': crane_demand,
            'Reefer_Demand_kW': reefer_demand,
            'Base_Demand_kW': base_demand,
            'Peak_Demand_kW': peak_demand,
            'Total_Demand_kW': total_demand,
            'Solar_Gen_kW': solar_gen,
            'Wind_Gen_kW': wind_gen,
            'Wind_Speed_ms': wind_speeds,
            'Total_Renewables_kW': total_renewables,
            'Net_Demand_kW': net_demand,
            'Vessel_Count': vessel_counts,
            'Cargo_Load_Tons': cargo_load
        })
        
        return df, vessels
