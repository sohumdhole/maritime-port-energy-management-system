# Maritime Port Energy Management System: A Digital Twin with ML-Driven Load Forecasting and MILP Dispatch Optimisation

> **Note on Data Fidelity**: This project is a **synthetic data prototype**. All demand figures, vessel logs, weather events, and microgrid profiles are dynamically generated using discrete-event simulation principles and statistical modeling. It does not contain proprietary or real-world port operational data, making it fully open-access and ready for academic replication.

---

## 📌 Project Overview & Academic Motivation

Maritime ports are highly energy-intensive hubs representing critical interfaces between maritime shipping networks, logistics terminals, and regional power grids. As ports adopt carbon-reduction initiatives—such as **Cold Ironing (Onshore Power Supply - OPS)**, electrification of Quay Cranes (QCs) and Rubber-Tyred Gantry (RTG) cranes, and local renewable microgrids—they face extreme, multi-megawatt demand spikes and scheduling complexities.

This repository implements a **Digital Twin Prototype** coupled with an **Energy Management System (EMS)** designed to simulate and optimize a maritime port microgrid. 

```
                                  [PORT MICROGRID SYSTEM ARCHITECTURE]
                                  
   ☀️ Solar PV Arrays  ───┐
                          ├───>  [ Port AC Busbar ]  <───>  🔋 Battery Energy Storage (BESS)
   🌀 Wind Turbines   ───┘             │
                                       ├───>  🚢 Vessel Onshore Power Supply (OPS)
   ⚡ Regional Grid    <━━━━━━━━━━━━━━━┥───>  🏗️ Cargo Cranes & Handling Equipment
    (Time-of-Use)                      ├───>  ❄️ Reefer Containers (Cooling)
                                       └───>  🏢 Port Base Infrastructure Load
```

### Key Modules:
1. **Discrete-Event Simulation (Port Load Twin)**: Simulates stochastic vessel arrivals, duration-dependent auxiliary berthing loads, crane loading demand spikes, temperature-varying refrigerated container (reefer) base loads, and infrastructure.
2. **Renewable Microgeneration Models**: Simulates hourly solar PV generation (using atmospheric diurnal profiles) and wind generation (incorporating a Weibull wind distribution and standard turbine power curve).
3. **ML-Driven Load Forecasting**: Implements a `GradientBoostingRegressor` to predict next-hour electricity demand based on temporal features, historical load lags, scheduled vessels, and weather.
4. **MILP Dispatch Optimization**: Formulates a Mixed-Integer Linear Program solved via **Google OR-Tools** (with a robust fallback heuristic) to charge and discharge the battery system, shave peak demands, utilize green energy, and minimize utility costs.

---

## 🏫 Relevance to the STREAM PhD Program

This prototype has been structured to align with the core research vectors of the **STREAM Industrial Doctorate Centre (Sustainable Smart Energy Solutions for Maritime Ports)**:
* **Microgrid Integration**: Proposes a practical tool to integrate local solar, wind, and electrochemical BESS.
* **Grid Support & Peak Shaving**: Models how active port dispatch can prevent transformer congestion at the regional grid connection point.
* **Decarbonisation Pathways**: Quantifies carbon dioxide abatement and renewable energy utilization.
* **Academic Rigour**: Translates operational variables directly into standard operations research formulations.

---

## 📐 Mathematical Formulation

### 1. Port Demand Simulation
The total port load $P_{\text{demand}}(t)$ at hour $t$ is formulated as:
$$P_{\text{demand}}(t) = P_{\text{ops}}(t) + P_{\text{crane}}(t) + P_{\text{reefer}}(t) + P_{\text{base}}(t) + P_{\text{peak}}(t)$$

* **Onshore Power Supply (OPS)**:
  $$P_{\text{ops}}(t) = \sum_{v \in \mathcal{V}_t} P_{\text{ops}, v}$$
  where $\mathcal{V}_t$ is the set of vessels berthed at hour $t$, and $P_{\text{ops}, v}$ is the rated hoteling load of vessel $v$.
* **Cargo Handling**:
  $$P_{\text{crane}}(t) = \sum_{v \in \mathcal{V}_t} N_{\text{cranes}, v} \cdot P_{\text{crane, avg}} \cdot \epsilon_t$$
  where $\epsilon_t \sim U(0.8, 1.2)$ is an hourly activity coefficient.

### 2. Mixed-Integer Linear Program (MILP)
The dispatcher solves the following optimization model over a horizon $T$:

#### Objective Function
Minimize the combined operational cost ($J$):
$$\min \sum_{t=1}^{T} \left( P_{\text{grid}}(t) \cdot C_{\text{grid}}(t) \Delta t + C_{\text{wear}} \cdot (P_{\text{ch}}(t) + P_{\text{dis}}(t)) \Delta t + C_{\text{curt}} \cdot P_{\text{curt}}(t) \Delta t \right) + C_{\text{peak}} \cdot P_{\text{grid,peak}}$$

Where:
* $P_{\text{grid}}(t)$ is grid import power (kW).
* $C_{\text{grid}}(t)$ is the time-of-use tariff rate (\$/kWh).
* $C_{\text{wear}}$ is the battery degradation penalty (\$/kWh).
* $P_{\text{curt}}(t)$ is the curtailed renewable power (kW) penalized at $C_{\text{curt}}$ to prioritize green usage.
* $P_{\text{grid,peak}}$ is the monthly peak import load tracking variable penalized at $C_{\text{peak}}$ for peak-shaving.

#### Optimization Constraints
* **Microgrid Power Balance**:
  $$P_{\text{grid}}(t) + P_{\text{dis}}(t) + P_{\text{renew}}(t) - P_{\text{curt}}(t) = P_{\text{demand}}(t) + P_{\text{ch}}(t), \quad \forall t$$
* **Battery SoC Dynamics**:
  $$SoC(t) = SoC(t-1) + \eta_{\text{ch}} P_{\text{ch}}(t) \Delta t - \frac{P_{\text{dis}}(t)}{\eta_{\text{dis}}} \Delta t, \quad \forall t \ge 1$$
* **Battery Limits**:
  $$SoC_{\text{min}} \le SoC(t) \le SoC_{\text{max}}$$
  $$0 \le P_{\text{ch}}(t) \le z_{\text{ch}}(t) P_{\text{ch,max}}, \quad 0 \le P_{\text{dis}}(t) \le z_{\text{dis}}(t) P_{\text{dis,max}}$$
  $$z_{\text{ch}}(t) + z_{\text{dis}}(t) \le 1, \quad z_{\text{ch}}(t), z_{\text{dis}}(t) \in \{0, 1\}$$
  *(The binary variables $z$ prevent simultaneous charging and discharging).*
* **Peak Grid Tracking**:
  $$P_{\text{grid}}(t) \le P_{\text{grid,peak}}, \quad \forall t$$
* **Renewable Limits**:
  $$0 \le P_{\text{curt}}(t) \le P_{\text{renew}}(t)$$

---

## 📂 Repository Structure

```
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python package dependencies
├── README.md                   # Academic proposal overview (This file)
│
├── src/                        # Core algorithmic modules
│   ├── __init__.py             # Python package identifier
│   ├── simulation.py           # Port Digital Twin (Demand/Renewables simulation)
│   ├── forecasting.py          # Machine learning (Feature engineering/Training/Inference)
│   ├── optimization.py         # Operations Research (MILP solver and heuristic fallback)
│   └── visualization.py        # Chart generation (Plotly figures and dashboard styles)
│
└── data/                       # Directory containing simulated outputs
    ├── synthetic_port_demand_7days.csv
    └── optimized_dispatch_schedule.csv
```

---

## ⚙️ Installation & How to Run Locally

### 1. Prerequisites
Ensure you have Python 3.9+ installed on your system. 

### 2. Clone the Repository
```bash
git clone https://github.com/<your-username>/maritime-port-energy-management-system.git
cd maritime-port-energy-management-system
```

### 3. Create a Virtual Environment & Install Dependencies
On Windows:
```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```
On macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. Launch the Dashboard
Run the Streamlit application:
```bash
streamlit run app.py
```
This will spin up a local server and automatically open the interactive dashboard in your default browser (usually at `http://localhost:8501`).

---

## 🛠️ Verification and Code Validation

To verify that the microgrid modules are executing correctly:
1. Ensure all packages are installed.
2. Run the main script or imports via python:
```bash
python -c "import src.simulation; import src.forecasting; import src.optimization; print('✅ All core modules imported successfully.')"
```

---

## 🔮 Future Research Extensions

This prototype demonstrates a clean pipeline that can be expanded during PhD research:
1. **Stochastic Optimization**: Replace deterministic forecasts with **Scenario-Based Stochastic Programming** or **Robust Optimisation** to account for solar/wind fluctuations and vessel delays.
2. **Multi-Agent Systems**: Model vessels, crane subgrids, and BESS as autonomous agents negotiating power allocations.
3. **Hydrogen Co-Generation**: Model a port electrolyzer producing green hydrogen from excess renewables, supporting heavy terminal equipment (hydrogen straddle carriers) or fuel-cell vessel cold ironing.
4. **Vessel-to-Grid (V2G) Integration**: Investigate using massive hybrid marine vessel batteries as mobile grid storage devices while moored.

---

## 🎓 PhD Readiness Evidence

This prototype exhibits structural readiness for advanced graduate research by demonstrating:
* **System-Level Engineering**: Successfully connecting a synthetic physics-like simulator, a predictive ML model, and an optimization solver.
* **Rigorous Mathematical Formulation**: Adhering to standard linear programming dynamics.
* **Modern Tooling Proficiency**: Utilizing data science libraries (`pandas`, `numpy`, `scikit-learn`, `plotly`) alongside industrial-grade optimization wrappers (`google-ortools`).
* **Open Science & Reproducibility**: Providing structured code, inline documentation, and exportable datasets.

---

## ✍️ Author
* **Sohum Vivek Dhole**
* *MSc. in Business Analytics, Dublin Business School*
