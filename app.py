import os
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import control as ctrl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# --- Page Config ---
st.set_page_config(
    page_title="Predictive Cloud Auto-Scaler | Control Systems",
    page_icon="⚡",
    layout="wide",
)

# --- Data & ML Model ---
def generate_traffic(n_steps=600, scenario="Flash Sale (Step + Ramp)", seed=42):
    rng = np.random.default_rng(seed)
    t = np.arange(n_steps)
    base = 500 + 200 * np.sin(2 * np.pi * t / 300) + 60 * np.sin(2 * np.pi * t / 75)
    noise = rng.normal(0, 18, n_steps)
    traffic = np.clip(base + noise, 100, None)

    if scenario == "Flash Sale (Step + Ramp)":
        traffic[150:280] += 480
        traffic[360:440] += np.linspace(0, 520, 80)
    elif scenario == "DDoS / Sudden Burst (Impulse)":
        traffic[180:215] += 850
        traffic[380:415] += 700
    elif scenario == "Gradual Evening Peak (Ramp)":
        traffic[120:350] += np.linspace(0, 600, 230)
        traffic[350:500] += np.linspace(600, 0, 150)

    return traffic


def build_features(traffic_series):
    df = pd.DataFrame({"traffic": traffic_series})
    for lag in [1, 2, 3, 6, 12]:
        df[f"lag_{lag}"] = df["traffic"].shift(lag)
    df["roll_mean_6"] = df["traffic"].rolling(6).mean()
    df["roll_std_6"] = df["traffic"].rolling(6).std()
    df["diff_1"] = df["traffic"].diff(1)
    return df


@st.cache_resource
def load_or_train_model():
    if os.path.exists("traffic_forecaster.pkl"):
        return joblib.load("traffic_forecaster.pkl")

    # Fallback: train a fast LightGBM model on synthetic history
    hist = generate_traffic(n_steps=2000, scenario="Flash Sale (Step + Ramp)", seed=42)
    df = build_features(hist)
    df["target"] = df["traffic"].shift(-3)
    df = df.dropna().reset_index(drop=True)

    features = [c for c in df.columns if c not in ["traffic", "target"]]
    model = lgb.LGBMRegressor(
        n_estimators=150, learning_rate=0.05, max_depth=5, verbose=-1, random_state=42
    )
    model.fit(df[features], df["target"])
    return model


# --- Control Loop Simulation ---
def run_simulation(
    traffic_series,
    ml_preds,
    mode="pid",
    Kp=0.04,
    Ki=0.008,
    Kd=0.01,
    target_cpu=70.0,
    boot_delay=3,
    cap_per_server=100.0,
):
    n = len(traffic_series)
    active_servers = np.zeros(n)
    cpu_load = np.zeros(n)

    server_pipeline = [8.0] * (boot_delay + 1)
    integral_err = 0.0
    prev_err = 0.0

    for t in range(n):
        current_servers = max(1.0, server_pipeline.pop(0))
        active_servers[t] = current_servers

        cpu = (traffic_series[t] / (current_servers * cap_per_server)) * 100.0
        cpu_load[t] = cpu
        error = cpu - target_cpu

        if mode == "rule_based":
            if cpu > 80:
                next_servers = current_servers + 2
            elif cpu < 50:
                next_servers = current_servers - 1
            else:
                next_servers = current_servers

        elif mode == "pid":
            integral_err = np.clip(integral_err + error, -200, 200)
            derivative_err = error - prev_err
            u = (Kp * error) + (Ki * integral_err) + (Kd * derivative_err)
            next_servers = current_servers + u
            prev_err = error

        elif mode == "ml_pid":
            integral_err = np.clip(integral_err + error, -200, 200)
            derivative_err = error - prev_err
            u_pid = (Kp * error) + (Ki * integral_err) + (Kd * derivative_err)

            req_servers = ml_preds[t] / (cap_per_server * (target_cpu / 100.0))
            u_ff = 0.5 * (req_servers - current_servers)
            next_servers = current_servers + u_pid + u_ff
            prev_err = error

        server_pipeline.append(max(1.0, next_servers))

    return cpu_load, active_servers


def compute_metrics(cpu, srv, target_cpu=70.0):
    overshoot = max(0.0, float(np.max(cpu) - target_cpu))
    sla_breaches = int(np.sum(cpu > 95.0))
    mae_err = float(np.mean(np.abs(cpu - target_cpu)))
    cost_inr = float(np.sum(srv) * 0.5)  # ₹0.50 per server-step
    return overshoot, sla_breaches, mae_err, cost_inr


# --- Sidebar Controls ---
st.sidebar.header("Control Center")

scenario = st.sidebar.selectbox(
    "Traffic Disturbance Scenario",
    [
        "Flash Sale (Step + Ramp)",
        "DDoS / Sudden Burst (Impulse)",
        "Gradual Evening Peak (Ramp)",
        "Normal Day",
    ],
)

target_cpu = st.sidebar.slider("Setpoint CPU (%)", 50, 85, 70)
boot_delay = st.sidebar.slider("Server Boot Delay (x10s)", 1, 6, 3)

st.sidebar.subheader("PID Gains (Unit II)")
Kp = st.sidebar.slider("Proportional (Kp)", 0.005, 0.120, 0.040, step=0.005, format="%.3f")
Ki = st.sidebar.slider("Integral (Ki)", 0.000, 0.030, 0.008, step=0.002, format="%.3f")
Kd = st.sidebar.slider("Derivative (Kd)", 0.000, 0.050, 0.010, step=0.005, format="%.3f")

enable_ml = st.sidebar.toggle("Enable ML Feedforward Predictor", value=True)

# --- Run Pipeline ---
model = load_or_train_model()
raw_traffic = generate_traffic(n_steps=500, scenario=scenario, seed=99)
df_feat = build_features(raw_traffic).dropna().reset_index(drop=True)
feature_cols = [c for c in df_feat.columns if c != "traffic"]

traffic = df_feat["traffic"].values
ml_preds = model.predict(df_feat[feature_cols])

cpu_rule, srv_rule = run_simulation(traffic, ml_preds, "rule_based", Kp, Ki, Kd, target_cpu, boot_delay)
cpu_pid, srv_pid = run_simulation(traffic, ml_preds, "pid", Kp, Ki, Kd, target_cpu, boot_delay)
cpu_ml, srv_ml = run_simulation(traffic, ml_preds, "ml_pid", Kp, Ki, Kd, target_cpu, boot_delay)

active_cpu, active_srv = (cpu_ml, srv_ml) if enable_ml else (cpu_pid, srv_pid)
active_label = "ML + PID" if enable_ml else "Pure PID"

ov_act, sla_act, err_act, cost_act = compute_metrics(active_cpu, active_srv, target_cpu)
ov_rul, sla_rul, err_rul, cost_rul = compute_metrics(cpu_rule, srv_rule, target_cpu)

# --- Header & KPI Cards ---
st.title("⚡ Predictive Cloud Server Auto-Scaler")
st.caption("Closed-Loop PID Feedback + LightGBM Feedforward Compensation for Server Provisioning")

c1, c2, c3, c4 = st.columns(4)
c1.metric(
    f"SLA Breaches ({active_label})",
    f"{sla_act}",
    delta=f"{sla_act - sla_rul} vs Rule-Based",
    delta_color="inverse",
)
c2.metric(
    "Max CPU Overshoot",
    f"{ov_act:.1f}%",
    delta=f"{ov_act - ov_rul:.1f}% vs Rule-Based",
    delta_color="inverse",
)
c3.metric(
    "Avg Tracking Error",
    f"{err_act:.1f}%",
    delta=f"{err_act - err_rul:.1f}% vs Rule-Based",
    delta_color="inverse",
)
c4.metric(
    "Estimated Cloud Cost",
    f"₹{cost_act:,.0f}",
    delta=f"₹{cost_act - cost_rul:,.0f} vs Rule-Based",
    delta_color="inverse",
)

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(
    ["📈 Live Simulation", "🤖 ML Traffic Forecast", "📐 Stability & Bode Analysis (Units III–IV)"]
)

time_axis = np.arange(len(traffic)) * 10  # seconds

with tab1:
    # Chart 1: CPU Utilization
    fig_cpu = go.Figure()
    fig_cpu.add_trace(
        go.Scatter(x=time_axis, y=cpu_rule, name="Rule-Based", line=dict(dash="dash", color="#94a3b8"))
    )
    fig_cpu.add_trace(
        go.Scatter(x=time_axis, y=cpu_pid, name="Classical PID", line=dict(color="#38bdf8"))
    )
    if enable_ml:
        fig_cpu.add_trace(
            go.Scatter(x=time_axis, y=cpu_ml, name="ML Feedforward + PID", line=dict(color="#22c55e", width=2.5))
        )
    fig_cpu.add_hline(y=target_cpu, line_dash="dot", line_color="#eab308", annotation_text=f"Setpoint ({target_cpu}%)")
    fig_cpu.add_hline(y=95, line_dash="dot", line_color="#ef4444", annotation_text="Crash Limit (95%)")
    fig_cpu.update_layout(
        title="CPU Utilization Response",
        xaxis_title="Time (seconds)",
        yaxis_title="CPU Load (%)",
        height=400,
        template="plotly_dark",
    )
    st.plotly_chart(fig_cpu, use_container_width=True)

    # Chart 2: Active Servers
    fig_srv = go.Figure()
    fig_srv.add_trace(go.Scatter(x=time_axis, y=srv_rule, name="Rule-Based", line=dict(dash="dash", color="#94a3b8")))
    fig_srv.add_trace(go.Scatter(x=time_axis, y=srv_pid, name="Classical PID", line=dict(color="#38bdf8")))
    if enable_ml:
        fig_srv.add_trace(go.Scatter(x=time_axis, y=srv_ml, name="ML Feedforward + PID", line=dict(color="#22c55e", width=2.5)))
    fig_srv.update_layout(
        title="Active Servers Provisioned",
        xaxis_title="Time (seconds)",
        yaxis_title="Server Count",
        height=340,
        template="plotly_dark",
    )
    st.plotly_chart(fig_srv, use_container_width=True)

with tab2:
    fig_ml = go.Figure()
    fig_ml.add_trace(go.Scatter(x=time_axis, y=traffic, name="Actual Traffic (req/s)", line=dict(color="#38bdf8")))
    fig_ml.add_trace(
        go.Scatter(x=time_axis, y=ml_preds, name="LightGBM 30s-Ahead Forecast", line=dict(color="#f59e0b", dash="dot"))
    )
    fig_ml.update_layout(
        title="LightGBM Traffic Forecaster (Feeds the Feedforward Compensator)",
        xaxis_title="Time (seconds)",
        yaxis_title="Requests / sec",
        height=420,
        template="plotly_dark",
    )
    st.plotly_chart(fig_ml, use_container_width=True)

with tab3:
    st.markdown(
        r"Linearized Plant Model (First-Order Plus Dead-Time): "
        r"$G(s) = \frac{1}{Ts + 1} e^{-\tau s}$"
    )
    T_const = 20.0
    tau_sec = float(boot_delay * 10)

    num_d, den_d = ctrl.pade(tau_sec, 1)
    G_plant = ctrl.tf([1], [T_const, 1]) * ctrl.tf(num_d, den_d)
    # Scale continuous PID gains for frequency-domain representation
    C_pid = ctrl.tf([Kd * 100, Kp * 20, Ki * 5], [1, 0])
    L = C_pid * G_plant

    gm, pm, wg, wp = ctrl.margin(L)
    gm_db = 20 * np.log10(gm) if (gm is not None and gm > 0 and np.isfinite(gm)) else float("inf")
    is_stable = (gm > 1 if np.isfinite(gm) else True) and (pm > 0)

    m1, m2, m3 = st.columns(3)
    m1.metric("Gain Margin", f"{gm_db:.2f} dB")
    m2.metric("Phase Margin", f"{pm:.2f}°")
    m3.metric("Closed-Loop Stability", "STABLE" if is_stable else "UNSTABLE")

    omega = np.logspace(-3, 1, 300)
    mag, phase, w = ctrl.frequency_response(L, omega)
    mag_db = 20 * np.log10(np.maximum(mag, 1e-12))
    phase_deg = np.degrees(phase)

    fig_bode = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Magnitude (dB)", "Phase (deg)"))
    fig_bode.add_trace(go.Scatter(x=w, y=mag_db, name="Magnitude", line=dict(color="#38bdf8")), row=1, col=1)
    fig_bode.add_hline(y=0, line_dash="dot", line_color="#ef4444", row=1, col=1)
    fig_bode.add_trace(go.Scatter(x=w, y=phase_deg, name="Phase", line=dict(color="#22c55e")), row=2, col=1)
    fig_bode.add_hline(y=-180, line_dash="dot", line_color="#ef4444", row=2, col=1)
    fig_bode.update_xaxes(type="log", title_text="Frequency (rad/s)", row=2, col=1)
    fig_bode.update_layout(height=500, template="plotly_dark", showlegend=False)
    st.plotly_chart(fig_bode, use_container_width=True)
