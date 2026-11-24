import json
import numpy as np
from scipy.optimize import minimize
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Inverted Pendulum | Control Systems Lab",
    page_icon="⚖️",
    layout="wide",
)

# --- Session State Initialization ---
defaults = {"kp": 65.0, "ki": 4.0, "kd": 14.0, "controller_on": True}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# --- Nonlinear Cart-Pole Physics (for Python AI Auto-Tuner) ---
def cartpole_derivs(state, force, M, m, L, g=9.81):
    x, x_dot, theta, theta_dot = state
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    total_m = M + m

    temp = (force + m * L * (theta_dot ** 2) * sin_t) / total_m
    theta_acc = (g * sin_t - cos_t * temp) / (L * (4.0 / 3.0 - (m * cos_t ** 2) / total_m))
    x_acc = temp - (m * L * theta_acc * cos_t) / total_m - 0.1 * x_dot
    return np.array([x_dot, x_acc, theta_dot, theta_acc])


def rk4_step(state, force, dt, M, m, L):
    k1 = cartpole_derivs(state, force, M, m, L)
    k2 = cartpole_derivs(state + 0.5 * dt * k1, force, M, m, L)
    k3 = cartpole_derivs(state + 0.5 * dt * k2, force, M, m, L)
    k4 = cartpole_derivs(state + dt * k3, force, M, m, L)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def ai_auto_tune(M, m, L):
    """Finds optimal (Kp, Ki, Kd) to stabilize the inverted pendulum."""
    def loss_fn(params):
        kp, ki, kd = params
        if kp < 5 or ki < 0 or kd < 1 or kp > 250 or ki > 50 or kd > 80:
            return 1e6

        state = np.array([0.0, 0.0, 0.18, 0.0])
        integral = 0.0
        dt = 0.02
        cost = 0.0

        for step in range(250):
            x, x_dot, theta, theta_dot = state
            if abs(theta) > 1.2 or abs(x) > 2.4:
                return 1e5 + (250 - step) * 100.0

            integral = np.clip(integral + theta * dt, -2.0, 2.0)
            force = (kp * theta + ki * integral + kd * theta_dot) + (2.0 * x + 3.5 * x_dot)
            force = float(np.clip(force, -55.0, 55.0))

            state = rk4_step(state, force, dt, M, m, L)
            t = step * dt
            cost += t * abs(theta) + 0.05 * abs(x) + 0.0005 * abs(force)

        return float(cost)

    res = minimize(
        loss_fn,
        x0=[65.0, 4.0, 14.0],
        method="Nelder-Mead",
        options={"maxiter": 140},
    )
    kp, ki, kd = np.clip(res.x, [10.0, 0.0, 2.0], [200.0, 40.0, 60.0])
    return round(float(kp), 1), round(float(ki), 1), round(float(kd), 1)


# --- Preset Callbacks ---
def set_balanced():
    st.session_state.controller_on = True
    st.session_state.kp = 65.0
    st.session_state.ki = 4.0
    st.session_state.kd = 14.0


def set_unstable_osc():
    st.session_state.controller_on = True
    st.session_state.kp = 45.0
    st.session_state.ki = 0.0
    st.session_state.kd = 0.0  # Zero damping -> unstable oscillations


def set_open_loop():
    st.session_state.controller_on = False


def run_autotune(cart_m, pole_m, pole_l):
    best_kp, best_ki, best_kd = ai_auto_tune(cart_m, pole_m, pole_l / 2.0)
    st.session_state.controller_on = True
    st.session_state.kp = best_kp
    st.session_state.ki = best_ki
    st.session_state.kd = best_kd


# --- Sidebar Controls ---
st.sidebar.header("⚡ Quick Demo Presets")
p1, p2 = st.sidebar.columns(2)
p1.button("🟢 Balanced", on_click=set_balanced, use_container_width=True)
p2.button("🔴 Unstable", on_click=set_unstable_osc, use_container_width=True)
st.sidebar.button("⚫ Open-Loop (No Control)", on_click=set_open_loop, use_container_width=True)

st.sidebar.divider()
st.sidebar.header("⚙️ Plant Parameters")
pole_length = st.sidebar.slider("Pole Length (m)", 0.6, 2.0, 1.0, step=0.1)
pole_mass = st.sidebar.slider("Pole Mass (kg)", 0.1, 1.5, 0.25, step=0.05)
cart_mass = st.sidebar.slider("Cart Mass (kg)", 0.5, 3.0, 1.0, step=0.1)

st.sidebar.divider()
st.sidebar.header("🎛️ PID Controller")
controller_on = st.sidebar.toggle("Enable PID Feedback Control", key="controller_on")

st.sidebar.button(
    "🤖 AI Auto-Tune PID",
    type="primary",
    use_container_width=True,
    on_click=run_autotune,
    args=(cart_mass, pole_mass, pole_length),
)

kp = st.sidebar.slider("Proportional (Kp)", 0.0, 200.0, step=1.0, key="kp")
ki = st.sidebar.slider("Integral (Ki)", 0.0, 40.0, step=0.5, key="ki")
kd = st.sidebar.slider("Derivative (Kd)", 0.0, 60.0, step=0.5, key="kd")

# --- Header ---
st.title("⚖️ Inverted Pendulum (Cart-Pole) Control Simulator")
st.caption("Real-Time 60 FPS Nonlinear Physics (RK4) • Closed-Loop PID Stabilization • Live Oscilloscope")

config = json.dumps({
    "M": cart_mass,
    "m": pole_mass,
    "L": pole_length / 2.0,
    "kp": kp,
    "ki": ki,
    "kd": kd,
    "enabled": controller_on,
})

html_code = f"""
<style>
  .lab-wrap {{
    background: #0b1120;
    border: 1px solid #1e293b;
    border-radius: 16px;
    padding: 18px;
    font-family: 'Segoe UI', system-ui, sans-serif;
    color: #e2e8f0;
  }}
  .toolbar {{
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    align-items: center;
    gap: 10px;
    margin-bottom: 14px;
  }}
  .btn-group {{
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
  }}
  .ctrl-btn {{
    border: none;
    padding: 9px 15px;
    border-radius: 8px;
    font-weight: 600;
    font-size: 13px;
    cursor: pointer;
    transition: transform 0.08s ease, opacity 0.15s ease;
    display: inline-flex;
    align-items: center;
    gap: 6px;
  }}
  .ctrl-btn:hover {{ opacity: 0.9; transform: translateY(-1px); }}
  .ctrl-btn:active {{ transform: scale(0.97); }}
  .btn-blue {{ background: #2563eb; color: #fff; }}
  .btn-amber {{ background: #f59e0b; color: #0f172a; }}
  .btn-purple {{ background: #7c3aed; color: #fff; }}
  .btn-slate {{ background: #334155; color: #f8fafc; }}
  .btn-red {{ background: #dc2626; color: #fff; }}
  .badge {{
    padding: 7px 16px;
    border-radius: 999px;
    font-weight: 700;
    font-size: 13px;
    letter-spacing: 0.4px;
  }}
  .telemetry {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    margin: 14px 0;
  }}
  .metric-card {{
    background: #131c31;
    border: 1px solid #1e293b;
    padding: 10px 14px;
    border-radius: 10px;
    text-align: center;
  }}
  .metric-label {{
    font-size: 11px;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.6px;
  }}
  .metric-val {{
    font-size: 22px;
    font-weight: 700;
    font-family: 'Consolas', monospace;
    margin-top: 2px;
  }}
  .chart-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
    font-size: 13px;
    font-weight: 600;
    color: #cbd5e1;
  }}
  .legend {{
    display: flex;
    gap: 16px;
    font-size: 12px;
  }}
  .dot {{
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    margin-right: 5px;
  }}
</style>

<div class="lab-wrap">
  <!-- Top Action Bar -->
  <div class="toolbar">
    <div class="btn-group">
      <button id="pushLeft" class="ctrl-btn btn-blue">⬅️ Push Left</button>
      <button id="pushRight" class="ctrl-btn btn-blue">➡️ Push Right</button>
      <button id="strongGust" class="ctrl-btn btn-amber">⚡ Strong Impulse</button>
      <button id="windBtn" class="ctrl-btn btn-purple">💨 Wind: OFF</button>
      <button id="slowBtn" class="ctrl-btn btn-slate">🐢 1.0x</button>
      <button id="pauseBtn" class="ctrl-btn btn-slate">⏸ Pause</button>
      <button id="resetBtn" class="ctrl-btn btn-red">↺ Reset</button>
    </div>
    <div id="statusBadge" class="badge" style="background:#065f46; color:#34d399;">
      ● BALANCED
    </div>
  </div>

  <!-- Main 60 FPS Simulation Canvas -->
  <canvas id="pendulumCanvas" width="1000" height="390" style="width:100%; background:#0f172a; border-radius:10px; display:block; border:1px solid #1e293b;"></canvas>

  <!-- Telemetry Cards -->
  <div class="telemetry">
    <div class="metric-card">
      <div class="metric-label">Pole Angle (θ)</div>
      <div id="valAngle" class="metric-val" style="color:#38bdf8;">0.0°</div>
