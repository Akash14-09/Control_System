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
    st.session_state.kd = 0.0


def set_open_loop():
    st.session_state.controller_on = False


def run_autotune(cart_m, pole_m, pole_l):
    best_kp, best_ki, best_kd = ai_auto_tune(cart_m, pole_m, pole_l / 2.0)
    st.session_state.controller_on = True
    st.session_state.kp = best_kp
    st.session_state.ki = best_ki
    st.session_state.kd = best_kd


# --- Sidebar Controls ---
st.sidebar.markdown(
    "<h4 style='color:#f59e0b; font-weight:800; margin-bottom:4px;'>⚡ OPEN-LOOP VS. CLOSED-LOOP STABILITY</h4>",
    unsafe_allow_html=True,
)
p1, p2 = st.sidebar.columns(2)
p1.button("🟢 Balanced", on_click=set_balanced, use_container_width=True)
p2.button("🔴 Unstable", on_click=set_unstable_osc, use_container_width=True)
st.sidebar.button("⚫ Open-Loop (No Control)", on_click=set_open_loop, use_container_width=True)

st.sidebar.divider()
st.sidebar.markdown(
    "<h4 style='color:#38bdf8; font-weight:800; margin-bottom:4px;'>⚙️ PLANT DYNAMICS (SYSTEM PARAMETERS)</h4>",
    unsafe_allow_html=True,
)
pole_length = st.sidebar.slider("Pole Length (m)", 0.6, 2.0, 1.0, step=0.1)
pole_mass = st.sidebar.slider("Pole Mass (kg)", 0.1, 1.5, 0.25, step=0.05)
cart_mass = st.sidebar.slider("Cart Mass (kg)", 0.5, 3.0, 1.0, step=0.1)

st.sidebar.divider()
st.sidebar.markdown(
    "<h4 style='color:#22c55e; font-weight:800; margin-bottom:4px;'>🎛️ PID FEEDBACK CONTROLLER & ITAE AUTO-TUNER</h4>",
    unsafe_allow_html=True,
)
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
  .sec-head {{
    font-weight: 800;
    font-size: 13px;
    letter-spacing: 1px;
    margin-bottom: 6px;
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
    margin: 6px 0 14px 0;
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
  <!-- Section 1: Disturbance Controls -->
  <div class="sec-head" style="color:#f59e0b;">
    ⚡ EXTERNAL DISTURBANCE INJECTION (IMPULSE &amp; STEP)
  </div>
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

  <!-- Section 2: Nonlinear Plant Animation -->
  <div class="sec-head" style="color:#38bdf8;">
    ⚖️ CLOSED-LOOP NONLINEAR PLANT (INVERTED PENDULUM)
  </div>
  <canvas id="pendulumCanvas" width="1000" height="390" style="width:100%; background:#0f172a; border-radius:10px; display:block; border:1px solid #1e293b;"></canvas>

  <!-- Section 3: Feedback Sensors & Control Effort -->
  <div class="sec-head" style="color:#22c55e; margin-top:14px;">
    📡 REAL-TIME FEEDBACK SENSORS &amp; CONTROL EFFORT
  </div>
  <div class="telemetry">
    <div class="metric-card">
      <div class="metric-label">Pole Angle (θ)</div>
      <div id="valAngle" class="metric-val" style="color:#38bdf8;">0.0°</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Cart Position (x)</div>
      <div id="valPos" class="metric-val" style="color:#f8fafc;">0.00 m</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Control Force (F)</div>
      <div id="valForce" class="metric-val" style="color:#22c55e;">0.0 N</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Balance Time</div>
      <div id="valTime" class="metric-val" style="color:#f59e0b;">0.0 s</div>
    </div>
  </div>

  <!-- Section 4: Time-Domain Response Oscilloscope -->
  <div class="chart-header">
    <span style="color:#a855f7; font-weight:800; font-size:13px; letter-spacing:1px;">
      📈 TIME-DOMAIN RESPONSE &amp; STABILITY ANALYSIS
    </span>
    <div class="legend">
      <span><span class="dot" style="background:#38bdf8;"></span>Pole Angle θ (±45°)</span>
      <span><span class="dot" style="background:#22c55e;"></span>Control Force F (±55 N)</span>
      <span><span class="dot" style="background:#eab308;"></span>Setpoint (0°)</span>
    </div>
  </div>
  <canvas id="scopeCanvas" width="1000" height="220" style="width:100%; background:#0f172a; border-radius:10px; display:block; border:1px solid #1e293b;"></canvas>
</div>

<script>
  const cfg = {config};
  const canvas = document.getElementById("pendulumCanvas");
  const ctx = canvas.getContext("2d");
  const scope = document.getElementById("scopeCanvas");
  const sctx = scope.getContext("2d");

  const elAngle = document.getElementById("valAngle");
  const elPos = document.getElementById("valPos");
  const elForce = document.getElementById("valForce");
  const elTime = document.getElementById("valTime");
  const statusBadge = document.getElementById("statusBadge");

  // State: [x, x_dot, theta, theta_dot]
  let state = [0.0, 0.0, 0.12, 0.0];
  let integral = 0.0;
  let force = 0.0;
  let simTime = 0.0;
  let fallen = false;
  let paused = false;
  let slowMo = false;
  let windOn = false;
  let wheelAngle = 0.0;

  const maxHist = 360;
  let angleHist = [];
  let forceHist = [];

  const g = 9.81;
  const baseDt = 0.016;
  const trackLimit = 2.6;

  function resetSim() {{
    state = [0.0, 0.0, 0.12, 0.0];
    integral = 0.0;
    force = 0.0;
    simTime = 0.0;
    fallen = false;
    angleHist = [];
    forceHist = [];
  }}

  document.getElementById("pushLeft").onclick = () => {{ state[3] -= 1.5; }};
  document.getElementById("pushRight").onclick = () => {{ state[3] += 1.5; }};
  document.getElementById("strongGust").onclick = () => {{
    state[3] += (Math.random() > 0.5 ? 3.2 : -3.2);
  }};
  document.getElementById("windBtn").onclick = (e) => {{
    windOn = !windOn;
    e.target.textContent = windOn ? "💨 Wind: ON" : "💨 Wind: OFF";
    e.target.style.background = windOn ? "#ec4899" : "#7c3aed";
  }};
  document.getElementById("slowBtn").onclick = (e) => {{
    slowMo = !slowMo;
    e.target.textContent = slowMo ? "🐢 0.4x" : "🐢 1.0x";
  }};
  document.getElementById("pauseBtn").onclick = (e) => {{
    paused = !paused;
    e.target.textContent = paused ? "▶ Resume" : "⏸ Pause";
  }};
  document.getElementById("resetBtn").onclick = resetSim;

  function derivs(s, f) {{
    const [x, xd, th, thd] = s;
    const sinT = Math.sin(th);
    const cosT = Math.cos(th);
    const totalM = cfg.M + cfg.m;
    const windTorque = windOn ? 2.2 * Math.cos(th) : 0.0;
    const temp = (f + cfg.m * cfg.L * thd * thd * sinT) / totalM;
    const thAcc = (g * sinT - cosT * temp + windTorque) / (cfg.L * (4.0 / 3.0 - (cfg.m * cosT * cosT) / totalM));
    const xAcc = temp - (cfg.m * cfg.L * thAcc * cosT) / totalM - 0.1 * xd;
    return [xd, xAcc, thd, thAcc];
  }}

  function rk4(s, f, h) {{
    const k1 = derivs(s, f);
    const s2 = s.map((v, i) => v + 0.5 * h * k1[i]);
    const k2 = derivs(s2, f);
    const s3 = s.map((v, i) => v + 0.5 * h * k2[i]);
    const k3 = derivs(s3, f);
    const s4 = s.map((v, i) => v + h * k3[i]);
    const k4 = derivs(s4, f);
    return s.map((v, i) => v + (h / 6.0) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]));
  }}

  function stepPhysics() {{
    if (paused) return;
    const dt = slowMo ? baseDt * 0.4 : baseDt;
    const [x, xd, th, thd] = state;

    if (cfg.enabled && !fallen) {{
      integral = Math.max(-2.0, Math.min(2.0, integral + th * dt));
      const uAngle = cfg.kp * th + cfg.ki * integral + cfg.kd * thd;
      const uPos = 2.0 * x + 3.5 * xd;
      force = Math.max(-55.0, Math.min(55.0, uAngle + uPos));
    }} else {{
      force = 0.0;
    }}

    state = rk4(state, force, dt);
    wheelAngle += state[1] * dt * 8.0;

    if (state[0] > trackLimit) {{ state[0] = trackLimit; state[1] *= -0.45; }}
    if (state[0] < -trackLimit) {{ state[0] = -trackLimit; state[1] *= -0.45; }}

    if (Math.abs(state[2]) > 1.60) {{
      fallen = true;
      state[2] = Math.sign(state[2]) * 1.60;
      state[3] *= -0.25;
    }} else if (!fallen) {{
      simTime += dt;
    }}

    angleHist.push(state[2] * 57.2958);
    forceHist.push(force);
    if (angleHist.length > maxHist) {{
      angleHist.shift();
      forceHist.shift();
    }}
  }}

  function drawPendulum() {{
    const W = canvas.width;
    const H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    ctx.strokeStyle = "#162033";
    ctx.lineWidth = 1;
    for (let gx = 0; gx < W; gx += 40) {{
      ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, H); ctx.stroke();
    }}
    for (let gy = 0; gy < H; gy += 40) {{
      ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(W, gy); ctx.stroke();
    }}

    const originX = W / 2;
    const trackY = H * 0.74;
    const pxPerMeter = 165;

    if (windOn) {{
      ctx.strokeStyle = "rgba(236, 72, 153, 0.35)";
      ctx.lineWidth = 2;
      const offset = (Date.now() * 0.3) % 120;
      for (let wy = 50; wy < trackY - 20; wy += 45) {{
        for (let wx = -120; wx < W; wx += 120) {{
          ctx.beginPath();
          ctx.moveTo(wx + offset, wy);
          ctx.lineTo(wx + offset + 45, wy);
          ctx.stroke();
        }}
      }}
    }}

    const leftEdge = originX - trackLimit * pxPerMeter;
    const rightEdge = originX + trackLimit * pxPerMeter;
    ctx.strokeStyle = "#475569";
    ctx.lineWidth = 7;
    ctx.lineCap = "round";
    ctx.beginPath();
    ctx.moveTo(leftEdge - 25, trackY + 28);
    ctx.lineTo(rightEdge + 25, trackY + 28);
    ctx.stroke();

    ctx.fillStyle = "#64748b";
    ctx.font = "11px monospace";
    ctx.textAlign = "center";
    for (let m = -2; m <= 2; m++) {{
      const tx = originX + m * pxPerMeter;
      ctx.fillRect(tx - 1, trackY + 28, 2, 8);
      ctx.fillText(m + "m", tx, trackY + 50);
    }}

    ctx.strokeStyle = "rgba(234, 179, 8, 0.35)";
    ctx.setLineDash([6, 6]);
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(originX, 25);
    ctx.lineTo(originX, trackY + 25);
    ctx.stroke();
    ctx.setLineDash([]);

    const [x, xd, th, thd] = state;
    const cartX = originX + x * pxPerMeter;
    const cartW = 92;
    const cartH = 44;

    if (Math.abs(force) > 0.5 && !fallen) {{
      const arrowLen = Math.max(-130, Math.min(130, force * 2.4));
      ctx.strokeStyle = "#22c55e";
      ctx.fillStyle = "#22c55e";
      ctx.lineWidth = 5;
      const startX = cartX + (arrowLen > 0 ? cartW / 2 + 8 : -cartW / 2 - 8);
      const endX = startX + arrowLen;
      ctx.beginPath();
      ctx.moveTo(startX, trackY);
      ctx.lineTo(endX, trackY);
      ctx.stroke();

      const dir = arrowLen > 0 ? 1 : -1;
      ctx.beginPath();
      ctx.moveTo(endX, trackY);
      ctx.lineTo(endX - dir * 12, trackY - 7);
      ctx.lineTo(endX - dir * 12, trackY + 7);
      ctx.closePath();
      ctx.fill();
    }}

    ctx.fillStyle = fallen ? "#450a0a" : "#1e293b";
    ctx.strokeStyle = fallen ? "#ef4444" : "#38bdf8";
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.roundRect(cartX - cartW / 2, trackY - cartH / 2, cartW, cartH, 9);
    ctx.fill();
    ctx.stroke();

    for (const wx of [cartX - 26, cartX + 26]) {{
      const wy = trackY + cartH / 2 + 2;
      ctx.fillStyle = "#94a3b8";
      ctx.beginPath();
      ctx.arc(wx, wy, 10, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = "#0f172a";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(wx + Math.cos(wheelAngle) * 8, wy + Math.sin(wheelAngle) * 8);
      ctx.lineTo(wx - Math.cos(wheelAngle) * 8, wy - Math.sin(wheelAngle) * 8);
      ctx.stroke();
    }}

    ctx.strokeStyle = "rgba(56, 189, 248, 0.5)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    const startA = -Math.PI / 2;
    const endA = -Math.PI / 2 + th;
    ctx.arc(cartX, trackY, 42, Math.min(startA, endA), Math.max(startA, endA));
    ctx.stroke();

    const polePx = cfg.L * 2.0 * pxPerMeter;
    const tipX = cartX + Math.sin(th) * polePx;
    const tipY = trackY - Math.cos(th) * polePx;

    const deg = Math.abs(th * 57.2958);
    const poleColor = fallen ? "#ef4444" : (deg < 5 ? "#22c55e" : (deg < 20 ? "#f59e0b" : "#ef4444"));

    ctx.strokeStyle = "#e2e8f0";
    ctx.lineWidth = 9;
    ctx.lineCap = "round";
    ctx.beginPath();
    ctx.moveTo(cartX, trackY);
    ctx.lineTo(tipX, tipY);
    ctx.stroke();

    ctx.fillStyle = "#38bdf8";
    ctx.beginPath();
    ctx.arc(cartX, trackY, 7, 0, Math.PI * 2);
    ctx.fill();

    ctx.shadowColor = poleColor;
    ctx.shadowBlur = 14;
    ctx.fillStyle = poleColor;
    ctx.beginPath();
    ctx.arc(tipX, tipY, 12 + cfg.m * 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.shadowBlur = 0;

    elAngle.textContent = (th * 57.2958).toFixed(1) + "°";
    elPos.textContent = x.toFixed(2) + " m";
    elForce.textContent = force.toFixed(1) + " N";
    elTime.textContent = simTime.toFixed(1) + " s";

    if (!cfg.enabled) {{
      statusBadge.textContent = "⚫ OPEN-LOOP (CONTROLLER OFF)";
      statusBadge.style.background = "#334155";
      statusBadge.style.color = "#e2e8f0";
    }} else if (fallen) {{
      statusBadge.textContent = "✖ UNSTABLE / FALLEN";
      statusBadge.style.background = "#7f1d1d";
      statusBadge.style.color = "#fca5a5";
    }} else if (deg < 5) {{
      statusBadge.textContent = "● BALANCED";
      statusBadge.style.background = "#065f46";
      statusBadge.style.color = "#34d399";
    }} else {{
      statusBadge.textContent = "▲ RECOVERING";
      statusBadge.style.background = "#78350f";
      statusBadge.style.color = "#fcd34d";
    }}
  }}

  function drawScope() {{
    const W = scope.width;
    const H = scope.height;
    sctx.clearRect(0, 0, W, H);

    sctx.strokeStyle = "#1e293b";
    sctx.lineWidth = 1;
    for (let i = 1; i < 4; i++) {{
      const gy = (H / 4) * i;
      sctx.beginPath(); sctx.moveTo(0, gy); sctx.lineTo(W, gy); sctx.stroke();
    }}
    for (let gx = 0; gx < W; gx += 60) {{
      sctx.beginPath(); sctx.moveTo(gx, 0); sctx.lineTo(gx, H); sctx.stroke();
    }}

    sctx.strokeStyle = "rgba(234, 179, 8, 0.55)";
    sctx.setLineDash([6, 6]);
    sctx.lineWidth = 1.5;
    sctx.beginPath();
    sctx.moveTo(0, H / 2);
    sctx.lineTo(W, H / 2);
    sctx.stroke();
    sctx.setLineDash([]);

    sctx.fillStyle = "#64748b";
    sctx.font = "11px monospace";
    sctx.fillText("+45°", 8, 16);
    sctx.fillText("0° (Setpoint)", 8, H / 2 - 5);
    sctx.fillText("-45°", 8, H - 8);

    sctx.strokeStyle = "rgba(34, 197, 94, 0.75)";
    sctx.lineWidth = 2;
    sctx.beginPath();
    for (let i = 0; i < forceHist.length; i++) {{
      const px = (i / (maxHist - 1)) * W;
      const py = H / 2 - Math.max(-55, Math.min(55, forceHist[i])) * (H / 120);
      if (i === 0) sctx.moveTo(px, py); else sctx.lineTo(px, py);
    }}
    sctx.stroke();

    sctx.strokeStyle = "#38bdf8";
    sctx.lineWidth = 2.8;
    sctx.beginPath();
    for (let i = 0; i < angleHist.length; i++) {{
      const px = (i / (maxHist - 1)) * W;
      const py = H / 2 - Math.max(-45, Math.min(45, angleHist[i])) * (H / 98);
      if (i === 0) sctx.moveTo(px, py); else sctx.lineTo(px, py);
    }}
    sctx.stroke();
  }}

  function loop() {{
    stepPhysics();
    drawPendulum();
    drawScope();
    requestAnimationFrame(loop);
  }}

  loop();
</script>
"""

components.html(html_code, height=860)
