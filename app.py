import json
import numpy as np
from scipy.optimize import minimize
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="2D Self-Driving Car | Control Systems",
    page_icon="🚗",
    layout="wide",
)

# --- Session State for PID Gains ---
if "kp" not in st.session_state:
    st.session_state.kp = 0.15
if "ki" not in st.session_state:
    st.session_state.ki = 0.01
if "kd" not in st.session_state:
    st.session_state.kd = 0.25


# --- Track & Kinematic Bicycle Physics ---
def road_center_y(x, track_type):
    if track_type == "S-Curves (Highway)":
        return 6.0 * np.sin(0.035 * x) + 2.5 * np.sin(0.08 * x)
    elif track_type == "Sharp Chicane":
        return 8.0 * np.sin(0.055 * x)
    else:  # Step Lane Change (Unit II: Step Input)
        return np.where(x > 40.0, 5.0, 0.0)


def simulate_car(kp, ki, kd, speed, friction, wind, track_type, steps=400, dt=0.05):
    L = 2.5  # Wheelbase (meters)
    max_steer = np.radians(32)
    lane_half_width = 3.2

    x, y, psi = 0.0, 0.0, 0.0
    integral_err = 0.0
    prev_err = road_center_y(0.0, track_type) - y

    xs, ys, psis, steers, ctes, refs = [], [], [], [], [], []
    off_road_steps = 0

    for _ in range(steps):
        y_ref = float(road_center_y(x, track_type))
        cte = y_ref - y  # Cross-Track Error (Negative Feedback)

        integral_err = np.clip(integral_err + cte * dt, -15.0, 15.0)
        deriv_err = (cte - prev_err) / dt
        prev_err = cte

        # PID Steering Controller (Unit II)
        raw_steer = kp * cte + ki * integral_err + kd * deriv_err
        steer = float(np.clip(raw_steer, -max_steer, max_steer))

        # Road friction limits effective turning grip
        effective_steer = steer * friction

        # Kinematic update + crosswind disturbance
        x += speed * np.cos(psi) * dt
        y += (speed * np.sin(psi) + wind) * dt
        psi += (speed / L) * np.tan(effective_steer) * dt
        psi = float(np.clip(psi, -np.pi / 2.2, np.pi / 2.2))

        if abs(cte) > lane_half_width:
            off_road_steps += 1

        xs.append(round(x, 3))
        ys.append(round(y, 3))
        psis.append(round(psi, 4))
        steers.append(round(steer, 4))
        ctes.append(round(cte, 3))
        refs.append(round(y_ref, 3))

    return {
        "x": xs,
        "y": ys,
        "psi": psis,
        "steer": steers,
        "cte": ctes,
        "ref": refs,
        "off_road": off_road_steps,
        "max_err": float(np.max(np.abs(ctes))),
        "mae": float(np.mean(np.abs(ctes))),
    }


def ai_auto_tune(speed, friction, wind, track_type):
    """Minimizes ITAE + crash penalty to find optimal PID gains."""
    def loss_fn(params):
        kp, ki, kd = params
        if kp < 0 or ki < 0 or kd < 0 or kp > 2.0 or ki > 1.0 or kd > 2.0:
            return 1e6
        res = simulate_car(kp, ki, kd, speed, friction, wind, track_type, steps=300)
        cte = np.array(res["cte"])
        t = np.arange(len(cte)) * 0.05
        itae = np.mean(t * np.abs(cte))
        crash_penalty = res["off_road"] * 2.5
        return float(itae + crash_penalty)

    result = minimize(
        loss_fn,
        x0=[0.25, 0.05, 0.35],
        method="Nelder-Mead",
        options={"maxiter": 120},
    )
    kp, ki, kd = np.clip(result.x, [0.01, 0.0, 0.0], [1.5, 0.5, 1.5])
    return round(float(kp), 3), round(float(ki), 3), round(float(kd), 3)


# --- Sidebar Controls ---
st.sidebar.header("🚗 Environment & Disturbances")
track_type = st.sidebar.selectbox(
    "Track Profile",
    ["S-Curves (Highway)", "Sharp Chicane", "Step Lane Change (Unit II)"],
)
speed = st.sidebar.slider("Car Speed (m/s)", 8.0, 28.0, 16.0, step=1.0)

road_surface = st.sidebar.selectbox(
    "Road Condition (Friction)",
    ["Dry Asphalt (1.0)", "Wet / Rainy (0.65)", "Icy Road (0.35)"],
)
friction_map = {"Dry Asphalt (1.0)": 1.0, "Wet / Rainy (0.65)": 0.65, "Icy Road (0.35)": 0.35}
friction = friction_map[road_surface]

wind = st.sidebar.slider("Crosswind Disturbance (m/s)", -2.5, 2.5, 0.0, step=0.25)

st.sidebar.divider()
st.sidebar.header("🎛️ PID Steering Controller")

if st.sidebar.button("🤖 AI Auto-Tune PID", type="primary", use_container_width=True):
    with st.spinner("Optimizing PID gains for current road & wind..."):
        best_kp, best_ki, best_kd = ai_auto_tune(speed, friction, wind, track_type)
        st.session_state.kp = best_kp
        st.session_state.ki = best_ki
        st.session_state.kd = best_kd
    st.sidebar.success(f"Tuned! Kp={best_kp}, Ki={best_ki}, Kd={best_kd}")

kp = st.sidebar.slider("Proportional (Kp)", 0.01, 1.50, float(st.session_state.kp), step=0.01, key="kp")
ki = st.sidebar.slider("Integral (Ki)", 0.00, 0.50, float(st.session_state.ki), step=0.01, key="ki")
kd = st.sidebar.slider("Derivative (Kd)", 0.00, 1.50, float(st.session_state.kd), step=0.01, key="kd")

# --- Run Simulation ---
sim = simulate_car(kp, ki, kd, speed, friction, wind, track_type)

# --- Header & KPIs ---
st.title("🚗 2D Self-Driving Car: Lane-Keeping Control Simulator")
st.caption("Closed-Loop PID Steering + AI Auto-Tuner under Real-Time Road & Wind Disturbances")

c1, c2, c3, c4 = st.columns(4)
status = "✅ STABLE" if sim["off_road"] == 0 else "🚨 OFF-ROAD / UNSTABLE"
c1.metric("Driving Status", status)
c2.metric("Max Lane Deviation", f"{sim['max_err']:.2f} m", help="Road edge is at 3.20 m")
c3.metric("Avg Tracking Error", f"{sim['mae']:.2f} m")
c4.metric("Off-Road Frames", f"{sim['off_road']}")

# --- Main Tabs ---
tab_sim, tab_diag = st.tabs(["🎮 Live 2D Driving Simulator", "📊 Control Diagnostics (Unit II & III)"])

with tab_sim:
    sim_json = json.dumps({
        "x": sim["x"],
        "y": sim["y"],
        "psi": sim["psi"],
        "steer": sim["steer"],
        "cte": sim["cte"],
        "ref": sim["ref"],
        "wind": wind,
        "speed": speed,
    })

    html_canvas = f"""
    <div style="background:#0f172a; border-radius:12px; padding:12px; font-family:sans-serif; color:#e2e8f0;">
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
        <div>
          <button id="playBtn" style="background:#22c55e; color:#000; border:none; padding:6px 14px; border-radius:6px; font-weight:bold; cursor:pointer;">⏸ Pause</button>
          <button id="resetBtn" style="background:#334155; color:#fff; border:none; padding:6px 14px; border-radius:6px; margin-left:6px; cursor:pointer;">↺ Restart</button>
        </div>
        <div id="hud" style="font-size:14px; font-family:monospace;">Initializing...</div>
      </div>
      <canvas id="simCanvas" width="960" height="420" style="width:100%; background:#1e293b; border-radius:8px; display:block;"></canvas>
    </div>

    <script>
      const data = {sim_json};
      const canvas = document.getElementById("simCanvas");
      const ctx = canvas.getContext("2d");
      const hud = document.getElementById("hud");
      const playBtn = document.getElementById("playBtn");
      const resetBtn = document.getElementById("resetBtn");

      let idx = 0;
      let playing = true;
      const scale = 14; // pixels per meter
      const laneHalf = 3.2;

      playBtn.onclick = () => {{
        playing = !playing;
        playBtn.textContent = playing ? "⏸ Pause" : "▶ Play";
      }};
      resetBtn.onclick = () => {{ idx = 0; playing = true; playBtn.textContent = "⏸ Pause"; }};

      function draw() {{
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        const carX = data.x[idx];
        const carY = data.y[idx];
        const psi = data.psi[idx];
        const cte = data.cte[idx];
        const steer = data.steer[idx];

        // Camera follows carX
        const camX = carX - 18;
        const toScreenX = (wx) => (wx - camX) * scale;
        const toScreenY = (wy) => canvas.height / 2 - wy * scale;

        // Draw Road Surface
        ctx.beginPath();
        for (let i = 0; i < data.x.length; i++) {{
          const sx = toScreenX(data.x[i]);
          const sy = toScreenY(data.ref[i] + laneHalf);
          if (i === 0) ctx.moveTo(sx, sy); else ctx.lineTo(sx, sy);
        }}
        for (let i = data.x.length - 1; i >= 0; i--) {{
          const sx = toScreenX(data.x[i]);
          const sy = toScreenY(data.ref[i] - laneHalf);
          ctx.lineTo(sx, sy);
        }}
        ctx.closePath();
        ctx.fillStyle = "#334155";
        ctx.fill();

        // Draw Road Borders
        ctx.strokeStyle = "#f8fafc";
        ctx.lineWidth = 2;
        for (const offset of [laneHalf, -laneHalf]) {{
          ctx.beginPath();
          for (let i = 0; i < data.x.length; i++) {{
            const sx = toScreenX(data.x[i]);
            const sy = toScreenY(data.ref[i] + offset);
            if (i === 0) ctx.moveTo(sx, sy); else ctx.lineTo(sx, sy);
          }}
          ctx.stroke();
        }}

        // Draw Centerline (Target Setpoint)
        ctx.strokeStyle = "#eab308";
        ctx.setLineDash([10, 10]);
        ctx.beginPath();
        for (let i = 0; i < data.x.length; i++) {{
          const sx = toScreenX(data.x[i]);
          const sy = toScreenY(data.ref[i]);
          if (i === 0) ctx.moveTo(sx, sy); else ctx.lineTo(sx, sy);
        }}
        ctx.stroke();
        ctx.setLineDash([]);

        // Draw Car Trail
        ctx.strokeStyle = "#38bdf8";
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        for (let i = Math.max(0, idx - 80); i <= idx; i++) {{
          const sx = toScreenX(data.x[i]);
          const sy = toScreenY(data.y[i]);
          if (i === Math.max(0, idx - 80)) ctx.moveTo(sx, sy); else ctx.lineTo(sx, sy);
        }}
        ctx.stroke();

        // Draw Car Body
        const cx = toScreenX(carX);
        const cy = toScreenY(carY);
        const offRoad = Math.abs(cte) > laneHalf;

        ctx.save();
        ctx.translate(cx, cy);
        ctx.rotate(-psi);
        ctx.fillStyle = offRoad ? "#ef4444" : "#22c55e";
        ctx.fillRect(-18, -9, 36, 18);
        // Windshield
        ctx.fillStyle = "#0f172a";
        ctx.fillRect(4, -7, 8, 14);
        ctx.restore();

        // Update HUD
        hud.innerHTML =
          `Speed: <b>${{data.speed}} m/s</b> | ` +
          `CTE (Error): <b style="color:${{offRoad ? '#ef4444' : '#22c55e'}}">${{cte.toFixed(2)}} m</b> | ` +
          `Steer: <b>${{(steer * 57.3).toFixed(1)}}°</b> | ` +
          `Wind: <b>${{data.wind}} m/s</b>`;

        if (playing) {{
          idx = (idx + 1) % data.x.length;
        }}
        requestAnimationFrame(draw);
      }}

      draw();
    </script>
    """
    components.html(html_canvas, height=490)

with tab_diag:
    t_axis = np.arange(len(sim["cte"])) * 0.05
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_axis, y=sim["cte"], name="Cross-Track Error (m)", line=dict(color="#38bdf8")))
    fig.add_hline(y=3.2, line_dash="dot", line_color="#ef4444", annotation_text="Road Edge (+3.2m)")
    fig.add_hline(y=-3.2, line_dash="dot", line_color="#ef4444", annotation_text="Road Edge (-3.2m)")
    fig.update_layout(
        title="Closed-Loop Tracking Error Over Time (Unit II: Overshoot & Settling)",
        xaxis_title="Time (seconds)",
        yaxis_title="Error from Centerline (m)",
        template="plotly_dark",
        height=380,
    )
    st.plotly_chart(fig, use_container_width=True)
