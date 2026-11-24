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
    kp, ki, kd = np.clip(res.x, [10.0, 0.0, 2.0], [20
