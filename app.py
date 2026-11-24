import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal
from scipy.optimize import curve_fit
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score

st.set_page_config(page_title="AI Control Studio", page_icon="🎛️", layout="wide")
DT = 0.005  # simulation step (s)


def show(fig):
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def explain(what, try_this, look_for):
    """Small self-explaining box shown at the top of each tab."""
    st.markdown(
        f"""<div style="padding:14px 18px;border-radius:12px;background:rgba(59,130,246,0.10);
        border-left:5px solid #3b82f6;margin-bottom:14px;line-height:1.55;">
        <b>🔍 What is this?</b> {what}<br>
        <b>👆 Try this:</b> {try_this}<br>
        <b>👀 Look for:</b> {look_for}</div>""",
        unsafe_allow_html=True)


# ───────────────────────── Core control utilities ─────────────────────────
def motor_tf(R, L, J, B, K):
    """theta(s)/V(s) = K / ( s[(Js+B)(Ls+R)+K^2] )  (Kt = Kb = K)"""
    den = np.polyadd(np.polymul([J, B], [L, R]), [K * K])
    return np.array([K]), np.append(den, 0.0)


def motor_ss(R, L, J, B, K, out):
    A = np.array([[0, 1, 0], [0, -B / J, K / J], [0, -K / L, -R / L]])
    Bm = np.array([[0], [0], [1 / L]])
    C = {"Angle θ": [[1, 0, 0]], "Speed ω": [[0, 1, 0]], "Current i": [[0, 0, 1]]}[out]
    return A, Bm, np.array(C, float)


def simulate(num, den, kp, ki, kd, T, ref=1.0, dist=0.0, t_dist=None,
             umax=np.inf, N=50.0, keep_u=False):
    """Vectorised closed-loop PID simulation: many gain sets simulated at once."""
    kp, ki, kd = [np.atleast_1d(np.asarray(v, float)) for v in (kp, ki, kd)]
    A, B, C, D = signal.tf2ss(num, den)
    Ad, Bd, Cd, _, _ = signal.cont2discrete((A, B, C, D), DT, method="zoh")
    Bd, Cd, AdT = Bd[:, 0], Cd[0], Ad.T
    M, n = len(kp), A.shape[0]
    steps = int(round(T / DT)); t = np.arange(steps) * DT
    X = np.zeros((M, n)); integ = np.zeros(M); xf = np.zeros(M)
    Y = np.empty((steps, M)); U = np.empty((steps, M)) if keep_u else None
    af = 1 - np.exp(-N * DT)
    with np.errstate(all="ignore"):
        for k in range(steps):
            y = X @ Cd
            e = ref - y
            xf = xf + af * (e - xf)                      # filtered derivative
            u_un = kp * e + ki * integ + kd * N * (e - xf)
            u = np.clip(u_un, -umax, umax)
            wind = (np.abs(u_un) > umax) & (np.sign(u_un) == np.sign(e))
            integ = integ + np.where(wind, 0.0, e * DT)  # anti-windup
            d = dist if (t_dist is not None and t[k] >= t_dist) else 0.0
            X = X @ AdT + np.outer(u + d, Bd)
            Y[k] = y
            if keep_u:
                U[k] = u
    return t, Y, U


def metrics(t, Y, ref):
    dt = t[1] - t[0]
    Y = np.clip(np.nan_to_num(Y, nan=1e6, posinf=1e6, neginf=-1e6), -1e6, 1e6)
    Yn = Y / ref
    err = np.abs(1 - Yn)
    itae = (t[:, None] * err).sum(0) * dt
    mp = np.clip((Yn.max(0) - 1) * 100, 0, 500)
    out = err > 0.02
    last = len(t) - 1 - np.argmax(out[::-1], axis=0)
    ts = np.where(out.any(0), (last + 1) * dt, 0.0)

    def first(th):
        hit = Yn >= th
        return np.where(hit.any(0), np.argmax(hit, axis=0) * dt, t[-1])

    tr = first(0.9) - first(0.1)
    ess = err[-max(1, len(t) // 20):].mean(0) * 100
    return dict(Mp=mp, ts=ts, tr=tr, ess=ess, itae=itae)


def batch_eval(num, den, kp, ki, kd, T, chunk=1500):
    out = {k: [] for k in ["Mp", "ts", "tr", "ess", "itae"]}
    for s in range(0, len(kp), chunk):
        t, Y, _ = simulate(num, den, kp[s:s + chunk], ki[s:s + chunk], kd[s:s + chunk], T)
        m = metrics(t, Y, 1.0)
        for k in out:
            out[k].append(m[k])
    return {k: np.concatenate(v) for k, v in
