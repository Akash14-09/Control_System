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
    return {k: np.concatenate(v) for k, v in out.items()}


@st.cache_data(show_spinner=False)
def ultimate_gain(num, den):
    """Critical gain Ku (where closed-loop poles hit the jω axis) and period Pu."""
    num, den = np.array(num), np.array(den)

    def top(K):
        r = np.roots(np.polyadd(den, K * num))
        return r[np.argmax(r.real)]

    lo = None
    for K in np.logspace(-4, 7, 800):
        if top(K).real > 1e-9:
            hi = K
            break
        lo = K
    else:
        return None
    if lo is None:
        return None
    for _ in range(60):
        mid = np.sqrt(lo * hi)
        lo, hi = (lo, mid) if top(mid).real > 0 else (mid, hi)
    w = abs(top(hi).imag)
    return hi, (2 * np.pi / w if w > 1e-6 else 1.0)


def routh(coeffs):
    c = [float(x) for x in coeffs]
    while len(c) > 1 and c[0] == 0:
        c.pop(0)
    n = len(c) - 1
    cols = int(np.ceil((n + 1) / 2))
    R = np.zeros((n + 1, cols))
    R[0, :len(c[0::2])] = c[0::2]
    R[1, :len(c[1::2])] = c[1::2]
    for i in range(2, n + 1):
        if R[i - 1, 0] == 0:
            R[i - 1, 0] = 1e-9           # epsilon method
        for j in range(cols - 1):
            R[i, j] = (R[i - 1, 0] * R[i - 2, j + 1] - R[i - 2, 0] * R[i - 1, j + 1]) / R[i - 1, 0]
    col = R[:, 0]
    changes = int(np.sum(np.sign(col[:-1]) * np.sign(col[1:]) < 0))
    return R, changes


def routh_df(c):
    R, ch = routh(c)
    n = len(R) - 1
    return pd.DataFrame(R, index=[f"s^{n - i}" for i in range(n + 1)]).round(4), ch


def poly_str(c, v="s"):
    n, terms = len(c) - 1, []
    for i, a in enumerate(c):
        p = n - i
        if abs(a) < 1e-12:
            continue
        terms.append(f"{a:.4g}{v}^{{{p}}}" if p > 1 else (f"{a:.4g}{v}" if p == 1 else f"{a:.4g}"))
    return " + ".join(terms) if terms else "0"


def mat_latex(M):
    rows = r" \\ ".join(" & ".join(f"{x:.4g}" for x in r) for r in np.atleast_2d(M))
    return r"\begin{bmatrix}" + rows + r"\end{bmatrix}"


# ───────────────────────── Animated dial ─────────────────────────
def dial_fig(t, y, ref_rad):
    n = 90
    idx = np.linspace(0, len(t) - 1, n).astype(int)
    th = np.clip(np.nan_to_num(y[idx]), -10 * ref_rad, 10 * ref_rad)
    c = np.linspace(0, 2 * np.pi, 200)
    tx, ty = [], []
    for a in np.radians(np.arange(0, 360, 30)):
        tx += [0.9 * np.sin(a), np.sin(a), None]
        ty += [0.9 * np.cos(a), np.cos(a), None]
    data = [
        go.Scatter(x=np.sin(c), y=np.cos(c), mode="lines", line=dict(color="#64748b", width=3), hoverinfo="skip", showlegend=False),
        go.Scatter(x=[0, 1.15 * np.sin(ref_rad)], y=[0, 1.15 * np.cos(ref_rad)], mode="lines", line=dict(color="#22c55e", dash="dash", width=4), name="Target"),
        go.Scatter(x=[0, np.sin(th[0])], y=[0, np.cos(th[0])], mode="lines+markers", line=dict(color="#ef4444", width=7), marker=dict(size=[4, 18]), name="Motor shaft"),
        go.Scatter(x=[], y=[], mode="lines", line=dict(color="rgba(239,68,68,0.35)", width=3), name="Trail"),
        go.Scatter(x=tx, y=ty, mode="lines", line=dict(color="#94a3b8", width=2), hoverinfo="skip", showlegend=False),
    ]
    frames = [go.Frame(
        data=[go.Scatter(x=[0, np.sin(th[i])], y=[0, np.cos(th[i])]),
              go.Scatter(x=1.08 * np.sin(th[:i + 1]), y=1.08 * np.cos(th[:i + 1]))],
        traces=[2, 3],
        layout=go.Layout(title_text=f"t = {t[idx[i]]:.2f} s   |   angle = {np.degrees(th[i]):.1f}°"))
        for i in range(n)]
    dur = int(t[-1] * 1000 / n)
    fig = go.Figure(data=data, frames=frames)
    fig.update_layout(
        height=470, margin=dict(l=0, r=0, t=50, b=0), title_text="Press ▶ Play to watch the motor move",
        xaxis=dict(range=[-1.3, 1.3], visible=False),
        yaxis=dict(range=[-1.3, 1.3], visible=False, scaleanchor="x"),
        legend=dict(orientation="h", y=-0.02),
        updatemenus=[dict(type="buttons", x=0.0, y=1.12, direction="left", buttons=[
            dict(label="▶ Play", method="animate", args=[None, dict(frame=dict(duration=dur, redraw=True), fromcurrent=True, mode="immediate", transition=dict(duration=0))]),
            dict(label="⏸ Pause", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate", transition=dict(duration=0))])])])
    return fig


# ───────────────────────── AI design engine ─────────────────────────
def pid_feats(kp, ki, kd, ku, pu):
    return np.column_stack([np.log10(kp / ku), np.log10(kp / ki / pu), np.log10(kd / kp / pu + 1e-3)])


def ai_design(num, den, ku, pu, mp_max, ts_max, n_train, T):
    rng = np.random.default_rng(0)

    def sample(n):
        kp = ku * 10 ** rng.uniform(-1.6, -0.05, n)
        ti = pu * 10 ** rng.uniform(-0.4, 1.4, n)
        td = pu * 10 ** rng.uniform(-2.2, -0.4, n)
        td = np.where(rng.random(n) < 0.1, 0.0, td)
        return kp, kp / ti, kp * td

    kp, ki, kd = sample(n_train)
    m = batch_eval(num, den, kp, ki, kd, T)
    Xf = pid_feats(kp, ki, kd, ku, pu)
    Yt = np.column_stack([m["Mp"], m["ts"], np.log10(m["itae"] + 1e-6)])
    Xtr, Xte, Ytr, Yte = train_test_split(Xf, Yt, test_size=0.2, random_state=0)
    rf = RandomForestRegressor(n_estimators=150, min_samples_leaf=2, n_jobs=-1, random_state=0).fit(Xtr, Ytr)
    r2 = r2_score(Yte, rf.predict(Xte), multioutput="raw_values")

    ck, ci, cd = sample(30000)                      # ask the surrogate: what's best?
    pred = rf.predict(pid_feats(ck, ci, cd, ku, pu))
    feas = (pred[:, 0] <= 0.85 * mp_max) & (pred[:, 1] <= 0.85 * ts_max)
    if feas.sum() >= 5:
        pool = np.where(feas)[0]
        top = pool[np.argsort(pred[pool, 2])[:30]]
    else:
        viol = np.maximum(0, pred[:, 0] / mp_max - 1) + np.maximum(0, pred[:, 1] / ts_max - 1)
        top = np.argsort(viol)[:30]
    v = batch_eval(num, den, ck[top], ci[top], cd[top], T)   # verify with the real simulator
    ok = (v["Mp"] <= mp_max) & (v["ts"] <= ts_max)
    if ok.any():
        best = np.where(ok)[0][np.argmin(v["itae"][ok])]
    else:
        best = np.argmin(np.maximum(0, v["Mp"] / mp_max - 1) + np.maximum(0, v["ts"] / ts_max - 1))
    return dict(gains=(ck[top][best], ci[top][best], cd[top][best]), met_ok=bool(ok[best]),
                r2=r2, imp=rf.feature_importances_, n_feas=int(feas.sum()),
                test_true=Yte[:, 2], test_pred=rf.predict(Xte)[:, 2])


# ───────────────────────── System ID ─────────────────────────
def sec_order(t, K, z, wn):
    wd = wn * np.sqrt(1 - z ** 2)
    return K * (1 - np.exp(-z * wn * t) * (np.cos(wd * t) + z / np.sqrt(1 - z ** 2) * np.sin(wd * t)))


@st.cache_data(show_spinner=False)
def identify(seed, noise):
    rng = np.random.default_rng(seed)
    K, z, wn = rng.uniform(0.5, 5), rng.uniform(0.15, 0.8), rng.uniform(2, 12)
    t = np.linspace(0, 6 / (z * wn), 220)
    clean = sec_order(t, K, z, wn)
    y = clean + rng.normal(0, noise * K, len(t))
    tp = t[np.argmax(y)]
    p0 = [y[-20:].mean(), 0.3, np.pi / (max(tp, 1e-3) * np.sqrt(0.91))]
    bounds = ([0, 0.01, 0.1], [100, 0.99, 100])
    popt, _ = curve_fit(sec_order, t, y, p0=p0, bounds=bounds, maxfev=20000)
    fit = sec_order(t, *popt)
    res = y - fit
    boots = []
    for _ in range(80):
        yb = fit + rng.choice(res, len(res))
        try:
            boots.append(curve_fit(sec_order, t, yb, p0=popt, bounds=bounds, maxfev=5000)[0])
        except Exception:
            pass
    return dict(t=t, y=y, clean=clean, fit=fit, popt=popt, std=np.std(boots, axis=0), true=(K, z, wn))


# ───────────────────────── Stability AI ─────────────────────────
def eng_feats(a):
    a1, a2, a3, a4 = a.T
    d2 = a1 * a2 - a3
    d3 = a3 * d2 - a1 ** 2 * a4
    return np.column_stack([a1, d2, d3, a4 * d3])  # Hurwitz determinants


@st.cache_resource(show_spinner=False)
def stability_models():
    rng = np.random.default_rng(7)
    C = []
    for _ in range(12000):
        if rng.random() < 0.7:
            s, w = rng.uniform(-3, 0.8, 2), rng.uniform(0, 4, 2)
            r = [s[0] + 1j * w[0], s[0] - 1j * w[0], s[1] + 1j * w[1], s[1] - 1j * w[1]]
            C.append(np.poly(r).real)
        else:
            C.append(np.concatenate([[1], rng.normal(0, 3, 4)]))
    C = np.array(C)
    y = np.array([np.roots(c).real.max() < 0 for c in C]).astype(int)
    A = C[:, 1:]
    Xtr, Xte, ytr, yte = train_test_split(A, y, test_size=0.25, random_state=0)
    raw = RandomForestClassifier(150, n_jobs=-1, random_state=0).fit(Xtr, ytr)
    eng = RandomForestClassifier(150, n_jobs=-1, random_state=0).fit(eng_feats(Xtr), ytr)
    wrong = np.where(raw.predict(Xte) != yte)[0]
    return dict(raw=raw, eng=eng, acc_raw=accuracy_score(yte, raw.predict(Xte)),
                acc_eng=accuracy_score(yte, eng.predict(eng_feats(Xte))),
                Xte=Xte, wrong=wrong, stable_frac=y.mean())


def p_stable(model, X):
    return model.predict_proba(X)[0][list(model.classes_).index(1)]


def load_tricky():
    S = stability_models()
    if len(S["wrong"]):
        a = S["Xte"][np.random.choice(S["wrong"])]
        for k, v in zip(["a1", "a2", "a3", "a4"], a):
            st.session_state[k] = float(round(v, 4))


# ═════════════════════════ UI ═════════════════════════
st.markdown("""
<div style="padding:18px 24px;border-radius:14px;background:linear-gradient(90deg,#0f172a,#1e3a8a,#0e7490);color:white;">
<h1 style="margin:0;">🎛️ AI Control Studio</h1>
<p style="margin:4px 0 0 0;opacity:.9;">A virtual motor you can control, and an AI that learns to control it better than the textbook methods.</p>
</div>""", unsafe_allow_html=True)

st.markdown("")
st.markdown("""
<div style="line-height:1.6;">
<b>The big idea:</b> a motor needs a <i>controller</i> that decides how much voltage to apply so the shaft reaches a target angle quickly,
without overshooting or shaking. This app simulates a real DC servo motor and lets you
<b>build, break, test and automatically design</b> that controller using machine learning.
</div>
""", unsafe_allow_html=True)

g = st.columns(5)
cards = [
    ("🧩", "The Motor", "See the motor's physics turned into equations, and find out when it becomes unstable."),
    ("🎮", "Control It Live", "Set a target angle, tune the controller, push the motor with a disturbance and watch it react."),
    ("🤖", "AI Designer", "Tell the AI your goals (e.g. 'under 10% overshoot'). It finds the best controller settings."),
    ("🔬", "Mystery Plant", "A hidden system produces noisy data. The AI works out what is inside the box."),
    ("🛡️", "Stability Checker", "AI predicts if a system is stable, then a mathematical test checks if it was right."),
]
for col, (ic_, ti_, tx_) in zip(g, cards):
    col.markdown(
        f"""<div style="padding:12px;border-radius:12px;background:rgba(148,163,184,0.12);height:150px;">
        <div style="font-size:26px;">{ic_}</div><b>{ti_}</b><br><span style="font-size:13px;opacity:.85;">{tx_}</span></div>""",
        unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Motor Settings")
    st.caption("These are the physical properties of the motor. Change any of them and every tab updates instantly.")
    R = st.slider("Armature resistance R (Ω)", 0.2, 5.0, 1.0, 0.1, help="Electrical resistance of the motor winding.")
    L = st.slider("Armature inductance L (H)", 0.005, 0.5, 0.05, 0.005, help="How strongly the winding resists changes in current.")
    J = st.slider("Rotor inertia J (kg·m²)", 0.005, 0.2, 0.02, 0.005, help="How heavy / hard to spin the rotor is.")
    B = st.slider("Viscous friction B (N·m·s)", 0.01, 0.5, 0.05, 0.01, help="Friction that slows the shaft down.")
    K = st.slider("Motor constant Kt = Kb", 0.1, 1.5, 0.5, 0.05, help="How much torque you get per ampere (and back-voltage per rad/s).")
    T = st.slider("Simulation time (s)", 2.0, 10.0, 5.0, 0.5)

num, den = motor_tf(R, L, J, B, K)
pkey = str((R, L, J, B, K, T))
ug = ultimate_gain(tuple(num), tuple(den))
if ug is None:
    st.error("Could not find the critical gain for this motor. Adjust the parameters.")
    st.stop()
KU, PU = ug
ZN = (0.6 * KU, 1.2 * KU / PU, 0.075 * KU * PU)

st.markdown("---")
tabs = st.tabs(["🧩 The Motor", "🎮 Control It Live", "🤖 AI Designer", "🔬 Mystery Plant", "🛡️ Stability Checker"])

# ───────── TAB 1 ─────────
with tabs[0]:
    explain(
        "The motor's electrical and mechanical behaviour written as maths, plus a tool that shows when feedback control makes it <b>unstable</b>.",
        "Drag the <i>gain</i> slider on the right to increase the controller's strength.",
        "The red ✕ marks move across the plot. When they cross into the red zone, the motor goes unstable and the status flips to ❌.")
    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.subheader("1️⃣ The motor as an equation")
        st.caption("Input = voltage applied. Output = shaft angle. This formula is called the transfer function.")
        st.latex(r"G(s)=\frac{\theta(s)}{V(s)}=\frac{K_t}{s\,[(Js+B)(Ls+R)+K_tK_b]}")
        st.latex(rf"G(s)=\frac{{{poly_str(num)}}}{{{poly_str(den)}}}")
        poles = np.roots(den)
        st.write("**Open-loop poles** (natural modes of the motor):", ", ".join(f"{p:.3f}" for p in poles))

        st.subheader("2️⃣ The motor as a state model")
        st.caption("The same motor described by three internal quantities: angle, speed and current.")
        out = st.radio("Which quantity does the sensor measure?", ["Angle θ", "Speed ω", "Current i"], horizontal=True)
        A_, B_, C_ = motor_ss(R, L, J, B, K, out)
        st.latex(r"\dot x = " + mat_latex(A_) + "x + " + mat_latex(B_) + r"v,\quad x=[\theta,\omega,i]^T")
        st.latex("y = " + mat_latex(C_) + "x")
        Qc = np.hstack([B_, A_ @ B_, A_ @ A_ @ B_])
        Qo = np.vstack([C_, C_ @ A_, C_ @ A_ @ A_])
        rc, ro = np.linalg.matrix_rank(Qc, tol=1e-8 * np.abs(Qc).max()), np.linalg.matrix_rank(Qo, tol=1e-8 * np.abs(Qo).max())
        m1, m2 = st.columns(2)
        m1.metric("Can the voltage steer every internal state?", f"{rc} / 3", "Controllable ✅" if rc == 3 else "Not controllable ❌")
        m2.metric("Can the sensor reveal every internal state?", f"{ro} / 3", "Observable ✅" if ro == 3 else "Not observable ❌")
        if ro < 3:
            st.warning("With this sensor the shaft angle cannot be reconstructed, so position control needs an angle sensor.")

    with c2:
        st.subheader("3️⃣ Stability explorer")
        st.caption("Each dot is a closed-loop pole as the gain grows. Poles in the red zone (right side) mean instability.")
        kk = st.slider("Controller gain K", 0.0, float(2 * KU), float(0.3 * KU), float(KU / 200), key="rl" + pkey)
        Ks = np.linspace(0, 2 * KU, 400)
        pts = np.concatenate([np.roots(np.polyadd(den, k * num)) for k in Ks])
        ks_rep = np.repeat(Ks, len(den) - 1)
        cur = np.roots(np.polyadd(den, kk * num))
        fig = go.Figure()
        xr = [min(pts.real.min(), -1) * 1.1, max(pts.real.max(), 1) * 1.2]
        fig.add_vrect(x0=0, x1=xr[1], fillcolor="red", opacity=0.08, line_width=0)
        fig.add_trace(go.Scatter(x=pts.real, y=pts.imag, mode="markers", marker=dict(size=3, color=ks_rep, colorscale="Viridis"), name="Path of poles"))
        fig.add_trace(go.Scatter(x=cur.real, y=cur.imag, mode="markers", marker=dict(size=15, color="#ef4444", symbol="x"), name="Poles at chosen gain"))
        fig.update_layout(height=380, xaxis_title="Real part (decay / growth)", yaxis_title="Imaginary part (oscillation)", xaxis_range=xr, margin=dict(l=0, r=0, t=10, b=0))
        show(fig)
        st.metric("Closed-loop status", "STABLE ✅" if cur.real.max() < 0 else "UNSTABLE ❌", f"Critical gain = {KU:.3f}")

        st.subheader("4️⃣ Paper-and-pencil stability test")
        st.caption("A table that predicts stability without solving for the poles. Sign changes in the first column = number of unstable poles.")
        cl = np.polyadd(den, kk * num)
        df, ch = routh_df(cl)
        st.dataframe(df)
        st.caption(f"Sign changes in first column = {ch} → {ch} unstable pole(s).")

# ───────── TAB 2 ─────────
with tabs[1]:
    explain(
        "A live simulation of the motor shaft chasing a target angle, with a controller that adjusts the voltage many times per second.",
        "Pick a target angle, press ▶ Play on the dial, then tick <i>Inject load disturbance</i> or <i>Robustness test</i>.",
        "A good controller reaches the green target quickly, with little overshoot, and recovers after a push.")
    cc = st.columns([1, 1, 1.2])
    ai_ok = st.session_state.get("ai", {}).get("pkey") == pkey
    opts = ["Manual", "Ziegler–Nichols (textbook)"] + (["AI Designer"] if ai_ok else [])
    src = cc[0].radio("Where do the controller settings come from?", opts)
    if not ai_ok:
        cc[0].caption("Train the AI in the next tab to unlock the AI option here.")
    ref_deg = cc[1].slider("Target angle (°)", 10, 180, 90)
    use_sat = cc[1].checkbox("Limit the supply voltage", help="Real motors cannot receive unlimited voltage.")
    umax = cc[1].number_input("Voltage limit ± (V)", 1.0, 100.0, 12.0) if use_sat else np.inf
    dist_on = cc[2].checkbox("Inject load disturbance", value=False, help="A sudden push on the shaft halfway through.")
    dist_v = cc[2].slider("Push strength (V equivalent)", 0.0, 10.0, 3.0) if dist_on else 0.0
    rob = cc[2].checkbox("Robustness test: heavier rotor", help="What if the real motor is heavier than the model?")
    jk = cc[2].slider("Inertia multiplier", 0.5, 4.0, 2.0, 0.1) if rob else 1.0

    if src == "Manual":
        s1, s2, s3 = st.columns(3)
        pk = abs(hash(pkey))
        kp = s1.slider("Kp: reacts to the current error", 0.0, float(2 * KU), float(ZN[0]), float(KU / 200), key=f"kp{pk}")
        ki = s2.slider("Ki: removes leftover error over time", 0.0, float(4 * ZN[1]), float(ZN[1]), float(4 * ZN[1] / 200), key=f"ki{pk}")
        kd = s3.slider("Kd: calms oscillation", 0.0, float(4 * ZN[2]), float(ZN[2]), float(4 * ZN[2] / 200), key=f"kd{pk}")
    elif src.startswith("Ziegler"):
        kp, ki, kd = ZN
    else:
        kp, ki, kd = st.session_state["ai"]["gains"]
    st.info(f"**Controller settings →** Kp = {kp:.3f} | Ki = {ki:.3f} | Kd = {kd:.3f}")

    ref = np.radians(ref_deg)
    t, Y, U = simulate(num, den, kp, ki, kd, T, ref=ref, dist=dist_v, t_dist=T / 2 if dist_on else None, umax=umax, keep_u=True)
    y, u = Y[:, 0], U[:, 0]
    m = metrics(t, Y, ref)
    mm = st.columns(5)
    mm[0].metric("Overshoot", f"{m['Mp'][0]:.1f} %", help="How far it goes past the target.")
    mm[1].metric("Rise time", f"{m['tr'][0]:.3f} s", help="Time to go from 10% to 90% of the target.")
    mm[2].metric("Settling time", f"{m['ts'][0]:.2f} s", help="Time until it stays within 2% of the target.")
    mm[3].metric("Final error", f"{m['ess'][0]:.2f} %", help="How far from the target it ends up.")
    mm[4].metric("Overall error score", f"{m['itae'][0]:.3g}", help="Lower is better (ITAE).")

    a1, a2 = st.columns([1, 1.2])
    with a1:
        show(dial_fig(t, y, ref))
    with a2:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35], vertical_spacing=0.06)
        fig.add_trace(go.Scatter(x=t, y=np.degrees(y), name="Shaft angle", line=dict(color="#ef4444", width=3)), 1, 1)
        fig.add_trace(go.Scatter(x=t, y=np.full_like(t, ref_deg), name="Target", line=dict(color="#22c55e", dash="dash")), 1, 1)
        if rob:
            _, Yr, _ = simulate(*motor_tf(R, L, J * jk, B, K), kp, ki, kd, T, ref=ref, dist=dist_v, t_dist=T / 2 if dist_on else None, umax=umax)
            fig.add_trace(go.Scatter(x=t, y=np.degrees(Yr[:, 0]), name=f"Rotor ×{jk} heavier", line=dict(color="#f59e0b", width=2)), 1, 1)
        fig.add_trace(go.Scatter(x=t, y=u, name="Voltage sent to motor", line=dict(color="#3b82f6")), 2, 1)
        fig.update_yaxes(title_text="Angle (°)", row=1, col=1)
        fig.update_yaxes(title_text="Voltage (V)", row=2, col=1)
        fig.update_xaxes(title_text="Time (s)", row=2, col=1)
        fig.update_layout(height=470, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=1.08))
        show(fig)
    st.caption("💡 Experiment: set Ki = 0 and add a disturbance, so the shaft never returns to the target. Then raise Ki and it will. Switch on the heavier-rotor test to see feedback keep working even when the motor changes.")

# ───────── TAB 3 ─────────
with tabs[2]:
    explain(
        "You describe how you want the motor to behave (overshoot and speed). A machine-learning model then searches thousands of possible controllers and picks the best one.",
        "Set your targets below and press the blue button. Then go to <i>Control It Live</i> and choose <i>AI Designer</i> to watch it run.",
        "The AI result (blue) is compared with the classic textbook method (orange). Usually it is smoother and faster.")
    st.markdown("""**How it works, in four steps:**
1️⃣ Simulate thousands of random controllers → 2️⃣ Train a Random Forest to predict how each one performs →
3️⃣ Let the model quickly score 30,000 new candidates → 4️⃣ Re-test the best ones in the real simulator (the AI suggests, physics confirms).""")
    c = st.columns(3)
    mp_max = c[0].slider("Maximum overshoot allowed (%)", 1, 50, 10)
    ts_max = c[1].slider("Maximum settling time (s)", 0.2, float(T), min(1.5, float(T)), 0.1)
    n_train = c[2].slider("Training simulations", 1000, 8000, 4000, 500, help="More simulations = smarter AI but slower training.")

    if st.button("🚀 Train AI & design controller", type="primary"):
        with st.spinner("Simulating motors, training the model, searching for the best controller..."):
            res = ai_design(num, den, KU, PU, mp_max, ts_max, n_train, T)
            res.update(pkey=pkey, specs=(mp_max, ts_max))
            st.session_state["ai"] = res

    ai = st.session_state.get("ai")
    if ai and ai["pkey"] == pkey:
        g_ai = ai["gains"]
        rows, curves = [], {}
        for name, g in [("Ziegler–Nichols (textbook)", ZN), ("AI Designer", g_ai)]:
            t_, Y_, _ = simulate(num, den, *g, T)
            mt = metrics(t_, Y_, 1.0)
            curves[name] = Y_[:, 0]
            rows.append({"Method": name, "Kp": round(g[0], 3), "Ki": round(g[1], 3), "Kd": round(g[2], 3),
                         "Overshoot %": round(float(mt["Mp"][0]), 2), "Rise (s)": round(float(mt["tr"][0]), 3),
                         "Settling (s)": round(float(mt["ts"][0]), 3), "Final error %": round(float(mt["ess"][0]), 3),
                         "Error score": round(float(mt["itae"][0]), 4)})
        if ai["met_ok"]:
            st.success(f"✅ The AI found a controller that meets your goals (overshoot ≤ {ai['specs'][0]}%, settling ≤ {ai['specs'][1]} s).")
        else:
            st.warning("Your goals may be too strict for this motor. Showing the closest design found.")
        st.table(pd.DataFrame(rows).set_index("Method"))

        l, r = st.columns(2)
        with l:
            fig = go.Figure()
            fig.add_hline(y=1.0, line_dash="dash", line_color="#22c55e")
            fig.add_hrect(y0=0.98, y1=1.02, fillcolor="green", opacity=0.08, line_width=0)
            for nme, col in [("Ziegler–Nichols (textbook)", "#f59e0b"), ("AI Designer", "#3b82f6")]:
                fig.add_trace(go.Scatter(x=np.arange(len(curves[nme])) * DT, y=curves[nme], name=nme, line=dict(color=col, width=3)))
            fig.update_layout(height=340, title="Response to a step command (target = 1)", xaxis_title="Time (s)", margin=dict(l=0, r=0, t=40, b=0), yaxis_range=[0, 1.8])
            show(fig)
        with r:
            fig = go.Figure(go.Bar(x=["Proportional strength", "Integral time", "Derivative time"], y=ai["imp"], marker_color=["#0ea5e9", "#8b5cf6", "#f43f5e"]))
            fig.update_layout(height=340, title="Which setting matters most to the AI?", margin=dict(l=0, r=0, t=40, b=0))
            show(fig)
        with st.expander("How accurate is the AI's prediction?"):
            q = st.columns(3)
            q[0].metric("Overshoot prediction score (R²)", f"{ai['r2'][0]:.3f}")
            q[1].metric("Settling-time prediction score (R²)", f"{ai['r2'][1]:.3f}")
            q[2].metric("Error-score prediction (R²)", f"{ai['r2'][2]:.3f}")
            fig = go.Figure(go.Scatter(x=ai["test_true"], y=ai["test_pred"], mode="markers", marker=dict(size=4, opacity=0.5)))
            fig.add_shape(type="line", x0=ai["test_true"].min(), y0=ai["test_true"].min(), x1=ai["test_true"].max(), y1=ai["test_true"].max(), line=dict(dash="dash"))
            fig.update_layout(height=320, xaxis_title="Real value (from simulator)", yaxis_title="AI prediction", margin=dict(l=0, r=0, t=10, b=0))
            show(fig)
            st.caption(f"Points close to the dashed line = accurate predictions. The AI found {ai['n_feas']} of 30,000 candidates likely to meet your goals; the top 30 were re-checked in the true simulator.")
    else:
        st.info("👆 Set your goals and press the button to start.")

# ───────── TAB 4 ─────────
with tabs[3]:
    explain(
        "A hidden machine produces <b>noisy measurements</b> when we give it a step input. The AI must work out what is inside the machine, using only the data.",
        "Increase the noise, click <i>New mystery plant</i>, then tick <i>Reveal true parameters</i> to see how close the AI got.",
        "The red curve (AI's model) should follow the grey dots. More noise means bigger uncertainty (± values).")
    if "seed" not in st.session_state:
        st.session_state["seed"] = 1
    c = st.columns([1, 1, 1])
    noise = c[0].slider("Sensor noise (% of final value)", 0.0, 15.0, 4.0) / 100
    if c[1].button("🎲 New mystery plant"):
        st.session_state["seed"] += 1
    reveal = c[2].checkbox("Reveal true parameters")
    d = identify(st.session_state["seed"], noise)
    Ke, ze, we = d["popt"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["t"], y=d["y"], mode="markers", name="Measured (noisy)", marker=dict(size=5, color="#94a3b8")))
    fig.add_trace(go.Scatter(x=d["t"], y=d["fit"], name="AI's fitted model", line=dict(color="#ef4444", width=3)))
    if reveal:
        fig.add_trace(go.Scatter(x=d["t"], y=d["clean"], name="True hidden plant", line=dict(color="#22c55e", dash="dash")))
    fig.update_layout(height=380, xaxis_title="Time (s)", yaxis_title="Output", margin=dict(l=0, r=0, t=10, b=0))
    show(fig)
    st.markdown("**Equation the AI discovered:**")
    st.latex(rf"\hat G(s)=\frac{{{Ke*we**2:.4g}}}{{s^2+{2*ze*we:.4g}\,s+{we**2:.4g}}}")
    rows = []
    for i, nm in enumerate(["DC gain K (final output per unit input)", "Damping ratio ζ (how quickly wobbles die out)", "Natural frequency ωn (how fast it wants to swing)"]):
        r_ = {"Parameter": nm, "AI estimate": round(d["popt"][i], 4), "± uncertainty": round(d["std"][i], 4)}
        if reveal:
            r_["True value"] = round(d["true"][i], 4)
            r_["Error %"] = round(abs(d["popt"][i] - d["true"][i]) / d["true"][i] * 100, 2)
        rows.append(r_)
    st.table(pd.DataFrame(rows).set_index("Parameter"))
    st.markdown("**What the discovered model predicts:**")
    mpe = np.exp(-np.pi * ze / np.sqrt(1 - ze ** 2)) * 100
    q = st.columns(3)
    q[0].metric("Overshoot", f"{mpe:.1f} %")
    q[1].metric("Settling time (2%)", f"{4 / (ze * we):.2f} s")
    q[2].metric("Time to first peak", f"{np.pi / (we * np.sqrt(1 - ze ** 2)):.2f} s")

# ───────── TAB 5 ─────────
with tabs[4]:
    explain(
        "A system is <b>stable</b> if it settles down, and <b>unstable</b> if it blows up. Here an AI guesses stability from the equation's numbers, and a proven mathematical test (the Routh table) checks the answer.",
        "Type your own coefficients, or press the orange button to load a case where the basic AI makes a mistake.",
        "Three verdicts side by side: basic AI, smarter AI, and the exact maths. The exact test is always right.")
    S = stability_models()
    q = st.columns(3)
    q[0].metric("Basic AI accuracy", f"{S['acc_raw'] * 100:.1f} %", help="Sees only the raw coefficients.")
    q[1].metric("Smarter AI accuracy", f"{S['acc_eng'] * 100:.1f} %", f"+{(S['acc_eng'] - S['acc_raw']) * 100:.1f} pts", help="Also gets Hurwitz determinants as hints from control theory.")
    q[2].metric("Training examples", "12,000 equations")
    st.caption("Both AIs are the same model. The only difference is the inputs: the smarter one is given hints from control theory. Lesson: AI works best when combined with engineering knowledge.")

    st.markdown("**System equation:** s⁴ + a₁s³ + a₂s² + a₃s + a₄ = 0")
    for k, v in zip(["a1", "a2", "a3", "a4"], [8.0, 24.0, 32.0, 16.0]):
        st.session_state.setdefault(k, v)
    ic = st.columns(5)
    a = [ic[i].number_input(f"a{i + 1}", step=0.1, format="%.4f", key=f"a{i + 1}") for i in range(4)]
    ic[4].button("🎯 Load a case the basic AI gets wrong", on_click=load_tricky)

    arr = np.array([a])
    p_raw, p_eng = p_stable(S["raw"], arr), p_stable(S["eng"], eng_feats(arr))
    coeffs = [1] + a
    df, ch = routh_df(coeffs)
    truth = np.roots(coeffs).real.max() < 0
    rr = st.columns(3)
    rr[0].metric("Basic AI says", "STABLE" if p_raw > 0.5 else "UNSTABLE", f"confidence P(stable) = {p_raw:.2f}")
    rr[1].metric("Smarter AI says", "STABLE" if p_eng > 0.5 else "UNSTABLE", f"confidence P(stable) = {p_eng:.2f}")
    rr[2].metric("Exact math (Routh test) says", "STABLE" if truth else "UNSTABLE", f"{ch} unstable pole(s)")
    if (p_raw > 0.5) != truth:
        st.error("The basic AI got this one wrong. The exact test is always right, which is why AI assists engineers but does not replace the theory.")
    l, r = st.columns(2)
    with l:
        st.markdown("**Routh table** (sign changes in the first column = unstable poles)")
        st.dataframe(df)
    with r:
        st.markdown("**Where the system's poles are** (green = safe, red = unstable)")
        rt = np.roots(coeffs)
        fig = go.Figure(go.Scatter(x=rt.real, y=rt.imag, mode="markers", marker=dict(size=14, symbol="x", color=np.where(rt.real < 0, "#22c55e", "#ef4444"))))
        fig.add_vline(x=0, line_dash="dash")
        fig.update_layout(height=280, xaxis_title="Real", yaxis_title="Imag", margin=dict(l=0, r=0, t=10, b=0))
        show(fig)
