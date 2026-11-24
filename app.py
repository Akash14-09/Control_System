import json
import numpy as np
from scipy.optimize import minimize
import streamlit as stimport numpy as np
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
        go.Scatter(x=[0, 1.15 * np.sin(ref_rad)], y=[0, 1.15 * np.cos(ref_rad)], mode="lines", line=dict(color="#22c55e", dash="dash", width=4), name="Reference"),
        go.Scatter(x=[0, np.sin(th[0])], y=[0, np.cos(th[0])], mode="lines+markers", line=dict(color="#ef4444", width=7), marker=dict(size=[4, 18]), name="Motor shaft"),
        go.Scatter(x=[], y=[], mode="lines", line=dict(color="rgba(239,68,68,0.35)", width=3), name="Trail"),
        go.Scatter(x=tx, y=ty, mode="lines", line=dict(color="#94a3b8", width=2), hoverinfo="skip", showlegend=False),
    ]
    frames = [go.Frame(
        data=[go.Scatter(x=[0, np.sin(th[i])], y=[0, np.cos(th[i])]),
              go.Scatter(x=1.08 * np.sin(th[:i + 1]), y=1.08 * np.cos(th[:i + 1]))],
        traces=[2, 3],
        layout=go.Layout(title_text=f"t = {t[idx[i]]:.2f} s   |   θ = {np.degrees(th[i]):.1f}°"))
        for i in range(n)]
    dur = int(t[-1] * 1000 / n)
    fig = go.Figure(data=data, frames=frames)
    fig.update_layout(
        height=470, margin=dict(l=0, r=0, t=50, b=0), title_text="Press ▶ Play",
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
<p style="margin:4px 0 0 0;opacity:.9;">A machine-learning design lab for classical control systems — DC servo digital twin · AI-tuned PID · System identification · Stability prediction</p>
</div>""", unsafe_allow_html=True)

with st.expander("📚 Syllabus map (what to tell the examiner)"):
    st.markdown("""
| Module | Concepts used | Unit |
|---|---|---|
| Plant & Theory | Electromechanical modelling, transfer function, state-space, Routh–Hurwitz, root locus, controllability / observability | I, III, V |
| Live Lab | Negative feedback, PID, steady-state error, disturbance rejection, sensitivity to parameter change, time-domain specs | I, II |
| AI Designer | Mp, ts, tr, ess, Ziegler–Nichols via ultimate gain (Routh/root-locus), ML surrogate model | II, III |
| System ID | Standard 2nd-order step response, ζ, ωn, bootstrap uncertainty | I, II |
| Stability AI | Routh–Hurwitz vs ML, Hurwitz determinants as engineered features | III |
""")

with st.sidebar:
    st.header("⚙️ DC Servo Motor (Plant)")
    R = st.slider("Armature resistance R (Ω)", 0.2, 5.0, 1.0, 0.1)
    L = st.slider("Armature inductance L (H)", 0.005, 0.5, 0.05, 0.005)
    J = st.slider("Rotor inertia J (kg·m²)", 0.005, 0.2, 0.02, 0.005)
    B = st.slider("Viscous friction B (N·m·s)", 0.01, 0.5, 0.05, 0.01)
    K = st.slider("Motor constant Kt = Kb", 0.1, 1.5, 0.5, 0.05)
    T = st.slider("Simulation horizon (s)", 2.0, 10.0, 5.0, 0.5)
    st.caption("Change any physical parameter and every tab re-computes.")

num, den = motor_tf(R, L, J, B, K)
pkey = str((R, L, J, B, K, T))
ug = ultimate_gain(tuple(num), tuple(den))
if ug is None:
    st.error("Could not find the ultimate gain for this plant. Adjust the parameters.")
    st.stop()
KU, PU = ug
ZN = (0.6 * KU, 1.2 * KU / PU, 0.075 * KU * PU)

tabs = st.tabs(["🧩 Plant & Theory", "🎮 Live Lab", "🤖 AI Designer", "🔬 System ID", "🛡️ Stability AI"])

# ───────── TAB 1 ─────────
with tabs[0]:
    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.subheader("Mathematical model (Unit I)")
        st.latex(r"G(s)=\frac{\theta(s)}{V(s)}=\frac{K_t}{s\,[(Js+B)(Ls+R)+K_tK_b]}")
        st.latex(rf"G(s)=\frac{{{poly_str(num)}}}{{{poly_str(den)}}}")
        poles = np.roots(den)
        st.write("**Open-loop poles:**", ", ".join(f"{p:.3f}" for p in poles))

        st.subheader("State-space model (Unit V)")
        out = st.radio("Sensor measures", ["Angle θ", "Speed ω", "Current i"], horizontal=True)
        A_, B_, C_ = motor_ss(R, L, J, B, K, out)
        st.latex(r"\dot x = " + mat_latex(A_) + "x + " + mat_latex(B_) + r"v,\quad x=[\theta,\omega,i]^T")
        st.latex("y = " + mat_latex(C_) + "x")
        Qc = np.hstack([B_, A_ @ B_, A_ @ A_ @ B_])
        Qo = np.vstack([C_, C_ @ A_, C_ @ A_ @ A_])
        rc, ro = np.linalg.matrix_rank(Qc, tol=1e-8 * np.abs(Qc).max()), np.linalg.matrix_rank(Qo, tol=1e-8 * np.abs(Qo).max())
        m1, m2 = st.columns(2)
        m1.metric("Controllability rank", f"{rc} / 3", "Controllable ✅" if rc == 3 else "Not controllable ❌")
        m2.metric("Observability rank", f"{ro} / 3", "Observable ✅" if ro == 3 else "Not observable ❌")
        if ro < 3:
            st.warning("With this sensor the angle cannot be reconstructed, which is why position control needs an angle sensor.")

    with c2:
        st.subheader("Root-locus explorer (Unit III)")
        kk = st.slider("Proportional gain K in closed loop", 0.0, float(2 * KU), float(0.3 * KU), float(KU / 200), key="rl" + pkey)
        Ks = np.linspace(0, 2 * KU, 400)
        pts = np.concatenate([np.roots(np.polyadd(den, k * num)) for k in Ks])
        ks_rep = np.repeat(Ks, len(den) - 1)
        cur = np.roots(np.polyadd(den, kk * num))
        fig = go.Figure()
        xr = [min(pts.real.min(), -1) * 1.1, max(pts.real.max(), 1) * 1.2]
        fig.add_vrect(x0=0, x1=xr[1], fillcolor="red", opacity=0.08, line_width=0)
        fig.add_trace(go.Scatter(x=pts.real, y=pts.imag, mode="markers", marker=dict(size=3, color=ks_rep, colorscale="Viridis"), name="Locus"))
        fig.add_trace(go.Scatter(x=cur.real, y=cur.imag, mode="markers", marker=dict(size=15, color="#ef4444", symbol="x"), name="Poles at K"))
        fig.update_layout(height=380, xaxis_title="Real", yaxis_title="Imag", xaxis_range=xr, margin=dict(l=0, r=0, t=10, b=0))
        show(fig)
        st.metric("Closed-loop status", "STABLE ✅" if cur.real.max() < 0 else "UNSTABLE ❌", f"Critical gain Ku = {KU:.3f}")

        st.subheader("Routh array of closed loop")
        cl = np.polyadd(den, kk * num)
        df, ch = routh_df(cl)
        st.dataframe(df)
        st.caption(f"Sign changes in first column = {ch} → {ch} poles in the right-half plane.")

# ───────── TAB 2 ─────────
with tabs[1]:
    st.subheader("Control the motor shaft in real time")
    ai_ok = st.session_state.get("ai", {}).get("pkey") == pkey
    opts = ["Manual", "Ziegler–Nichols"] + (["AI Designer"] if ai_ok else [])
    cc = st.columns([1, 1, 1.2])
    src = cc[0].radio("Gain source", opts)
    ref_deg = cc[1].slider("Target angle (°)", 10, 180, 90)
    use_sat = cc[1].checkbox("Voltage limit (saturation)")
    umax = cc[1].number_input("Limit ± (V)", 1.0, 100.0, 12.0) if use_sat else np.inf
    dist_on = cc[2].checkbox("Inject load disturbance", value=False)
    dist_v = cc[2].slider("Disturbance (V equivalent)", 0.0, 10.0, 3.0) if dist_on else 0.0
    rob = cc[2].checkbox("Robustness test: change rotor inertia")
    jk = cc[2].slider("Inertia multiplier", 0.5, 4.0, 2.0, 0.1) if rob else 1.0

    if src == "Manual":
        s1, s2, s3 = st.columns(3)
        pk = abs(hash(pkey))
        kp = s1.slider("Kp", 0.0, float(2 * KU), float(ZN[0]), float(KU / 200), key=f"kp{pk}")
        ki = s2.slider("Ki", 0.0, float(4 * ZN[1]), float(ZN[1]), float(4 * ZN[1] / 200), key=f"ki{pk}")
        kd = s3.slider("Kd", 0.0, float(4 * ZN[2]), float(ZN[2]), float(4 * ZN[2] / 200), key=f"kd{pk}")
    elif src == "Ziegler–Nichols":
        kp, ki, kd = ZN
    else:
        kp, ki, kd = st.session_state["ai"]["gains"]
    st.info(f"**Gains →** Kp = {kp:.3f} | Ki = {ki:.3f} | Kd = {kd:.3f}")

    ref = np.radians(ref_deg)
    t, Y, U = simulate(num, den, kp, ki, kd, T, ref=ref, dist=dist_v, t_dist=T / 2 if dist_on else None, umax=umax, keep_u=True)
    y, u = Y[:, 0], U[:, 0]
    m = metrics(t, Y, ref)
    mm = st.columns(5)
    mm[0].metric("Overshoot", f"{m['Mp'][0]:.1f} %")
    mm[1].metric("Rise time", f"{m['tr'][0]:.3f} s")
    mm[2].metric("Settling (2%)", f"{m['ts'][0]:.2f} s")
    mm[3].metric("Steady-state error", f"{m['ess'][0]:.2f} %")
    mm[4].metric("ITAE", f"{m['itae'][0]:.3g}")

    a1, a2 = st.columns([1, 1.2])
    with a1:
        show(dial_fig(t, y, ref))
    with a2:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35], vertical_spacing=0.06)
        fig.add_trace(go.Scatter(x=t, y=np.degrees(y), name="Shaft angle", line=dict(color="#ef4444", width=3)), 1, 1)
        fig.add_trace(go.Scatter(x=t, y=np.full_like(t, ref_deg), name="Reference", line=dict(color="#22c55e", dash="dash")), 1, 1)
        if rob:
            _, Yr, _ = simulate(*motor_tf(R, L, J * jk, B, K), kp, ki, kd, T, ref=ref, dist=dist_v, t_dist=T / 2 if dist_on else None, umax=umax)
            fig.add_trace(go.Scatter(x=t, y=np.degrees(Yr[:, 0]), name=f"Inertia ×{jk}", line=dict(color="#f59e0b", width=2)), 1, 1)
        fig.add_trace(go.Scatter(x=t, y=u, name="Control voltage", line=dict(color="#3b82f6")), 2, 1)
        fig.update_yaxes(title_text="Angle (°)", row=1, col=1)
        fig.update_yaxes(title_text="Voltage (V)", row=2, col=1)
        fig.update_xaxes(title_text="Time (s)", row=2, col=1)
        fig.update_layout(height=470, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=1.08))
        show(fig)
    st.caption("Try: set Ki = 0 and inject a disturbance (steady-state error appears). Then restore Ki. Turn on the inertia test to see how feedback reduces sensitivity to parameter change.")

# ───────── TAB 3 ─────────
with tabs[2]:
    st.subheader("🤖 Let machine learning design the controller")
    st.markdown("""**Idea:** simulating thousands of controllers is expensive. So we (1) simulate a few thousand random PID settings, (2) train a **Random Forest surrogate** that predicts *overshoot, settling time and ITAE* from the gains, (3) let the surrogate search 30,000 candidates in seconds, and (4) **verify** the best ones with the real simulator.""")
    c = st.columns(3)
    mp_max = c[0].slider("Max overshoot (%)", 1, 50, 10)
    ts_max = c[1].slider("Max settling time (s)", 0.2, float(T), min(1.5, float(T)), 0.1)
    n_train = c[2].slider("Training simulations", 1000, 8000, 4000, 500)

    if st.button("🚀 Train surrogate & design controller", type="primary"):
        with st.spinner("Simulating plants, training model, searching design space..."):
            res = ai_design(num, den, KU, PU, mp_max, ts_max, n_train, T)
            res.update(pkey=pkey, specs=(mp_max, ts_max))
            st.session_state["ai"] = res

    ai = st.session_state.get("ai")
    if ai and ai["pkey"] == pkey:
        g_ai = ai["gains"]
        rows, curves = [], {}
        for name, g in [("Ziegler–Nichols", ZN), ("AI Designer", g_ai)]:
            t_, Y_, _ = simulate(num, den, *g, T)
            mt = metrics(t_, Y_, 1.0)
            curves[name] = Y_[:, 0]
            rows.append({"Method": name, "Kp": round(g[0], 3), "Ki": round(g[1], 3), "Kd": round(g[2], 3),
                         "Overshoot %": round(float(mt["Mp"][0]), 2), "Rise (s)": round(float(mt["tr"][0]), 3),
                         "Settling (s)": round(float(mt["ts"][0]), 3), "ESS %": round(float(mt["ess"][0]), 3),
                         "ITAE": round(float(mt["itae"][0]), 4)})
        if ai["met_ok"]:
            st.success(f"✅ AI design meets specs (Mp ≤ {ai['specs'][0]}%, ts ≤ {ai['specs'][1]} s). Choose 'AI Designer' in Live Lab to watch it run.")
        else:
            st.warning("The specs may be too strict for this plant. Showing the closest design found.")
        st.table(pd.DataFrame(rows).set_index("Method"))

        l, r = st.columns(2)
        with l:
            fig = go.Figure()
            fig.add_hline(y=1.0, line_dash="dash", line_color="#22c55e")
            fig.add_hrect(y0=0.98, y1=1.02, fillcolor="green", opacity=0.08, line_width=0)
            for nme, col in [("Ziegler–Nichols", "#f59e0b"), ("AI Designer", "#3b82f6")]:
                fig.add_trace(go.Scatter(x=np.arange(len(curves[nme])) * DT, y=curves[nme], name=nme, line=dict(color=col, width=3)))
            fig.update_layout(height=340, title="Unit-step response", xaxis_title="Time (s)", margin=dict(l=0, r=0, t=40, b=0), yaxis_range=[0, 1.8])
            show(fig)
        with r:
            fig = go.Figure(go.Bar(x=["Kp (rel. to Ku)", "Integral time Ti", "Derivative time Td"], y=ai["imp"], marker_color=["#0ea5e9", "#8b5cf6", "#f43f5e"]))
            fig.update_layout(height=340, title="What the model learned (feature importance)", margin=dict(l=0, r=0, t=40, b=0))
            show(fig)
        with st.expander("How accurate is the surrogate model?"):
            q = st.columns(3)
            q[0].metric("R² overshoot", f"{ai['r2'][0]:.3f}")
            q[1].metric("R² settling time", f"{ai['r2'][1]:.3f}")
            q[2].metric("R² log(ITAE)", f"{ai['r2'][2]:.3f}")
            fig = go.Figure(go.Scatter(x=ai["test_true"], y=ai["test_pred"], mode="markers", marker=dict(size=4, opacity=0.5)))
            fig.add_shape(type="line", x0=ai["test_true"].min(), y0=ai["test_true"].min(), x1=ai["test_true"].max(), y1=ai["test_true"].max(), line=dict(dash="dash"))
            fig.update_layout(height=320, xaxis_title="True log10(ITAE)", yaxis_title="Predicted", margin=dict(l=0, r=0, t=10, b=0))
            show(fig)
            st.caption(f"The surrogate found {ai['n_feas']} of 30,000 candidates predicted to meet the specs; the top 30 were re-checked with the true simulator (ML proposes, physics verifies).")
    else:
        st.info("Set your specs and press the button.")

# ───────── TAB 4 ─────────
with tabs[3]:
    st.subheader("🔬 System identification: can ML recover an unknown plant from noisy data?")
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
    fig.add_trace(go.Scatter(x=d["t"], y=d["fit"], name="ML fitted model", line=dict(color="#ef4444", width=3)))
    if reveal:
        fig.add_trace(go.Scatter(x=d["t"], y=d["clean"], name="True plant", line=dict(color="#22c55e", dash="dash")))
    fig.update_layout(height=380, xaxis_title="Time (s)", yaxis_title="Output", margin=dict(l=0, r=0, t=10, b=0))
    show(fig)
    st.latex(rf"\hat G(s)=\frac{{{Ke*we**2:.4g}}}{{s^2+{2*ze*we:.4g}\,s+{we**2:.4g}}}")
    rows = []
    for i, nm in enumerate(["DC gain K", "Damping ratio ζ", "Natural freq ωn (rad/s)"]):
        r_ = {"Parameter": nm, "Estimated": round(d["popt"][i], 4), "± std (bootstrap)": round(d["std"][i], 4)}
        if reveal:
            r_["True"] = round(d["true"][i], 4)
            r_["Error %"] = round(abs(d["popt"][i] - d["true"][i]) / d["true"][i] * 100, 2)
        rows.append(r_)
    st.table(pd.DataFrame(rows).set_index("Parameter"))
    mpe = np.exp(-np.pi * ze / np.sqrt(1 - ze ** 2)) * 100
    q = st.columns(3)
    q[0].metric("Predicted overshoot", f"{mpe:.1f} %")
    q[1].metric("Predicted settling time (2%)", f"{4 / (ze * we):.2f} s")
    q[2].metric("Peak time", f"{np.pi / (we * np.sqrt(1 - ze ** 2)):.2f} s")
    st.caption("Mp = exp(−πζ/√(1−ζ²)), ts ≈ 4/(ζωn), tp = π/ωd: textbook formulas applied to a model that ML learned from data. Raise the noise to see the uncertainty grow.")

# ───────── TAB 5 ─────────
with tabs[4]:
    st.subheader("🛡️ Stability AI vs Routh–Hurwitz")
    S = stability_models()
    q = st.columns(3)
    q[0].metric("ML accuracy: raw coefficients", f"{S['acc_raw'] * 100:.1f} %")
    q[1].metric("ML accuracy: Hurwitz features", f"{S['acc_eng'] * 100:.1f} %", f"+{(S['acc_eng'] - S['acc_raw']) * 100:.1f} pts")
    q[2].metric("Training set", "12,000 polynomials")
    st.caption("Same Random Forest, different inputs. Giving the model Hurwitz determinants (domain knowledge) fixes its mistakes, so ML needs control theory too.")

    st.markdown("**Characteristic equation:** s⁴ + a₁s³ + a₂s² + a₃s + a₄ = 0")
    for k, v in zip(["a1", "a2", "a3", "a4"], [8.0, 24.0, 32.0, 16.0]):
        st.session_state.setdefault(k, v)
    ic = st.columns(5)
    a = [ic[i].number_input(f"a{i + 1}", step=0.1, format="%.4f", key=f"a{i + 1}") for i in range(4)]
    ic[4].button("🎯 Load a polynomial the raw ML gets wrong", on_click=load_tricky)

    arr = np.array([a])
    p_raw, p_eng = p_stable(S["raw"], arr), p_stable(S["eng"], eng_feats(arr))
    coeffs = [1] + a
    df, ch = routh_df(coeffs)
    truth = np.roots(coeffs).real.max() < 0
    rr = st.columns(3)
    rr[0].metric("ML (raw coefficients)", "STABLE" if p_raw > 0.5 else "UNSTABLE", f"P(stable) = {p_raw:.2f}")
    rr[1].metric("ML (Hurwitz features)", "STABLE" if p_eng > 0.5 else "UNSTABLE", f"P(stable) = {p_eng:.2f}")
    rr[2].metric("Routh–Hurwitz (exact)", "STABLE" if truth else "UNSTABLE", f"{ch} RHP poles")
    if (p_raw > 0.5) != truth:
        st.error("The raw-coefficient ML model got this one wrong; Routh is exact, which is why ML assists but does not replace theory.")
    l, r = st.columns(2)
    with l:
        st.dataframe(df)
    with r:
        rt = np.roots(coeffs)
        fig = go.Figure(go.Scatter(x=rt.real, y=rt.imag, mode="markers", marker=dict(size=14, symbol="x", color=np.where(rt.real < 0, "#22c55e", "#ef4444"))))
        fig.add_vline(x=0, line_dash="dash")
        fig.update_layout(height=280, xaxis_title="Real", yaxis_title="Imag", margin=dict(l=0, r=0, t=10, b=0))
        show(fig)
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
