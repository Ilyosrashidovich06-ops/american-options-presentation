import base64, math, pathlib
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ── config ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="American Options Pricing",
    page_icon="",
    layout="wide",
)

BG   = "#08080f"
BG2  = "#0f0f1a"
CARD = "#13131f"
BD   = "#1e1e30"
CYAN = "#00d4ff"
LCYN = "#67e8f9"
PRP  = "#8b5cf6"
GRN  = "#10b981"
RED  = "#f87171"
GOLD = "#f59e0b"
TXT  = "#f0f0ff"
MUT  = "#6b7280"

CSS = f"""
<style>
  html, body, .stApp, .stMain, [data-testid="stAppViewContainer"],
  [data-testid="stHeader"], [data-testid="block-container"],
  section[data-testid="stSidebar"] {{ background:{BG} !important; }}
  .stApp {{ background-color:{BG} !important; }}
  #MainMenu, footer, [data-testid="stToolbar"] {{ display:none !important; }}
  body, .stMarkdown, p, li {{ color:{TXT}; font-family:'Inter',sans-serif; }}
  hr {{ border-color:{BD}; margin:2.5rem 0; }}

  .card {{
    background:{CARD}; border:1px solid {BD}; border-radius:12px;
    padding:1.6rem 2rem; margin-bottom:1rem;
  }}
  .kpi-wrap {{ display:flex; gap:1rem; flex-wrap:wrap; }}
  .kpi {{
    flex:1; min-width:160px; background:{BG2}; border:1px solid {BD};
    border-radius:12px; padding:1.2rem 1.4rem; text-align:center;
  }}
  .kpi-label {{ font-size:.78rem; color:{MUT}; text-transform:uppercase;
                letter-spacing:.08em; margin-bottom:.4rem; }}
  .kpi-value {{ font-size:2rem; font-weight:700; line-height:1; }}

  .sec-num {{
    font-size:.65rem; color:{MUT}; font-family:monospace;
    letter-spacing:.15em; margin-bottom:.3rem;
  }}
  .grad {{
    background:linear-gradient(90deg,#fff 0%,{CYAN} 60%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    background-clip:text;
    font-size:2.8rem; font-weight:800; line-height:1.15; margin:0;
  }}
  .grad-sm {{
    background:linear-gradient(90deg,#fff 0%,{CYAN} 70%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    background-clip:text;
    font-size:1.9rem; font-weight:700; line-height:1.2; margin:0 0 .5rem;
  }}
  .pill {{
    display:inline-block; border-radius:99px; padding:4px 14px;
    font-size:.78rem; font-weight:600; margin:.25rem .15rem;
  }}
  .step-row {{ display:flex; align-items:flex-start; gap:.9rem; margin:.55rem 0; }}
  .step-num {{
    background:{CYAN}18; border:1px solid {CYAN}44; border-radius:50%;
    width:28px; height:28px; min-width:28px;
    display:flex; align-items:center; justify-content:center;
    font-size:.72rem; font-weight:700; color:{CYAN};
  }}
  .step-txt {{ font-size:.9rem; color:{TXT}; padding-top:4px; }}
</style>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
"""
st.markdown(CSS, unsafe_allow_html=True)

# ── logo helper ──────────────────────────────────────────────────────────────
@st.cache_data
def logo_b64():
    p = pathlib.Path(__file__).parent / "frauas_logo.png"
    if p.exists():
        return base64.b64encode(p.read_bytes()).decode()
    return None

def logo_tag(h=70):
    b = logo_b64()
    if b:
        return (f'<img src="data:image/png;base64,{b}" height="{h}" '
                f'style="filter:brightness(0) invert(1);opacity:.85;">')
    return ""

# ── pricing functions ────────────────────────────────────────────────────────
@st.cache_data
def binomial_price(S0, K, r, sigma, T, N=200, option="put"):
    dt = T / N
    u  = math.exp(sigma * math.sqrt(dt))
    d  = 1 / u
    q  = (math.exp(r * dt) - d) / (u - d)
    disc = math.exp(-r * dt)
    prices = np.array([S0 * (u**j) * (d**(N - j)) for j in range(N + 1)])
    vals = np.maximum(K - prices, 0) if option == "put" else np.maximum(prices - K, 0)
    for _ in range(N - 1, -1, -1):
        prices = prices[:-1] / d
        hold   = disc * (q * vals[1:] + (1 - q) * vals[:-1])
        ex     = np.maximum(K - prices, 0) if option == "put" else np.maximum(prices - K, 0)
        vals   = np.maximum(hold, ex)
    return float(vals[0])


@st.cache_data
def black_scholes_put(S0, K, r, sigma, T):
    from scipy.stats import norm
    if T <= 0 or sigma <= 0:
        return max(K - S0, 0)
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)


@st.cache_data
def lsmc_put(S0, K, r, sigma, T, M=20000, N=50, seed=42):
    rng = np.random.default_rng(seed)
    dt  = T / N
    Z   = rng.standard_normal((M, N))
    S   = np.zeros((M, N + 1))
    S[:, 0] = S0
    for t in range(N):
        S[:, t + 1] = S[:, t] * np.exp(
            (r - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * Z[:, t]
        )
    payoff = np.maximum(K - S[:, -1], 0)
    for t in range(N - 1, 0, -1):
        itm = K - S[:, t] > 0
        if itm.sum() < 5:
            continue
        X    = S[itm, t]
        Y    = payoff[itm] * math.exp(-r * dt)
        A    = np.column_stack([np.ones_like(X), X, X**2])
        coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
        cont = A @ coef
        ex   = np.maximum(K - X, 0)
        idx  = np.where(itm)[0][ex > cont]
        payoff[idx] = ex[ex > cont]
    return float(np.mean(payoff) * math.exp(-r * T))


@st.cache_data
def implied_vol(S0, K, r, T, market_price, lo=0.01, hi=5.0, tol=1e-6):
    for _ in range(100):
        mid = (lo + hi) / 2
        v   = binomial_price(S0, K, r, mid, T)
        if abs(v - market_price) < tol:
            return mid
        if v < market_price:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


@st.cache_data
def simulate_paths(S0, r, sigma, T, M=1000, N=252, seed=42):
    rng = np.random.default_rng(seed)
    dt  = T / N
    Z   = rng.standard_normal((M, N))
    S   = np.zeros((M, N + 1))
    S[:, 0] = S0
    for t in range(N):
        S[:, t + 1] = S[:, t] * np.exp(
            (r - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * Z[:, t]
        )
    return S


# ── chart helpers ────────────────────────────────────────────────────────────
def lo(title=""):
    return dict(
        template="plotly_dark",
        paper_bgcolor=CARD,
        plot_bgcolor=CARD,
        font=dict(family="Inter", color=TXT, size=12),
        title=dict(text=title, font=dict(size=14, color=TXT), x=0.03, xanchor="left"),
        margin=dict(l=50, r=30, t=50, b=40),
        xaxis=dict(gridcolor=BD, zerolinecolor=BD),
        yaxis=dict(gridcolor=BD, zerolinecolor=BD),
    )


@st.cache_data
def fig_convergence():
    steps  = [5, 10, 20, 50, 100, 200, 500, 1000]
    prices = [binomial_price(100, 100, 0.05, 0.20, 1.0, N=n) for n in steps]
    bs_ref = black_scholes_put(100, 100, 0.05, 0.20, 1.0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=steps, y=prices, mode="lines+markers",
        line=dict(color=CYAN, width=2.5), marker=dict(size=7, color=CYAN),
        name="Binomial price",
    ))
    fig.add_hline(y=bs_ref, line=dict(color=GRN, dash="dash", width=1.5),
                  annotation_text=f"BS European ${bs_ref:.2f}",
                  annotation_font_color=GRN)
    fig.update_layout(**lo("Binomial Tree Convergence"),
                      xaxis_title="Steps N", yaxis_title="Put Price ($)")
    return fig, bs_ref


@st.cache_data
def fig_mc():
    S = simulate_paths(201.80, 0.037, 1.074, 93/365, M=1000, N=93)
    t = np.linspace(0, 93/365, 94)
    fig = go.Figure()
    for i in range(120):
        fig.add_trace(go.Scatter(
            x=t, y=S[i], mode="lines",
            line=dict(color="rgba(0,212,255,0.07)", width=1),
            showlegend=False, hoverinfo="skip",
        ))
    fig.add_hline(y=135, line=dict(color=RED, dash="dot", width=1.5),
                  annotation_text="Strike $135", annotation_font_color=RED)
    fig.add_hline(y=201.80, line=dict(color=GOLD, dash="dash", width=1),
                  annotation_text="S₀ $201.80", annotation_font_color=GOLD)
    fig.update_layout(**lo("GBM Simulation — SPCX"), xaxis_title="Years", yaxis_title="Price ($)")
    return fig, S


@st.cache_data
def fig_dist(S):
    finals = S[:, -1]
    below  = finals[finals < 135]
    above  = finals[finals >= 135]
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=above, nbinsx=40, name="Expire worthless",
                               marker_color=GRN + "99"))
    fig.add_trace(go.Histogram(x=below, nbinsx=40, name="In-the-money",
                               marker_color=RED + "cc"))
    fig.add_vline(x=135, line=dict(color=RED, dash="dot", width=2),
                  annotation_text="K=$135", annotation_font_color=RED)
    fig.update_layout(**lo("Terminal Price Distribution"),
                      xaxis_title="Final Price ($)", yaxis_title="Paths",
                      barmode="overlay",
                      legend=dict(orientation="h", y=1.08, x=0))
    return fig


@st.cache_data
def fig_tree_small():
    N = 3; u = 1.2; d = 1 / 1.2; S0 = 100
    nodes, edges = [], []
    for t in range(N + 1):
        for j in range(t + 1):
            px = t * 1.8
            py = (t - 2 * j) * 1.2
            S  = S0 * (u ** (t - j)) * (d ** j)
            pv = max(100 - S, 0)
            early = (t > 0) and (pv > 0) and (t < N)
            nodes.append((px, py, f"S={S:.0f}\nP={pv:.1f}", early, t, j))
            if t > 0:
                for nj in [j, j - 1]:
                    parent = next((n for n in nodes if n[4] == t - 1 and n[5] == nj), None)
                    if parent:
                        edges.append((parent[0], parent[1], px, py))

    x_lines, y_lines = [], []
    for e in edges:
        x_lines += [e[0], e[2], None]
        y_lines += [e[1], e[3], None]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_lines, y=y_lines, mode="lines",
                             line=dict(color=BD, width=1.5),
                             showlegend=False, hoverinfo="skip"))
    for n in nodes:
        clr  = RED + "cc" if n[3] else CYAN + "33"
        bclr = RED if n[3] else CYAN
        fig.add_shape(type="rect",
                      x0=n[0] - .55, y0=n[1] - .45,
                      x1=n[0] + .55, y1=n[1] + .45,
                      fillcolor=clr, line=dict(color=bclr, width=1.5))
        for i, line in enumerate(n[2].split("\n")):
            fig.add_annotation(x=n[0], y=n[1] + (0.12 if i == 0 else -0.12),
                               text=line, showarrow=False,
                               font=dict(color=TXT, size=11, family="Inter"))

    fig.update_layout(
        **lo("3-Step Binomial Tree (illustrative)"),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.8, 5.8]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=340,
    )
    return fig


@st.cache_data
def fig_price_split(am, eu, prem):
    fig = go.Figure(go.Bar(
        x=["European Put\n(Black-Scholes)", "Early Exercise\nPremium", "American Put\n(Binomial)"],
        y=[eu, prem, am],
        marker_color=[CYAN, GOLD, GRN],
        text=[f"${eu:.2f}", f"${prem:.4f}", f"${am:.2f}"],
        textposition="outside", textfont=dict(size=14, color=TXT),
        width=0.45,
    ))
    fig.update_layout(**lo("Price Decomposition — SPCX"),
                      yaxis_title="Price ($)", height=320,
                      yaxis=dict(gridcolor=BD),
                      xaxis=dict(gridcolor="rgba(0,0,0,0)"))
    return fig


# ── pre-compute key numbers ──────────────────────────────────────────────────
am_base  = binomial_price(100, 100, 0.05, 0.20, 1.0)
eu_base  = black_scholes_put(100, 100, 0.05, 0.20, 1.0)

S0_spx = 201.80; K_spx = 135; r_spx = 0.037; T_spx = 93 / 365; mkt_spx = 11.30
sig_iv   = implied_vol(S0_spx, K_spx, r_spx, T_spx, mkt_spx)
am_spx   = binomial_price(S0_spx, K_spx, r_spx, sig_iv, T_spx)
eu_spx   = black_scholes_put(S0_spx, K_spx, r_spx, sig_iv, T_spx)
prem_spx = am_spx - eu_spx
lsmc_spx = lsmc_put(S0_spx, K_spx, r_spx, sig_iv, T_spx)


# ════════════════════════════════════════════════════════════════════════════
# PAGE — single scrollable layout
# ════════════════════════════════════════════════════════════════════════════

# ── TITLE ────────────────────────────────────────────────────────────────────
col_l, col_c, col_r = st.columns([1, 3, 1])
with col_c:
    st.markdown(
        f'<div style="text-align:center;padding:3rem 0 1.5rem;">{logo_tag(h=90)}</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<p class="grad" style="text-align:center;">Pricing of American Options</p>',
                unsafe_allow_html=True)
    st.markdown(
        f'<p style="text-align:center;color:{MUT};font-size:1rem;margin-top:.5rem;">'
        f'Binomial Tree &nbsp;·&nbsp; Least-Squares Monte Carlo</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="text-align:center;margin-top:1.2rem;">'
        f'<span class="pill" style="background:{CYAN}18;color:{CYAN};border:1px solid {CYAN}44;">'
        f'Computer Based Investment Analysis</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p style="text-align:center;color:{MUT};font-size:.85rem;margin-top:.8rem;">'
        f'Ilyos Umurzakov &nbsp;1615067 &nbsp;·&nbsp; Leon Ye &nbsp;1616910<br>'
        f'Prof. Wöhrle &nbsp;·&nbsp; Prof. Müller &nbsp;·&nbsp; July 2026</p>',
        unsafe_allow_html=True,
    )

kc1, kc2, kc3 = st.columns(3)
with kc1:
    st.markdown(f"""<div class="kpi">
      <div class="kpi-label">Methods</div>
      <div class="kpi-value" style="color:{CYAN};font-size:1.4rem;">CRR + LSMC</div>
    </div>""", unsafe_allow_html=True)
with kc2:
    st.markdown(f"""<div class="kpi">
      <div class="kpi-label">Case Study</div>
      <div class="kpi-value" style="color:{GOLD};font-size:1.4rem;">SPCX / SpaceX</div>
    </div>""", unsafe_allow_html=True)
with kc3:
    st.markdown(f"""<div class="kpi">
      <div class="kpi-label">Implied Vol</div>
      <div class="kpi-value" style="color:{GRN};">{sig_iv*100:.1f}%</div>
    </div>""", unsafe_allow_html=True)

st.divider()

# ── 01 THE CHALLENGE ─────────────────────────────────────────────────────────
st.markdown('<p class="sec-num">01 / 05</p>', unsafe_allow_html=True)
st.markdown('<p class="grad-sm">The Challenge</p>', unsafe_allow_html=True)
st.markdown(
    f'<p style="color:{MUT};font-size:.95rem;max-width:680px;margin-bottom:1.4rem;">'
    f'European options have a closed-form solution (Black-Scholes, 1973). '
    f'American options add the right to <strong style="color:{TXT};">exercise early</strong> — '
    f'invalidating closed-form methods and requiring numerical pricing.</p>',
    unsafe_allow_html=True,
)

ch1, ch2 = st.columns(2)
with ch1:
    st.markdown(f"""<div class="card">
      <p style="color:{MUT};font-weight:600;margin-bottom:.9rem;font-size:.85rem;
                text-transform:uppercase;letter-spacing:.08em;">European Option</p>
      <div class="step-row"><div class="step-num">·</div>
        <div class="step-txt">Exercise only at expiry T</div></div>
      <div class="step-row"><div class="step-num">·</div>
        <div class="step-txt">Closed-form: Black-Scholes (1973)</div></div>
      <div class="step-row"><div class="step-num">·</div>
        <div class="step-txt">Simple, fast, exact</div></div>
      <div class="step-row"><div class="step-num">·</div>
        <div class="step-txt">Lower bound for American value</div></div>
    </div>""", unsafe_allow_html=True)

with ch2:
    st.markdown(f"""<div class="card" style="border-color:{CYAN}44;">
      <p style="color:{CYAN};font-weight:600;margin-bottom:.9rem;font-size:.85rem;
                text-transform:uppercase;letter-spacing:.08em;">American Option</p>
      <div class="step-row">
        <div class="step-num" style="color:{CYAN};border-color:{CYAN}44;background:{CYAN}18;">+</div>
        <div class="step-txt">Exercise at any time 0 &le; t &le; T</div></div>
      <div class="step-row">
        <div class="step-num" style="color:{CYAN};border-color:{CYAN}44;background:{CYAN}18;">+</div>
        <div class="step-txt">No closed-form — numerical methods required</div></div>
      <div class="step-row">
        <div class="step-num" style="color:{CYAN};border-color:{CYAN}44;background:{CYAN}18;">+</div>
        <div class="step-txt">Value &ge; European (early exercise premium)</div></div>
      <div class="step-row">
        <div class="step-num" style="color:{CYAN};border-color:{CYAN}44;background:{CYAN}18;">+</div>
        <div class="step-txt">Critical for deep ITM puts and dividend stocks</div></div>
    </div>""", unsafe_allow_html=True)

st.divider()

# ── 02 BINOMIAL TREE ─────────────────────────────────────────────────────────
st.markdown('<p class="sec-num">02 / 05</p>', unsafe_allow_html=True)
st.markdown('<p class="grad-sm">Method 1 — Binomial Tree (CRR)</p>', unsafe_allow_html=True)

bt1, bt2 = st.columns([3, 2])
with bt1:
    st.plotly_chart(fig_tree_small(), use_container_width=True)

with bt2:
    st.markdown(f"""<div class="card" style="margin-top:.5rem;">
      <p style="color:{CYAN};font-weight:600;margin-bottom:.9rem;font-size:.85rem;
                text-transform:uppercase;letter-spacing:.08em;">How it works</p>
      <div class="step-row">
        <div class="step-num" style="color:{CYAN};border-color:{CYAN}44;background:{CYAN}18;">1</div>
        <div class="step-txt">Build a price tree: at each node S moves up by u or down by d</div></div>
      <div class="step-row">
        <div class="step-num" style="color:{CYAN};border-color:{CYAN}44;background:{CYAN}18;">2</div>
        <div class="step-txt">Calculate terminal payoffs: max(K &minus; S, 0)</div></div>
      <div class="step-row">
        <div class="step-num" style="color:{CYAN};border-color:{CYAN}44;background:{CYAN}18;">3</div>
        <div class="step-txt">Roll back: at each node compare hold vs. exercise now</div></div>
      <div class="step-row">
        <div class="step-num" style="color:{CYAN};border-color:{CYAN}44;background:{CYAN}18;">4</div>
        <div class="step-txt"><span style="color:{RED};">Red nodes</span> = early exercise optimal</div></div>
    </div>
    <div class="kpi" style="margin-bottom:1rem;margin-top:1rem;">
      <div class="kpi-label">Base case N=200 — American put</div>
      <div class="kpi-value" style="color:{CYAN};">${am_base:.4f}</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">European benchmark (BS)</div>
      <div class="kpi-value" style="color:{MUT};font-size:1.3rem;">${eu_base:.4f}</div>
    </div>
    """, unsafe_allow_html=True)

conv_fig, _ = fig_convergence()
st.plotly_chart(conv_fig, use_container_width=True)
st.divider()

# ── 03 MONTE CARLO / LSMC ────────────────────────────────────────────────────
st.markdown('<p class="sec-num">03 / 05</p>', unsafe_allow_html=True)
st.markdown('<p class="grad-sm">Method 2 — Monte Carlo & LSMC</p>', unsafe_allow_html=True)

mc_fig, S_paths = fig_mc()
mc1, mc2 = st.columns([3, 2])
with mc1:
    st.plotly_chart(mc_fig, use_container_width=True)

with mc2:
    st.markdown(f"""<div class="card" style="margin-top:.5rem;">
      <p style="color:{PRP};font-weight:600;margin-bottom:.9rem;font-size:.85rem;
                text-transform:uppercase;letter-spacing:.08em;">Longstaff-Schwartz (2001)</p>
      <div class="step-row">
        <div class="step-num" style="color:{PRP};border-color:{PRP}44;background:{PRP}18;">1</div>
        <div class="step-txt">Simulate M paths of the underlying under Q-measure (risk-neutral)</div></div>
      <div class="step-row">
        <div class="step-num" style="color:{PRP};border-color:{PRP}44;background:{PRP}18;">2</div>
        <div class="step-txt">At each time step, regress continuation value on basis functions</div></div>
      <div class="step-row">
        <div class="step-num" style="color:{PRP};border-color:{PRP}44;background:{PRP}18;">3</div>
        <div class="step-txt">Exercise early where immediate payoff &gt; expected continuation</div></div>
      <div class="step-row">
        <div class="step-num" style="color:{PRP};border-color:{PRP}44;background:{PRP}18;">4</div>
        <div class="step-txt">Average discounted payoffs across all M paths &rarr; price</div></div>
    </div>""", unsafe_allow_html=True)

st.plotly_chart(fig_dist(S_paths), use_container_width=True)
st.divider()

# ── 04 SPACEX CASE STUDY ─────────────────────────────────────────────────────
st.markdown('<p class="sec-num">04 / 05</p>', unsafe_allow_html=True)
st.markdown('<p class="grad-sm">Case Study — SPCX (SpaceX)</p>', unsafe_allow_html=True)
st.markdown(
    f'<p style="color:{MUT};font-size:.95rem;max-width:680px;margin-bottom:1.4rem;">'
    f'SPCX is a publicly traded fund providing indirect exposure to SpaceX equity. '
    f'We price a real put option (K=$135, 93 days to expiry) and extract implied volatility '
    f'by inverting the binomial model on the observed market price of ${mkt_spx}.</p>',
    unsafe_allow_html=True,
)

kp1, kp2, kp3, kp4 = st.columns(4)
with kp1:
    st.markdown(f"""<div class="kpi">
      <div class="kpi-label">Current Price S&#8320;</div>
      <div class="kpi-value" style="color:{TXT};">$201.80</div>
    </div>""", unsafe_allow_html=True)
with kp2:
    st.markdown(f"""<div class="kpi">
      <div class="kpi-label">Implied Vol &sigma;</div>
      <div class="kpi-value" style="color:{CYAN};">{sig_iv*100:.1f}%</div>
    </div>""", unsafe_allow_html=True)
with kp3:
    st.markdown(f"""<div class="kpi">
      <div class="kpi-label">American Put (CRR)</div>
      <div class="kpi-value" style="color:{GRN};">${am_spx:.2f}</div>
    </div>""", unsafe_allow_html=True)
with kp4:
    st.markdown(f"""<div class="kpi">
      <div class="kpi-label">LSMC Price</div>
      <div class="kpi-value" style="color:{PRP};">${lsmc_spx:.2f}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

sp1, sp2 = st.columns([2, 3])
with sp1:
    st.markdown(f"""<div class="card">
      <p style="color:{GOLD};font-weight:700;margin-bottom:1rem;">Key Finding</p>
      <p style="color:{TXT};font-size:.95rem;line-height:1.6;">
        The implied volatility of <strong style="color:{CYAN};">{sig_iv*100:.1f}%</strong>
        reflects SpaceX's high uncertainty as a private company with limited price discovery.
      </p>
      <p style="color:{TXT};font-size:.95rem;line-height:1.6;margin-top:.8rem;">
        The early exercise premium is <strong style="color:{GRN};">${prem_spx:.4f}</strong>
        — nearly zero. The option is deep out of the money (S&#8320;=$201.80 vs K=$135),
        so holding is almost always better than exercising early.
      </p>
      <p style="color:{TXT};font-size:.95rem;line-height:1.6;margin-top:.8rem;">
        Both CRR and LSMC converge to the same price, confirming model consistency.
      </p>
      <div style="margin-top:1.2rem;">
        <span class="pill" style="background:{CYAN}18;color:{CYAN};border:1px solid {CYAN}33;">K = $135</span>
        <span class="pill" style="background:{MUT}18;color:{MUT};border:1px solid {MUT}33;">T = 93 days</span>
        <span class="pill" style="background:{GOLD}18;color:{GOLD};border:1px solid {GOLD}33;">r = 3.7%</span>
      </div>
    </div>""", unsafe_allow_html=True)

with sp2:
    st.plotly_chart(fig_price_split(am_spx, eu_spx, prem_spx), use_container_width=True)

st.divider()

# ── 05 TAKEAWAYS ─────────────────────────────────────────────────────────────
st.markdown('<p class="sec-num">05 / 05</p>', unsafe_allow_html=True)
st.markdown('<p class="grad-sm">Takeaways</p>', unsafe_allow_html=True)

ta1, ta2, ta3 = st.columns(3)
with ta1:
    st.markdown(f"""<div class="card" style="border-color:{CYAN}33;height:100%;">
      <p style="color:{CYAN};font-size:1.8rem;font-weight:800;margin:0;">01</p>
      <p style="color:{TXT};font-weight:600;margin:.5rem 0 .4rem;">Both Methods Agree</p>
      <p style="color:{MUT};font-size:.9rem;line-height:1.55;">
        CRR and LSMC produce consistent prices for SPCX, validating our implementation
        despite very different mathematical approaches.
      </p>
    </div>""", unsafe_allow_html=True)

with ta2:
    st.markdown(f"""<div class="card" style="border-color:{GOLD}33;height:100%;">
      <p style="color:{GOLD};font-size:1.8rem;font-weight:800;margin:0;">02</p>
      <p style="color:{TXT};font-weight:600;margin:.5rem 0 .4rem;">Implied Vol as Information</p>
      <p style="color:{MUT};font-size:.9rem;line-height:1.55;">
        A {sig_iv*100:.0f}% implied vol is exceptionally high — it encodes the market's
        uncertainty about SpaceX, not fundamental historical volatility.
      </p>
    </div>""", unsafe_allow_html=True)

with ta3:
    st.markdown(f"""<div class="card" style="border-color:{GRN}33;height:100%;">
      <p style="color:{GRN};font-size:1.8rem;font-weight:800;margin:0;">03</p>
      <p style="color:{TXT};font-weight:600;margin:.5rem 0 .4rem;">Early Exercise Is Rare Here</p>
      <p style="color:{MUT};font-size:.9rem;line-height:1.55;">
        The put is deep OTM (S&#8320;=$201.80 vs K=$135). The premium of ${prem_spx:.4f}
        confirms early exercise is almost never optimal.
      </p>
    </div>""", unsafe_allow_html=True)

# footer
st.markdown(f"""
<div style="text-align:center;padding:3rem 0 1.5rem;color:{MUT};font-size:.8rem;">
  Frankfurt University of Applied Sciences &nbsp;·&nbsp; CBIA &nbsp;·&nbsp; July 2026<br>
  Ilyos Umurzakov 1615067 &nbsp;·&nbsp; Leon Ye 1616910
</div>
""", unsafe_allow_html=True)
