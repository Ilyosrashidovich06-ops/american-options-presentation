"""
American Option Pricing — Binomial Tree & LSMC · SpaceX/SPCX Case Study
Computer Based Investment Analysis · Frankfurt UAS · Summer 2026
Authors: Ilyos Umurzakov · Leon Ye
"""

import base64
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from scipy.stats import norm

# ── CONFIG ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="American Option Pricing · FRA UAS",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── COLOURS ────────────────────────────────────────────────────────────────────
BG   = "#0f172a"
BG2  = "#1e293b"
BD   = "#334155"
TXT  = "#f1f5f9"
MUT  = "#94a3b8"
GOLD = "#f59e0b"
GLD2 = "#fbbf24"
BLUE = "#60a5fa"
GRN  = "#34d399"
RED  = "#f87171"

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

  section[data-testid="stSidebar"]       {{ display:none !important; }}
  button[data-testid="collapsedControl"] {{ display:none !important; }}
  html, body, [class*="css"]             {{ background:{BG} !important; font-family:'Inter',sans-serif !important; }}

  h1,h2,h3,h4,p,li,span,div {{ color:{TXT}; }}

  /* nav pill buttons */
  div[data-testid="stHorizontalBlock"] button {{
    background:{BG2} !important;
    border:1px solid {BD} !important;
    border-radius:6px !important;
    color:{MUT} !important;
    font-size:0.82rem !important;
    font-weight:600 !important;
    letter-spacing:0.04em !important;
    text-transform:uppercase !important;
    transition:all .15s !important;
  }}
  div[data-testid="stHorizontalBlock"] button:hover {{
    border-color:{GOLD} !important;
    color:{GOLD} !important;
  }}
  div[data-testid="stHorizontalBlock"] button[kind="primary"] {{
    background:{GOLD}22 !important;
    border-color:{GOLD} !important;
    color:{GOLD} !important;
  }}

  .metric {{
    background:{BG2};
    border:1px solid {BD};
    border-radius:10px;
    padding:28px 20px;
    text-align:center;
  }}
  .metric-val {{
    font-size:2.4rem;
    font-weight:800;
    color:{GOLD};
    line-height:1;
  }}
  .metric-lbl {{
    font-size:0.78rem;
    font-weight:600;
    color:{MUT};
    text-transform:uppercase;
    letter-spacing:0.08em;
    margin-top:10px;
  }}

  .step-row {{
    display:flex;
    align-items:flex-start;
    gap:20px;
    background:{BG2};
    border:1px solid {BD};
    border-radius:10px;
    padding:20px 24px;
    margin:8px 0;
  }}
  .step-num {{
    font-size:1.6rem;
    font-weight:800;
    color:{GOLD};
    min-width:36px;
    line-height:1;
    margin-top:2px;
  }}
  .step-head {{ font-size:1.05rem; font-weight:700; color:{TXT}; margin-bottom:4px; }}
  .step-body {{ font-size:0.92rem; color:{MUT}; line-height:1.6; }}

  .highlight {{
    background:{GOLD}18;
    border:1px solid {GOLD}55;
    border-radius:10px;
    padding:20px 26px;
    margin:12px 0;
    font-size:1.05rem;
    color:{TXT};
    line-height:1.7;
  }}

  .section-label {{
    font-size:0.7rem;
    font-weight:700;
    letter-spacing:0.14em;
    text-transform:uppercase;
    color:{GOLD};
    margin-bottom:6px;
  }}

  hr.divider {{
    border:none;
    border-top:1px solid {BD};
    margin:1.2rem 0;
  }}
</style>
""", unsafe_allow_html=True)

# ── LOGO HELPER ────────────────────────────────────────────────────────────────
def logo_tag(height=70):
    p = Path(__file__).parent / "frauas_logo.png"
    if p.exists():
        b64 = base64.b64encode(p.read_bytes()).decode()
        return (f'<div style="background:white;display:inline-block;'
                f'padding:10px 18px;border-radius:10px;">'
                f'<img src="data:image/png;base64,{b64}" style="height:{height}px;"></div>')
    return ""

# ── PLOTLY LAYOUT ──────────────────────────────────────────────────────────────
def lo(h=420, title="", xlab="", ylab=""):
    d = dict(
        paper_bgcolor=BG, plot_bgcolor=BG2,
        font=dict(color=TXT, family="Inter", size=12),
        xaxis=dict(gridcolor=BD, linecolor=BD, color=TXT,
                   title=dict(text=xlab, font=dict(size=12))),
        yaxis=dict(gridcolor=BD, linecolor=BD, color=TXT,
                   title=dict(text=ylab, font=dict(size=12))),
        legend=dict(bgcolor=BG2, bordercolor=BD, borderwidth=1,
                    font=dict(size=11, color=TXT)),
        margin=dict(l=60, r=20, t=55 if title else 28, b=55),
        height=h,
        hovermode="x unified",
    )
    if title:
        d["title"] = dict(text=title, font=dict(size=14, color=GLD2))
    return d

# ── PRICING FUNCTIONS ──────────────────────────────────────────────────────────
@st.cache_data
def binomial_price(S0, K, r, sigma, T, N, american=True):
    dt   = T / N
    u    = np.exp(sigma * np.sqrt(dt))
    d    = 1 / u
    p    = (np.exp(r * dt) - d) / (u - d)
    disc = np.exp(-r * dt)
    vals = [max(K - S0 * u**j * d**(N - j), 0) for j in range(N + 1)]
    for step in range(N - 1, -1, -1):
        nv = []
        for j in range(step + 1):
            s    = S0 * u**j * d**(step - j)
            cont = disc * (p * vals[j + 1] + (1 - p) * vals[j])
            ex   = max(K - s, 0)
            nv.append(max(cont, ex) if american else cont)
        vals = nv
    return vals[0]

@st.cache_data
def black_scholes_put(S, K, r, sigma, T):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

@st.cache_data
def simulate_paths(S0, r, sigma, T, N, num_paths, seed=42):
    rng   = np.random.default_rng(seed)
    dt    = T / N
    p     = np.zeros((num_paths, N + 1))
    p[:, 0] = S0
    for t in range(1, N + 1):
        z = rng.standard_normal(num_paths)
        p[:, t] = p[:, t-1] * np.exp((r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*z)
    return p

@st.cache_data
def lsmc_put(S0, K, r, sigma, T, N, num_paths, seed=42):
    paths = simulate_paths(S0, r, sigma, T, N, num_paths, seed)
    dt    = T / N
    disc  = np.exp(-r * dt)
    cf    = np.maximum(K - paths[:, -1], 0)
    for t in range(N - 1, 0, -1):
        cf  *= disc
        stk  = paths[:, t]
        ex   = np.maximum(K - stk, 0)
        itm  = ex > 0
        if itm.sum() < 3:
            continue
        x    = stk[itm]
        X    = np.column_stack([np.ones_like(x), x, x**2])
        beta = np.linalg.lstsq(X, cf[itm], rcond=None)[0]
        cont = X @ beta
        idx  = np.where(itm)[0][ex[itm] > cont]
        cf[idx] = ex[idx]
    return np.mean(cf * disc)

@st.cache_data
def implied_vol(mkt, S0, K, r, T, N):
    lo, hi = 0.01, 3.00
    for _ in range(80):
        mid = (lo + hi) / 2
        (lo if binomial_price(S0, K, r, mid, T, N) < mkt else hi)
        if binomial_price(S0, K, r, mid, T, N) < mkt:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2

@st.cache_data
def small_tree(S0, K, r, sigma, T, N_vis=4):
    dt   = T / N_vis
    u    = np.exp(sigma * np.sqrt(dt))
    d    = 1 / u
    p    = (np.exp(r * dt) - d) / (u - d)
    disc = np.exp(-r * dt)
    stk  = [[S0 * u**j * d**(t-j) for j in range(t+1)] for t in range(N_vis+1)]
    vals = [None] * (N_vis + 1)
    ex   = [None] * (N_vis + 1)
    vals[-1] = [max(K - s, 0) for s in stk[-1]]
    ex[-1]   = [False] * (N_vis + 1)
    cur = vals[-1]
    for t in range(N_vis - 1, -1, -1):
        vals[t], ex[t] = [], []
        for j in range(t + 1):
            cont = disc * (p * cur[j+1] + (1-p) * cur[j])
            e    = max(K - stk[t][j], 0)
            vals[t].append(max(cont, e))
            ex[t].append(e > cont and e > 0)
        cur = vals[t]
    return stk, vals, ex

# ── CHARTS ─────────────────────────────────────────────────────────────────────
def fig_tree(stk, vals, ex):
    N = 4
    fig = go.Figure()
    ex_x, ex_y, no_x, no_y = [], [], [], []
    for t in range(N):
        for j in range(t + 1):
            x0, y0 = t, j - t/2
            for dj in [0, 1]:
                fig.add_trace(go.Scatter(
                    x=[x0, t+1, None], y=[y0, (j+dj)-(t+1)/2, None],
                    mode="lines", line=dict(color=BD, width=1.8),
                    showlegend=False, hoverinfo="skip"))
    for t in range(N + 1):
        for j in range(t + 1):
            (ex_x if ex[t][j] else no_x).append(t)
            (ex_y if ex[t][j] else no_y).append(j - t/2)
    if no_x:
        fig.add_trace(go.Scatter(x=no_x, y=no_y, mode="markers",
            marker=dict(size=58, color=BG2, line=dict(color=BLUE, width=2.5)),
            name="Hold", hoverinfo="skip"))
    if ex_x:
        fig.add_trace(go.Scatter(x=ex_x, y=ex_y, mode="markers",
            marker=dict(size=58, color="#7f1d1d", line=dict(color=RED, width=2.5)),
            name="Exercise", hoverinfo="skip"))
    for t in range(N + 1):
        for j in range(t + 1):
            fig.add_annotation(
                x=t, y=j - t/2,
                text=f"<b>S={stk[t][j]:.0f}</b><br>V={vals[t][j]:.2f}",
                showarrow=False, align="center",
                font=dict(size=10.5, color=RED if ex[t][j] else TXT, family="Inter"))
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(color=TXT, family="Inter"),
        xaxis=dict(showgrid=False, zeroline=False,
                   tickvals=list(range(5)),
                   ticktext=["t = 0","t = 1","t = 2","t = 3","t = 4"],
                   linecolor=BD, color=MUT),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        legend=dict(bgcolor=BG2, bordercolor=BD, borderwidth=1, x=0, y=-0.1,
                    orientation="h", font=dict(size=12)),
        margin=dict(l=20, r=20, t=50, b=60), height=500,
        title=dict(text="Binomial Tree — 4 steps shown, 200 steps used for pricing",
                   font=dict(size=13, color=GLD2)),
    )
    return fig

def fig_mc(paths, K, S0, T):
    tg  = np.linspace(0, T, paths.shape[1])
    fig = go.Figure()
    for i in range(120):
        fig.add_trace(go.Scatter(
            x=tg, y=paths[i], mode="lines",
            line=dict(color=f"rgba(96,165,250,0.10)", width=0.8),
            showlegend=False, hoverinfo="skip"))
    fig.add_hline(y=K, line_color=RED, line_dash="dash", line_width=2,
                  annotation_text="Strike: $135", annotation_font_color=RED,
                  annotation_position="top left")
    fig.update_layout(**lo(430, "50,000 Simulated SpaceX/SPCX Stock Paths",
                           "Time to maturity (years)", "Stock price (USD)"))
    return fig

def fig_dist(terminal, K, S0):
    payoffs = np.maximum(K - terminal, 0)
    itm_pct = (terminal < K).mean() * 100
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=terminal, nbinsx=60,
        marker=dict(color=GOLD, opacity=0.75, line=dict(color=BG, width=0.3)),
        name="Terminal price", hovertemplate="$%{x:.0f}<br>Count: %{y}<extra></extra>"))
    fig.add_vline(x=K,  line_color=RED,  line_dash="dash", line_width=2,
                  annotation_text="Strike $135", annotation_font_color=RED)
    fig.add_vline(x=S0, line_color=GRN, line_dash="dot",  line_width=2,
                  annotation_text="S₀ $201.80", annotation_font_color=GRN)
    fig.update_layout(**lo(430, f"Terminal Price Distribution  ·  {itm_pct:.1f}% of paths end ITM",
                           "Terminal price (USD)", "Number of paths"))
    return fig, itm_pct, payoffs

# ── NAVIGATION ─────────────────────────────────────────────────────────────────
SLIDES = ["Title", "The Challenge", "Binomial Tree", "Monte Carlo", "SpaceX / SPCX", "Takeaways"]

if "slide" not in st.session_state:
    st.session_state.slide = 0

cols = st.columns(len(SLIDES))
for i, (col, label) in enumerate(zip(cols, SLIDES)):
    with col:
        active = st.session_state.slide == i
        if st.button(label, use_container_width=True, key=f"n{i}",
                     type="primary" if active else "secondary"):
            st.session_state.slide = i
            st.rerun()

st.markdown("<hr class='divider'>", unsafe_allow_html=True)
slide = st.session_state.slide


# ══════════════════════════════════════════════════════════════════════════════
#  0 · TITLE
# ══════════════════════════════════════════════════════════════════════════════
if slide == 0:
    st.markdown(f"""
    <div style="text-align:center; padding:40px 0 20px 0;">
      {logo_tag(height=100)}
      <div style="margin-top:28px; font-size:0.72rem; font-weight:700;
                  letter-spacing:0.16em; text-transform:uppercase; color:{MUT};">
        Frankfurt University of Applied Sciences &nbsp;·&nbsp; Faculty 3 – Business and Law
      </div>
      <div style="margin-top:6px; font-size:0.85rem; color:{MUT};">
        Computer Based Investment Analysis &nbsp;|&nbsp; Summer Semester 2026
      </div>
      <div style="margin-top:48px; font-size:2.8rem; font-weight:800;
                  color:{TXT}; line-height:1.2; letter-spacing:-0.01em;">
        Pricing of American Options
      </div>
      <div style="margin-top:12px; font-size:1.25rem; color:{MUT}; font-weight:400;">
        Binomial Tree &nbsp;&amp;&nbsp; Least-Squares Monte Carlo
      </div>
      <div style="margin-top:8px; font-size:1.05rem; font-weight:600; color:{GOLD};">
        SpaceX / SPCX Case Study
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr class='divider' style='margin:36px 0;'>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    for col, name, matno in [
        (c1, "", ""),
        (c2, "Ilyos Umurzakov · Leon Ye", "Matriculation No. 1615067 · 1616910"),
        (c3, "", ""),
    ]:
        with col:
            if name:
                st.markdown(f"""
                <div style="text-align:center;">
                  <div style="font-size:1.1rem; font-weight:700; color:{TXT};">{name}</div>
                  <div style="font-size:0.82rem; color:{MUT}; margin-top:6px;">{matno}</div>
                  <div style="font-size:0.82rem; color:{MUT}; margin-top:4px;">
                    Lecturers: Ferdinand Wöhrle &nbsp;·&nbsp; Lukas Müller
                  </div>
                </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    for col, val, lbl in [
        (c1, "2 Methods", "Binomial Tree + LSMC"),
        (c2, "$1,130", "SpaceX put contract price"),
        (c3, "107%", "Implied volatility"),
    ]:
        with col:
            st.markdown(f"""<div class="metric">
              <div class="metric-val">{val}</div>
              <div class="metric-lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  1 · THE CHALLENGE
# ══════════════════════════════════════════════════════════════════════════════
elif slide == 1:
    st.markdown(f'<div class="section-label">The Problem</div>', unsafe_allow_html=True)
    st.markdown("## Why Are American Options Hard to Price?")

    col_l, col_r = st.columns(2, gap="large")
    with col_l:
        for num, head, body in [
            ("01", "A put option gives you the right to sell",
             "You can sell the underlying stock at the fixed strike price K — even if the market price has dropped far below it."),
            ("02", "European: exercise only at expiry",
             "The Black–Scholes formula handles this perfectly. One date, one decision."),
            ("03", "American: exercise any time before expiry",
             "At every moment the holder asks: <i>is it better to exercise now or wait?</i> This creates thousands of decision points."),
            ("04", "No closed-form solution",
             "Because of the early exercise choice, there is no simple formula. We need numerical methods."),
        ]:
            st.markdown(f"""<div class="step-row">
              <div class="step-num">{num}</div>
              <div>
                <div class="step-head">{head}</div>
                <div class="step-body">{body}</div>
              </div>
            </div>""", unsafe_allow_html=True)

    with col_r:
        st.markdown(f'<div class="section-label">The Early Exercise Premium</div>',
                    unsafe_allow_html=True)
        st.markdown(f"""<div class="highlight" style="margin-bottom:20px;">
          The American option is always worth <b>at least as much</b> as the European option.<br><br>
          The extra value — called the <b style="color:{GOLD};">early exercise premium</b> —
          comes from the right to act before expiry.<br><br>
          For a put on a stock that has fallen sharply, it may be better to
          take the cash <i>today</i> rather than wait until maturity.
        </div>""", unsafe_allow_html=True)

        # Simple visual: timeline
        fig = go.Figure()
        # European
        fig.add_shape(type="line", x0=0, x1=1, y0=0.7, y1=0.7,
                      line=dict(color=MUT, width=3))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0.7, 0.7], mode="markers+text",
            marker=dict(size=[10, 16], color=[BLUE, BLUE]),
            text=["Start", "Expiry only"], textposition=["top center", "top center"],
            textfont=dict(color=BLUE, size=11), showlegend=False, hoverinfo="skip"))
        # American
        fig.add_shape(type="line", x0=0, x1=1, y0=0.3, y1=0.3,
                      line=dict(color=GOLD, width=3))
        for xi in [0.2, 0.4, 0.6, 0.8, 1.0]:
            fig.add_trace(go.Scatter(x=[xi], y=[0.3], mode="markers",
                marker=dict(size=12, color=GOLD), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[0.5], y=[0.3], mode="text",
            text=["Exercise any time"], textposition="bottom center",
            textfont=dict(color=GOLD, size=11), showlegend=False, hoverinfo="skip"))

        fig.add_annotation(x=-0.06, y=0.7, text="European", showarrow=False,
                           font=dict(color=BLUE, size=12, family="Inter"), xanchor="right")
        fig.add_annotation(x=-0.06, y=0.3, text="American", showarrow=False,
                           font=dict(color=GOLD, size=12, family="Inter"), xanchor="right")

        fig.update_layout(
            paper_bgcolor=BG, plot_bgcolor=BG2,
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False,
                       range=[-0.08, 1.1]),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False,
                       range=[0, 1]),
            height=240, margin=dict(l=80, r=20, t=30, b=20),
            title=dict(text="Exercise Window", font=dict(size=12, color=GLD2)),
        )
        st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        c1.markdown(f"""<div class="metric">
          <div class="metric-val" style="font-size:1.5rem;">V<sub style="font-size:1rem">Am</sub> ≥ V<sub style="font-size:1rem">Eu</sub></div>
          <div class="metric-lbl">Always true</div>
        </div>""", unsafe_allow_html=True)
        c2.markdown(f"""<div class="metric">
          <div class="metric-val" style="font-size:1.5rem; color:{GRN};">Puts</div>
          <div class="metric-lbl">Early exercise most relevant for puts</div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  2 · BINOMIAL TREE
# ══════════════════════════════════════════════════════════════════════════════
elif slide == 2:
    st.markdown(f'<div class="section-label">Method 1</div>', unsafe_allow_html=True)
    st.markdown("## Cox–Ross–Rubinstein Binomial Tree")

    col_l, col_r = st.columns([1, 1.6], gap="large")
    with col_l:
        st.markdown("### How it works")
        for num, head, body in [
            ("01", "Build a price tree",
             "The stock can move UP or DOWN each time step. After N steps you have a full lattice of possible prices."),
            ("02", "Value at maturity",
             "At the final nodes, the put payoff is simply max(K − S, 0)."),
            ("03", "Work backwards",
             "At each node, compare: keep holding (continuation value) vs. exercise now (K − S). Take the maximum."),
            ("04", "Price = root node",
             "The option value at t = 0 is the answer. N = 200 steps gives 4-decimal accuracy."),
        ]:
            st.markdown(f"""<div class="step-row">
              <div class="step-num">{num}</div>
              <div>
                <div class="step-head">{head}</div>
                <div class="step-body">{body}</div>
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown(f"""<div class="highlight" style="margin-top:16px;">
          <b style="color:{GOLD};">Key strength:</b> every single node is readable.
          You can trace exactly <i>why</i> the option is exercised at a specific price and time.
        </div>""", unsafe_allow_html=True)

    with col_r:
        # SpaceX tree
        S0, K, r, T = 201.80, 135, 0.037, 93/365
        with st.spinner("Computing…"):
            sigma_iv = implied_vol(11.30, S0, K, r, T, 200)
            stk, vals, ex = small_tree(S0, K, r, sigma_iv, T, N_vis=4)
        st.plotly_chart(fig_tree(stk, vals, ex), use_container_width=True)
        st.markdown(f"""<div style="font-size:0.85rem; color:{MUT}; text-align:center;">
          <span style="color:{RED}; font-weight:700;">Red nodes</span> = early exercise optimal &nbsp;|&nbsp;
          S = stock price &nbsp;|&nbsp; V = put value (USD)
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  3 · MONTE CARLO / LSMC
# ══════════════════════════════════════════════════════════════════════════════
elif slide == 3:
    st.markdown(f'<div class="section-label">Method 2</div>', unsafe_allow_html=True)
    st.markdown("## Least-Squares Monte Carlo (LSMC)")

    col_l, col_r = st.columns([1, 1.6], gap="large")
    with col_l:
        st.markdown("### How it works")
        for num, head, body in [
            ("01", "Simulate 50,000 paths",
             "Each path is one possible future for the stock price, generated with random normal shocks."),
            ("02", "Start at maturity",
             "At expiry, each path either pays off (stock below strike) or expires worthless."),
            ("03", "Regression step",
             "Moving backwards: for each time step, use regression to estimate whether waiting is worth more than exercising now."),
            ("04", "Average the result",
             "The option price is the average discounted cashflow across all 50,000 paths."),
        ]:
            st.markdown(f"""<div class="step-row">
              <div class="step-num">{num}</div>
              <div>
                <div class="step-head">{head}</div>
                <div class="step-body">{body}</div>
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown(f"""<div class="highlight" style="margin-top:16px;">
          <b style="color:{GOLD};">vs. Binomial Tree:</b> less transparent but more flexible —
          the same method scales to options with multiple risk factors or complex payoffs.
        </div>""", unsafe_allow_html=True)

    with col_r:
        S0, K, r, T = 201.80, 135, 0.037, 93/365
        with st.spinner("Simulating…"):
            sigma_iv = implied_vol(11.30, S0, K, r, T, 200)
            paths    = simulate_paths(S0, r, sigma_iv, T, 200, 50_000)
        tab1, tab2 = st.tabs(["Stock Paths", "Terminal Distribution"])
        with tab1:
            st.plotly_chart(fig_mc(paths, K, S0, T), use_container_width=True)
        with tab2:
            fd, itm_pct, payoffs = fig_dist(paths[:, -1], K, S0)
            st.plotly_chart(fd, use_container_width=True)
            st.markdown(f"""<div style="text-align:center; font-size:0.88rem; color:{MUT};">
              Average terminal payoff (ITM paths only):
              <b style="color:{GOLD};">${np.mean(payoffs[payoffs>0]):.2f}</b> &nbsp;|&nbsp;
              Paths ending in-the-money: <b style="color:{GOLD};">{itm_pct:.1f}%</b>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  4 · SPACEX / SPCX
# ══════════════════════════════════════════════════════════════════════════════
elif slide == 4:
    st.markdown(f'<div class="section-label">Case Study</div>', unsafe_allow_html=True)
    st.markdown("## SpaceX / SPCX — September 2026 Put")

    # Context bar
    c1, c2, c3, c4 = st.columns(4)
    for col, val, lbl in [
        (c1, "$135",    "IPO price · 12 Jun 2026"),
        (c2, "$201.80", "Stock price · 16 Jun 2026"),
        (c3, "$1,130",  "Market premium per contract"),
        (c4, "93 days", "Time to Sep 18 expiry"),
    ]:
        with col:
            st.markdown(f"""<div class="metric">
              <div class="metric-val" style="font-size:1.8rem;">{val}</div>
              <div class="metric-lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    col_l, col_r = st.columns(2, gap="large")

    S0, K, r, T, mkt = 201.80, 135, 0.037, 93/365, 11.30
    with st.spinner("Computing model prices…"):
        sigma_iv = implied_vol(mkt, S0, K, r, T, 200)
        b_am  = binomial_price(S0, K, r, sigma_iv, T, 200, american=True)
        b_eu  = binomial_price(S0, K, r, sigma_iv, T, 200, american=False)
        l_am  = lsmc_put(S0, K, r, sigma_iv, T, 200, 50_000)
        ee    = b_am - b_eu

    with col_l:
        st.markdown("### What drives the price?")
        st.markdown(f"""<div class="highlight">
          The put strike is <b>$135</b> but the stock trades at <b>$201.80</b> —
          the option is deeply <b>out of the money</b>.<br><br>
          A premium of $11.30/share can only exist if the market expects
          <b style="color:{GOLD};">very large price swings</b> before September.
          That expectation is captured by the implied volatility.
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""<div class="metric" style="margin-top:16px;">
          <div class="metric-val" style="font-size:3rem;">{sigma_iv*100:.1f}%</div>
          <div class="metric-lbl">Implied Volatility — extracted by bisection search</div>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""<div style="margin-top:16px; font-size:0.85rem; color:{MUT}; line-height:1.7;">
          A typical stock has σ ≈ 20–30%. SpaceX/SPCX shows 107% because
          it is a newly listed stock with no price history and high uncertainty.
          The market charges a large premium to compensate for that risk.
        </div>""", unsafe_allow_html=True)

    with col_r:
        st.markdown("### Model output")
        c1, c2 = st.columns(2)
        c1.markdown(f"""<div class="metric">
          <div class="metric-val">${b_am:.2f}</div>
          <div class="metric-lbl">Binomial American put</div>
        </div>""", unsafe_allow_html=True)
        c2.markdown(f"""<div class="metric">
          <div class="metric-val">${l_am:.2f}</div>
          <div class="metric-lbl">LSMC American put</div>
        </div>""", unsafe_allow_html=True)

        c3, c4 = st.columns(2)
        c3.markdown(f"""<div class="metric" style="margin-top:8px;">
          <div class="metric-val">${b_eu:.2f}</div>
          <div class="metric-lbl">Binomial European put</div>
        </div>""", unsafe_allow_html=True)
        c4.markdown(f"""<div class="metric" style="margin-top:8px;">
          <div class="metric-val" style="color:{GRN};">${ee:.4f}</div>
          <div class="metric-lbl">Early exercise premium</div>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""<div class="highlight" style="margin-top:16px;">
          The early exercise premium is only <b style="color:{GRN};">${ee:.4f}</b> —
          less than 0.3% of the $11.30 price.<br><br>
          This is a <b style="color:{GOLD};">volatility story</b>, not an early exercise story.
          Both methods agree closely, which validates the models.
        </div>""", unsafe_allow_html=True)

        # Mini bar chart: what makes up the price
        fig_split = go.Figure(go.Bar(
            x=["European put value", "Early exercise premium"],
            y=[b_eu, ee],
            marker_color=[BLUE, GOLD],
            text=[f"${b_eu:.2f}", f"${ee:.4f}"],
            textposition="outside",
            textfont=dict(color=TXT, size=12),
        ))
        fig_split.update_layout(
            paper_bgcolor=BG, plot_bgcolor=BG2,
            font=dict(color=TXT, family="Inter", size=12),
            xaxis=dict(gridcolor=BD, linecolor=BD, color=MUT),
            yaxis=dict(gridcolor=BD, linecolor=BD, color=MUT,
                       title="USD", range=[0, b_am * 1.18]),
            margin=dict(l=50, r=20, t=40, b=40), height=230,
            title=dict(text="Price decomposition (per share)",
                       font=dict(size=12, color=GLD2)),
            showlegend=False,
        )
        st.plotly_chart(fig_split, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  5 · TAKEAWAYS
# ══════════════════════════════════════════════════════════════════════════════
elif slide == 5:
    st.markdown(f'<div class="section-label">Conclusion</div>', unsafe_allow_html=True)
    st.markdown("## Key Takeaways")

    c1, c2, c3 = st.columns(3, gap="large")

    with c1:
        st.markdown(f"""<div class="metric" style="height:160px; display:flex; flex-direction:column; justify-content:center;">
          <div class="metric-val" style="font-size:2rem;">Binomial Tree</div>
          <div class="metric-lbl" style="margin-top:14px; line-height:1.6;">
            Most transparent method.<br>
            Every node shows exactly when and why early exercise is optimal.
          </div>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""<div class="step-row" style="margin-top:12px;">
          <div class="step-num" style="font-size:1.2rem;">→</div>
          <div class="step-body">
            European price converges to Black–Scholes as N → ∞.<br>
            N = 200 is sufficient for 4-decimal accuracy.
          </div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""<div class="metric" style="height:160px; display:flex; flex-direction:column; justify-content:center;">
          <div class="metric-val" style="font-size:2rem;">LSMC</div>
          <div class="metric-lbl" style="margin-top:14px; line-height:1.6;">
            Same answer via simulation.<br>
            More flexible — scales to complex multi-factor options.
          </div>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""<div class="step-row" style="margin-top:12px;">
          <div class="step-num" style="font-size:1.2rem;">→</div>
          <div class="step-body">
            Small differences from binomial are normal —
            simulation noise and regression approximation, not errors.
          </div>
        </div>""", unsafe_allow_html=True)

    with c3:
        S0, K, r, T = 201.80, 135, 0.037, 93/365
        sigma_iv = implied_vol(11.30, S0, K, r, T, 200)
        b_am = binomial_price(S0, K, r, sigma_iv, T, 200, american=True)
        b_eu = binomial_price(S0, K, r, sigma_iv, T, 200, american=False)
        ee   = b_am - b_eu

        st.markdown(f"""<div class="metric" style="height:160px; display:flex; flex-direction:column; justify-content:center;">
          <div class="metric-val" style="font-size:2rem;">SpaceX/SPCX</div>
          <div class="metric-lbl" style="margin-top:14px; line-height:1.6;">
            $1,130 contract price explained by<br>
            107% implied volatility — not early exercise.
          </div>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""<div class="step-row" style="margin-top:12px;">
          <div class="step-num" style="font-size:1.2rem;">→</div>
          <div class="step-body">
            Early exercise premium = ${ee:.4f} out of $11.30.<br>
            Less than 0.3% of the total option value.
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='divider' style='margin-top:32px;'>", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:{BG2}; border:1px solid {GOLD}44;
                border-radius:12px; padding:28px 36px; margin-top:8px; text-align:center;">
      <div style="font-size:1.25rem; font-weight:700; color:{TXT}; line-height:1.8;">
        The SpaceX/SPCX put option is a <span style="color:{GOLD};">volatility story</span>,
        not an early exercise story.<br>
        Both numerical methods confirm the same result — giving us confidence in the pricing.
      </div>
      <div style="margin-top:20px; font-size:0.82rem; color:{MUT};">
        Frankfurt UAS &nbsp;·&nbsp; Computer Based Investment Analysis &nbsp;·&nbsp;
        Ilyos Umurzakov · Leon Ye &nbsp;·&nbsp; Summer 2026
      </div>
    </div>
    """, unsafe_allow_html=True)
