import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go
import time
from datetime import datetime
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="QuantAI Pro",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CUSTOM CSS (PREMIUM GLASSMORPHISM) ---
st.markdown("""
    <style>
    /* Global Background */
    .stApp {
        background: radial-gradient(circle at 10% 20%, #0f172a 0%, #000000 90%);
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Default Elements */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    div[data-testid="stToolbar"] {visibility: hidden;}
    
    /* Glassmorphism Containers */
    .metric-card {
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.4);
    }
    
    /* Typography */
    h1, h2, h3 {
        color: #f8fafc;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    .big-stat {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-stat {
        font-size: 0.9rem;
        color: #94a3b8;
        font-weight: 500;
    }
    
    /* Status Badge */
    .status-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        box-shadow: 0 0 10px rgba(0,0,0,0.2);
    }
    .status-up { background: rgba(34, 197, 94, 0.15); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.3); }
    .status-down { background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); }
    
    /* Chart Container */
    .chart-container {
        border-radius: 16px; 
        overflow: hidden; 
        border: 1px solid rgba(255,255,255,0.05);
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: #0f172a; }
    ::-webkit-scrollbar-thumb { background: #334155; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #475569; }
    
    </style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---
def load_state():
    try:
        if os.path.exists("state/bot_status.json"):
            with open("state/bot_status.json", "r") as f:
                return json.load(f)
    except Exception:
        pass
    return None

def load_data(symbol):
    try:
        data_dir = "data/raw"
        if not os.path.exists(data_dir): return None
        
        # Handle 'US500' vs 'US500.csv' etc
        files = [f for f in os.listdir(data_dir) if symbol in f and f.endswith(".csv")]
        if not files: return None
        
        latest_file = max([os.path.join(data_dir, f) for f in files], key=os.path.getmtime)
        df = pd.read_csv(latest_file)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        return df
    except Exception:
        return None

# --- UI LAYOUT ---

c_head1, c_head2 = st.columns([3, 1])
with c_head1:
    st.markdown("<h1>⚡ QuantAI <span style='font-size:1.2rem; color:#64748b; font-weight:300'>| PRO TERMINAL</span></h1>", unsafe_allow_html=True)

# Load State
state = load_state()

# Status Indicator
if state:
    last_ts = datetime.fromisoformat(state['timestamp'])
    is_live = (datetime.now() - last_ts).seconds < 120
    status_html = f"""
        <div style="text-align:right; padding-top:15px;">
            <span class="status-badge {'status-up' if is_live else 'status-down'}">
                {'● SYSTEM ONLINE' if is_live else '● DISCONNECTED'}
            </span>
        </div>
    """
else:
    status_html = """
        <div style="text-align:right; padding-top:15px;">
            <span class="status-badge status-down">● OFFLINE</span>
        </div>
    """
    
with c_head2:
    st.markdown(status_html, unsafe_allow_html=True)

if not state:
    st.markdown("""
        <div style="display:flex; justify-content:center; align-items:center; height:60vh; flex-direction:column; color:#64748b;">
            <h2>Waiting for Neural Core...</h2>
            <p>Please ensure trading_bot.py is running</p>
        </div>
    """, unsafe_allow_html=True)
    time.sleep(2)
    st.rerun()

# --- METRICS DASHBOARD ---
acc = state.get('account', {})
m1, m2, m3, m4 = st.columns(4)

with m1:
    pnl = acc.get('return_pct', 0)
    pnl_color = "#4ade80" if pnl >= 0 else "#f87171"
    st.markdown(f"""
        <div class="metric-card">
            <div class="sub-stat">Total Equity</div>
            <div class="big-stat">${acc.get('equity', 0):,.2f}</div>
            <div style="color:{pnl_color}; font-size:0.9rem; font-weight:600; margin-top:5px;">
                {pnl:+.2f}% Lifetime
            </div>
        </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown(f"""
        <div class="metric-card">
            <div class="sub-stat">Available Capital</div>
            <div class="big-stat" style="font-size:1.8rem; color:#e2e8f0; background:none;">${acc.get('buying_power', 0):,.2f}</div>
            <div class="sub-stat" style="color:#64748b; margin-top:5px;">Cash Balance: ${acc.get('cash', 0):,.2f}</div>
        </div>
    """, unsafe_allow_html=True)

with m3:
    active_pos = len(state.get('positions', []))
    st.markdown(f"""
        <div class="metric-card">
            <div class="sub-stat">Active Positions</div>
            <div class="big-stat" style="font-size:1.8rem; color:#e2e8f0; background:none;">{active_pos}</div>
            <div class="sub-stat" style="color:#64748b; margin-top:5px;">Open Orders</div>
        </div>
    """, unsafe_allow_html=True)

with m4:
    market = state.get('market_type', 'UNK').upper()
    st.markdown(f"""
        <div class="metric-card">
            <div class="sub-stat">Market Protocol</div>
            <div class="big-stat" style="font-size:1.8rem; color:#fff; background:none; text-shadow:0 0 15px rgba(255,255,255,0.2);">{market}</div>
            <div class="sub-stat" style="color:#64748b; margin-top:5px;">MT5 Feed Active</div>
        </div>
    """, unsafe_allow_html=True)

# --- CHARTING & SIGNALS ---
col_main, col_side = st.columns([2.5, 1])

with col_main:
    st.markdown("### 📈 Live Market Intelligence")
    
    symbols = state.get('symbols', [])
    selected_symbol = st.selectbox("Select Asset Feed", symbols, label_visibility="collapsed")
    
    df = load_data(selected_symbol)
    
    if df is not None:
        # Plotly Chart
        fig = go.Figure(data=[go.Candlestick(
            x=df['Date'],
            open=df['Open'], high=df['High'],
            low=df['Low'], close=df['Close'],
            increasing_line_color='#4ade80', 
            decreasing_line_color='#f87171'
        )])
        
        fig.update_layout(
            xaxis_rangeslider_visible=False,
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)', # Transparent background
            margin=dict(l=0, r=0, t=0, b=0),
            height=450,
            xaxis=dict(showgrid=False, color='#64748b'),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', color='#64748b')
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown("""
            <div class="metric-card" style="height:450px; display:flex; align-items:center; justify-content:center; color:#64748b;">
                NO MARKET DATA AVAILABLE
            </div>
        """, unsafe_allow_html=True)

with col_side:
    st.markdown("### 🧠 AI Signal")
    
    preds = state.get('predictions', {}).get(selected_symbol, {})
    prob = preds.get('prob', 0.0)
    signal = preds.get('signal', 'NEUTRAL')
    
    # Signal Card Colors
    if signal == 'UP':
        sig_color = "#4ade80"
        sig_bg = "rgba(74, 222, 128, 0.1)"
    elif signal == 'DOWN':
        sig_color = "#f87171"
        sig_bg = "rgba(248, 113, 113, 0.1)"
    else:
        sig_color = "#94a3b8"
        sig_bg = "rgba(148, 163, 184, 0.1)"
        
    st.markdown(f"""
        <div class="metric-card" style="text-align:center; padding: 30px;">
            <div style="font-size: 0.9rem; color: #94a3b8; letter-spacing: 1px;">AI PREDICTION</div>
            <div style="font-size: 4rem; font-weight: 800; color: {sig_color}; text-shadow: 0 0 20px {sig_color}40;">
                {signal}
            </div>
            <div style="font-size: 1.2rem; font-weight: 600; color: #e2e8f0; margin-top: 5px;">
                {prob:.1%} Confidence
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Active Position Card
    my_pos = next((p for p in state.get('positions', []) if p['symbol'] == selected_symbol), None)
    if my_pos:
        st.markdown(f"""
            <div class="metric-card" style="border-left:4px solid #38bdf8;">
                <div style="color:#38bdf8; font-weight:700; font-size:0.8rem;">ACTIVE POSITION</div>
                <div style="font-size:1.5rem; font-weight:700; color:#fff;">{my_pos['side'].upper()}</div>
                <div style="color:#94a3b8; font-size:0.9rem;">{my_pos['qty']} Units @ {my_pos.get('entry_price', 0):.2f}</div>
            </div>
        """, unsafe_allow_html=True)


# --- LIVE FEED ---
st.markdown("### 📜 Algo Feed")
trades = state.get('trade_history', [])

if not trades:
    st.caption("AI is scanning... No trades executed this session yet.")
else:
    for t in reversed(trades[-5:]): # Show last 5
        color = "#4ade80" if t['side'] == 'BUY' else "#f87171"
        direction_icon = "↗" if t['side'] == 'BUY' else "↘"
        
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.02); border-bottom:1px solid rgba(255,255,255,0.05); padding:12px; display:flex; align-items:center; justify-content:space-between; margin-bottom:5px; border-radius:8px;">
            <div style="display:flex; align-items:center; gap:15px;">
                <div style="font-size:1.2rem; background:{color}20; width:40px; height:40px; display:flex; align-items:center; justify-content:center; border-radius:50%; color:{color};">
                    {direction_icon}
                </div>
                <div>
                    <div style="font-weight:700; color:#e2e8f0;">{t['symbol']}</div>
                    <div style="font-size:0.8rem; color:{color}; font-weight:600;">{t['side']}</div>
                </div>
            </div>
            <div style="text-align:right;">
                <div style="font-weight:600; color:#cbd5e1;">{t['quantity']} Units</div>
                <div style="font-size:0.8rem; color:#64748b;">@ {t['price']:.2f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Auto Refresh loop
time.sleep(1)
st.rerun()
