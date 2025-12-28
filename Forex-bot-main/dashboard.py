"""
Streamlit Dashboard for AI Trading Bot.
Run with: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go
import plotly.express as px
import time
import os
from datetime import datetime

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="QuantAI Pro Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "Powered by QuantAI XGBoost Engine"
    }
)

# Constants
STATE_FILE = "state/bot_status.json"
LOG_FILE = "logs/trading.log"
REFRESH_RATE = 2

# ============================================================
# PREMIUM STYLING (CSS)
# ============================================================
st.markdown("""
<style>
    /* 1. Global Reset & Fonts */
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@300;400;600;800&display=swap');
    
    .stApp {
        background-color: #050505;
        background-image: radial-gradient(circle at 50% 0%, #1a1f35 0%, #050505 70%);
        color: #e0e0e0;
        font-family: 'Inter', sans-serif;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif !important;
        letter-spacing: -0.5px;
        color: #ffffff !important;
    }
    
    code, .stCodeBlock {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* 2. Glassmorphism Cards */
    div[data-testid="stMetric"], .css-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }

    div[data-testid="stMetric"]:hover {
        border-color: rgba(255, 255, 255, 0.2);
        transform: translateY(-2px);
    }
    
    div[data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
        color: #888 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #fff !important;
        text-shadow: 0 0 20px rgba(255,255,255,0.1);
    }

    /* 3. Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0a0a0a;
        border-right: 1px solid #1f1f1f;
    }

    /* 4. Tables */
    div[data-testid="stDataFrame"] {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 8px;
    }

    /* 5. Custom Status Indicators */
    .status-badge {
        display: inline-flex;
        align-items: center;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    
    .status-live { background: rgba(0, 255, 157, 0.1); color: #00ff9d; border: 1px solid rgba(0, 255, 157, 0.2); }
    .status-stopped { background: rgba(255, 75, 75, 0.1); color: #ff4b4b; border: 1px solid rgba(255, 75, 75, 0.2); }
    
    /* 6. Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border: none;
        color: #888;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #fff;
        border-bottom: 2px solid #007bff; /* Accent color */
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_state():
    """Load latest bot state with error handling."""
    if not os.path.exists(STATE_FILE):
        return None
    try:
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    except:
        return None

def load_logs(n_lines=50):
    """Load tail of log file."""
    if not os.path.exists(LOG_FILE):
        return ["Waiting for logs..."]
    try:
        with open(LOG_FILE, 'r') as f:
            lines = f.readlines()
            # formatting logs for better readability
            clean_logs = []
            for line in lines[-n_lines:]:
                if "INFO" in line:
                    clean_logs.append(f"ℹ️ {line.strip()}")
                elif "WARNING" in line:
                    clean_logs.append(f"⚠️ {line.strip()}")
                elif "ERROR" in line:
                    clean_logs.append(f"❌ {line.strip()}")
                else:
                    clean_logs.append(f"📝 {line.strip()}")
            return clean_logs[::-1]
    except:
        return ["Error reading logs."]

def get_latest_data_file(symbol):
    """Finds the most recent CSV for a symbol."""
    data_dir = "data/raw"
    if not os.path.exists(data_dir):
        return None
    
    # Handle Yahoo Finance suffix mismatch
    search_symbol = symbol.replace("=X", "")
    
    files = [f for f in os.listdir(data_dir) if search_symbol in f and f.endswith(".csv")]
    if not files:
        return None
    latest_file = max([os.path.join(data_dir, f) for f in files], key=os.path.getmtime)
    return latest_file

def create_price_chart(symbol):
    """Generates a high-performance Plotly candlestick chart."""
    file_path = get_latest_data_file(symbol)
    if not file_path:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            annotations=[dict(text="No Data Available", showarrow=False, font=dict(size=20, color="#555"))],
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        return fig

    try:
        df = pd.read_csv(file_path)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        
        df = df.tail(100) # Performance optimization
        
        fig = go.Figure(data=[go.Candlestick(
            x=df.index,
            open=df['Open'], high=df['High'],
            low=df['Low'], close=df['Close'],
            name=symbol,
            increasing_line_color='#00ff9d', # Neon Green
            decreasing_line_color='#ff4b4b', # Neon Red
            opacity=0.9
        )])

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=450,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_rangeslider_visible=False,
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
            title=dict(text=f'{symbol} • M1', font=dict(size=14, color="#888"), x=0)
        )
        return fig
    except Exception as e:
        return go.Figure()

# ============================================================
# MAIN LAYOUT
# ============================================================

# Header Section
c1, c2 = st.columns([3, 1])
with c1:
    st.markdown("""
    <div style='display: flex; align-items: center; gap: 15px;'>
        <h1 style='margin: 0; font-size: 2.5rem; background: linear-gradient(90deg, #fff, #888); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>QuantAI Terminal</h1>
        <span class='status-badge status-live'>● SYSTEM ONLINE</span>
    </div>
    <p style='margin-top: 5px; color: #666; font-size: 0.9rem;'>Advanced Algorithmic Trading • XGBoost Neural Engine • v2.1.0</p>
    """, unsafe_allow_html=True)

with c2:
    if st.button("↻ Force Update", type="primary", use_container_width=True):
        st.rerun()

st.divider()

# Load State
state = load_state()

if state:
    account = state.get('account', {})
    
    # --- METRICS ROW ---
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Equity", f"${account.get('equity', 0):,.2f}", f"{account.get('return_pct', 0):+.2f}%")
    with m2:
        st.metric("Available Cash", f"${account.get('cash', 0):,.2f}")
    with m3:
        st.metric("Position Value", f"${(account.get('equity', 0) - account.get('cash', 0)):,.2f}")
    with m4:
        st.metric("Buying Power", f"${account.get('buying_power', 0):,.2f}")

    # --- MAIN WORKSPACE ---
    st.markdown("<br>", unsafe_allow_html=True)
    
    tab_live, tab_history = st.tabs(["📈 START COMMAND CENTER", "🕰️ HISTORY LOGS"])
    
    with tab_live:
        symbols = state.get('symbols', [])
        
        if not symbols:
            st.warning("No Symbols Configured")
        else:
            # Layout: Left for Charts, Right for AI Signals
            col_charts, col_signals = st.columns([2, 1])
            
            with col_charts:
                selected_symbol = st.selectbox("Select Asset Feed", symbols, label_visibility="collapsed")
                st.plotly_chart(create_price_chart(selected_symbol), use_container_width=True)
            
            with col_signals:
                st.markdown("### AI SIGNAL PROCESSOR")
                
                preds = state.get('predictions', {}).get(selected_symbol, {})
                prediction_val = preds.get('prob', 0.5)
                signal_type = preds.get('signal', 'NEUTRAL')
                
                # Custom Signal Card
                card_color = "#00ff9d" if signal_type == "UP" else "#ff4b4b" if signal_type == "DOWN" else "#fff"
                bg_color = "rgba(0, 255, 157, 0.1)" if signal_type == "UP" else "rgba(255, 75, 75, 0.1)" if signal_type == "DOWN" else "rgba(255,255,255,0.05)"
                
                st.markdown(f"""
                <div style="background: {bg_color}; border: 1px solid {card_color}; padding: 20px; border-radius: 12px; text-align: center;">
                    <h2 style="color: {card_color} !important; margin: 0; font-size: 2rem;">{signal_type}</h2>
                    <p style="color: #ccc; margin: 5px 0 0 0; font-size: 0.8rem;">MODEL CONFIDENCE</p>
                    <h3 style="margin: 0; font-size: 1.5rem;">{prediction_val:.1%}</h3>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("### Active Position")
                pos = next((p for p in state.get('positions', []) if p['symbol'] == selected_symbol), None)
                
                if pos:
                    st.info(f"OPEN: {pos['side'].upper()} | QTY: {pos['qty']} | Entry: ${pos['entry_price']:.4f}")
                    upl = pos.get('unrealized_pl', 0)
                    st.metric("Floating P&L", f"${upl:.2f}", delta=upl)
                else:
                    st.markdown("""
                    <div style="background: rgba(255,255,255,0.03); padding: 15px; border-radius: 8px; text-align: center; color: #666;">
                        NO ACTIVE POSITIONS
                    </div>
                    """, unsafe_allow_html=True)

    with tab_history:
        st.markdown("### Recent Trade Executions")
        trades = state.get('trade_history', [])
        if trades:
            df_t = pd.DataFrame(trades)
            st.dataframe(
                df_t[['timestamp', 'symbol', 'side', 'quantity', 'price', 'pnl']].style.applymap(
                    lambda v: 'color: #00ff9d' if v == 'BUY' else 'color: #ff4b4b' if v == 'SELL' else '', subset=['side']
                ),
                use_container_width=True,
                height=400
            )
        else:
            st.caption("No trade history available yet.")

    # Status Footer
    with st.expander("System Logs & Diagnostics", expanded=False):
        st.code("\n".join(load_logs(30)), language="text")

else:
    # Empty State / Loading State
    st.warning("⚠️ CONNECTING TO NEURAL CORE...")
    st.markdown("Waiting for `state/bot_status.json` to be generated...")
    time.sleep(2)
    st.rerun()

# Auto Refresh logic
time.sleep(REFRESH_RATE)
st.rerun()
