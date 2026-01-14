import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go
import time
from datetime import datetime
import os

# --- CONFIGURATION ---
st.set_page_config(
    page_title="QuantAI Pro",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- LOAD CUSTOM CSS ---
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

try:
    load_css("assets/style.css")
except FileNotFoundError:
    st.error("CSS file not found. Please ensure 'assets/style.css' exists.")

# --- HELPER: GLASS METRIC CARD ---
def glass_metric_card(label, value, subtext=None, delta=None):
    delta_html = ""
    # If delta is provided, color the subtext based on it
    if delta is not None:
        color_class = "delta-pos" if "+" in str(delta) or (isinstance(delta, (int, float)) and delta > 0) else "delta-neg"
        # If subtext provided, wrap it in color. If not, maybe use delta as subtext?
        # Preserving original logic: subtext IS the delta string usually.
        # But let's color the subtext if delta is passed.
        delta_html = f'<span class="metric-delta {color_class}">{subtext}</span>'
    elif subtext:
        delta_html = f'<span class="sub-stat">{subtext}</span>'
        
    html = f"""
    <div class="glass-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


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

# --- SIDEBAR CONTROLS ---
st.sidebar.markdown("## 🎮 Control Panel")

def send_command(cmd, params=None):
    try:
        with open('command.json', 'w') as f:
            json.dump({"command": cmd, "params": params or {}, "timestamp": datetime.now().isoformat()}, f)
        st.toast(f"Command Sent: {cmd}", icon="🚀")
    except Exception as e:
        st.error(f"Failed to send command: {e}")

# Main Controls
col_btn1, col_btn2 = st.sidebar.columns(2)
if col_btn1.button("▶ RUN", type="primary"):
    send_command("RUN")
if col_btn2.button("⏸ PAUSE"):
    send_command("STOP")

st.sidebar.markdown("---")

# Manual Trading
st.sidebar.markdown("### 🛠 Manual Trade")
if state: 
    symbols_list = state.get('symbols', [])
else:
    symbols_list = ["R_75", "CRASH500", "BOOM1000"] # Fallback

with st.sidebar.form("manual_trade_form"):
    mt_symbol = st.selectbox("Asset", symbols_list)
    mt_side = st.selectbox("Direction", ["BUY", "SELL"])
    mt_qty = st.number_input("Volume (Lots)", min_value=0.001, value=0.01, step=0.01, format="%.3f")
    
    if st.form_submit_button("⚡ Execute Now"):
        send_command("MANUAL_TRADE", {"symbol": mt_symbol, "side": mt_side, "qty": mt_qty})

st.sidebar.markdown("---")
# st.sidebar.info("System Status: Monitoring")
auto_refresh = st.sidebar.checkbox("⚡ Auto-Refresh", value=True)
if not auto_refresh:
    st.sidebar.warning("⚠️ Real-time updates paused")
st.sidebar.info(f"System Status: {'Monitoring' if auto_refresh else 'Paused'}")


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

# --- TABS ---
tab_live, tab_backtest = st.tabs(["🚀 Live Terminal", "🧪 Backtester"])

with tab_live:
    # --- METRICS DASHBOARD ---
    acc = state.get('account', {})
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        equity = acc.get('equity', 0)
        pnl = acc.get('return_pct', 0)
        glass_metric_card("Total Equity", f"${equity:,.2f}", f"{pnl:+.2f}% Lifetime", delta=pnl)
    
    with m2:
        buying_power = acc.get('buying_power', 0)
        cash = acc.get('cash', 0)
        glass_metric_card("Available Capital", f"${buying_power:,.2f}", f"Cash: ${cash:,.2f}")
    
    with m3:
        active_pos = len(state.get('positions', []))
        glass_metric_card("Active Positions", str(active_pos), "Open Orders")
        
    with m4:
        daily_pnl = acc.get('daily_pnl', 0.0)
        glass_metric_card("Daily P&L", f"${daily_pnl:,.2f}", "Today's Realized", delta=daily_pnl)

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
                decreasing_line_color='#f87171',
                name="Price"
            )])
            
            # Add Trade Markers
            chart_trades = [t for t in state.get('trade_history', []) if t['symbol'] == selected_symbol]
            if chart_trades:
                for t in chart_trades:
                    marker_symbol = "triangle-up" if t['side'] == 'BUY' else "triangle-down"
                    marker_color = "#4ade80" if t['side'] == 'BUY' else ("#f87171" if t['side'] in ['SELL', 'SHORT'] else "#fbbf24")
                    
                    if t['side'] == 'CLOSE':
                        marker_symbol = "circle" 
                        pnl_val = t.get('pnl')
                        if pnl_val is not None:
                             marker_color = "#4ade80" if pnl_val >= 0 else "#f87171"
                    
                    fig.add_trace(go.Scatter(
                        x=[t['timestamp']], 
                        y=[t['price']],
                        mode='markers',
                        marker=dict(symbol=marker_symbol, size=10, color=marker_color, line=dict(width=1, color='white')),
                        name=f"{t['side']} {t['quantity']}",
                        hovertext=f"{t['side']} {t['quantity']} @ {t['price']}<br>{t.get('notes', '')}"
                    ))
            
            fig.update_layout(
                xaxis_rangeslider_visible=False,
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)', 
                margin=dict(l=0, r=0, t=0, b=0),
                height=450,
                xaxis=dict(showgrid=False, color='#64748b'),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', color='#64748b')
            )
            st.plotly_chart(fig)
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

        # Ensemble Breakdown
        ensemble_data = preds.get('ensemble')
        if ensemble_data and isinstance(ensemble_data, dict):
            xgb_val = ensemble_data.get('xgboost', 0.0)
            lstm_val = ensemble_data.get('lstm', 0.0)
            
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; margin-top:-10px; margin-bottom:15px; gap:10px;">
                <div class="metric-card" style="flex:1; text-align:center; padding:12px; background:rgba(30, 41, 59, 0.6); margin-bottom:0;">
                    <div style="font-size:0.75rem; color:#94a3b8; letter-spacing:1px;">XGBOOST</div>
                    <div style="font-size:1.2rem; font-weight:700; color:{'#4ade80' if xgb_val > 0.5 else '#f87171'};">
                        {xgb_val:.1%}
                    </div>
                </div>
                <div class="metric-card" style="flex:1; text-align:center; padding:12px; background:rgba(30, 41, 59, 0.6); margin-bottom:0;">
                    <div style="font-size:0.75rem; color:#94a3b8; letter-spacing:1px;">LSTM</div>
                    <div style="font-size:1.2rem; font-weight:700; color:{'#4ade80' if lstm_val > 0.5 else '#f87171'};">
                        {lstm_val:.1%}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Active Position Card
        my_pos = next((p for p in state.get('positions', []) if p['symbol'] == selected_symbol), None)
        if my_pos:
            entry_price = my_pos.get('entry_price', 0)
            qty = my_pos['qty']
            position_value = qty * entry_price
            
            st.markdown(f"""
                <div class="metric-card" style="border-left:4px solid #38bdf8;">
                    <div style="color:#38bdf8; font-weight:700; font-size:0.8rem;">ACTIVE POSITION</div>
                    <div style="font-size:1.5rem; font-weight:700; color:#fff;">{my_pos['side'].upper()}</div>
                    <div style="color:#e2e8f0; font-size:1.1rem; font-weight:600;">${position_value:,.2f} Value</div>
                    <div style="color:#94a3b8; font-size:0.8rem;">{qty:.5f} Units @ {entry_price:,.2f}</div>
                </div>
            """, unsafe_allow_html=True)


    # --- LIVE FEED ---
    st.markdown("### 📜 Algo Feed")
    trades = state.get('trade_history', [])

    if not trades:
        st.caption("AI is scanning... No trades executed this session yet.")
    else:
        for t in reversed(trades[-10:]): # Show last 10
            side = t['side'].upper()
            pnl_val = t.get('pnl')
            price = t['price']
            qty = t['quantity']
            trade_value = qty * price
            
            if side == 'BUY':
                color = "#4ade80"
                icon = "↗"
                label = "BUY ENTRY"
            elif side in ['SELL', 'SHORT']:
                color = "#f87171" 
                icon = "↘"
                label = "SELL ENTRY"
            elif side == 'CLOSE':
                icon = "✓"
                if pnl_val and pnl_val >= 0:
                    color = "#4ade80"
                    label = f"PROFIT (+${pnl_val:.2f})"
                else:
                    color = "#f87171"
                    label = f"LOSS (${pnl_val:.2f})" if pnl_val else "CLOSE"
            else:
                color = "#94a3b8"
                icon = "•"
                label = side
                
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.02); border-bottom:1px solid rgba(255,255,255,0.05); padding:12px; display:flex; align-items:center; justify-content:space-between; margin-bottom:5px; border-radius:8px;">
                <div style="display:flex; align-items:center; gap:15px;">
                    <div style="font-size:1.2rem; background:{color}20; width:40px; height:40px; display:flex; align-items:center; justify-content:center; border-radius:50%; color:{color};">
                        {icon}
                    </div>
                    <div>
                        <div style="font-weight:700; color:#e2e8f0;">{t['symbol']}</div>
                        <div style="font-size:0.8rem; color:{color}; font-weight:600;">{label}</div>
                    </div>
                </div>
                <div style="text-align:right;">
                    <div style="font-weight:600; color:#cbd5e1;">${trade_value:,.2f}</div>
                    <div style="font-size:0.8rem; color:#64748b;">{qty:.5f} Units @ {price:,.0f}</div>
                    <div style="font-size:0.7rem; color:#475569;">{t['timestamp'][11:19]}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # --- PERFORMANCE ANALYTICS ---
    st.markdown("### 📊 Performance Analytics")

    if trades:
        # 1. Calculate Stats
        closed_trades = [t for t in trades if t['side'] == 'CLOSE']
        total_trades = len(closed_trades)
        
        if total_trades > 0:
            realized_pnls = [t.get('pnl', 0) for t in closed_trades]
            total_pnl = sum(realized_pnls)
            winning_trades = len([p for p in realized_pnls if p > 0])
            win_rate = winning_trades / total_trades
        else:
            total_pnl = 0.0
            win_rate = 0.0
            
        # 2. Display Stats Row
        p1, p2, p3 = st.columns(3)
        
        with p1:
            color = "#4ade80" if total_pnl >= 0 else "#f87171"
            st.markdown(f"""
                <div class="metric-card">
                    <div class="sub-stat">Total Realized P&L</div>
                    <div class="big-stat" style="color:{color};">${total_pnl:,.2f}</div>
                    <div class="sub-stat">All Time Closed Trades</div>
                </div>
            """, unsafe_allow_html=True)
            
        with p2:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="sub-stat">Win Rate</div>
                    <div class="big-stat" style="color:#e2e8f0;">{win_rate:.1%}</div>
                    <div class="sub-stat">{winning_trades}/{total_trades} Trades Won</div>
                </div>
            """, unsafe_allow_html=True)
            
        with p3:
             st.markdown(f"""
                <div class="metric-card">
                    <div class="sub-stat">Total Executions</div>
                    <div class="big-stat" style="color:#e2e8f0;">{len(trades)}</div>
                    <div class="sub-stat">Includes Entry & Exit</div>
                </div>
            """, unsafe_allow_html=True)

        # 3. Detailed History Table
        st.markdown("#### 📜 Trade Log")
        
        # Prepare DataFrame
        history_data = []
        for t in reversed(trades):
            history_data.append({
                "Time": t['timestamp'][11:19],
                "Symbol": t['symbol'],
                "Side": t['side'],
                "Qty": f"{t['quantity']:.5f}",
                "Price": f"${t['price']:,.2f}",
                "Value": f"${(t['quantity'] * t['price']):,.2f}",
                "P&L": f"${t.get('pnl', 0):,.2f}" if t['side'] == 'CLOSE' else "-"
            })
            
        df_history = pd.DataFrame(history_data)
        st.dataframe(df_history, hide_index=True)

    else:
        st.info("No trading history available yet.")

    # Refresh Loop
    if auto_refresh:
        time.sleep(1)
        st.rerun()

with tab_backtest:
    st.markdown("### 🧪 Strategy Backtester")
    
    # Imports
    try:
        from utils.backtest_utils import run_backtest_session
    except ImportError:
        st.error("Backtest Utilities not found.")
        run_backtest_session = None

    col_bt1, col_bt2 = st.columns([1, 2])
    
    with col_bt1:
        with st.form("backtest_config"):
            bt_symbol = st.selectbox("Symbol", state.get('symbols', ["R_75", "CRASH500", "BOOM1000"]))
            bt_capital = st.number_input("Initial Capital ($)", value=200.0)
            bt_risk = st.slider("Risk Per Trade", 0.01, 0.10, 0.01)
            
            run_bt = st.form_submit_button("Running Simulation 🚀")
            
    with col_bt2:
        if run_bt and run_backtest_session:
            # Load Data
            df_bt = load_data(bt_symbol)
            if df_bt is not None:
                with st.spinner("Simulating Strategy..."):
                    results, processed_df = run_backtest_session(bt_symbol, df_bt, bt_capital, bt_risk)
                    
                st.success("Backtest Complete")
                
                # --- HUD (Heads-Up Display) ---
                col1, col2, col3, col4 = st.columns(4)
                
                # Placeholder values for demonstration as actual variables (eq, avail, cash, num_positions, bot.daily_pnl)
                # are not defined in this backtest context.
                # Assuming these are meant for a "Live Terminal" section, but placed here as per instruction.
                eq = results['final_portfolio_value'] if 'final_portfolio_value' in results else bt_capital
                avail = results['final_cash'] if 'final_cash' in results else bt_capital
                cash = avail # Assuming cash is same as available capital for this context
                num_positions = results['total_trades'] if 'total_trades' in results else 0
                pnl_val = results['total_return_dollar'] if 'total_return_dollar' in results else 0.0

                with col1:
                    glass_metric_card("Total Equity", f"${eq:.2f}", f"{((eq - bt_capital)/bt_capital)*100:.2f}% Lifetime")
                with col2:
                    glass_metric_card("Available Capital", f"${avail:.2f}", f"Cash: ${cash:.2f}")
                with col3:
                    glass_metric_card("Active Positions", str(num_positions), "Total Trades")
                with col4:
                    glass_metric_card("Daily P&L", f"${pnl_val:.2f}", "Simulation P&L")
                    
                st.markdown("---")
                
                # Equity Curve
                st.line_chart(processed_df['Portfolio_Value_Tracked'] if 'Portfolio_Value_Tracked' in processed_df else [])
                
                if results['trades_list'] is not None and not results['trades_list'].empty:
                    st.dataframe(results['trades_list'])
            else:
                st.error("No data found for symbol")

    # --- HYPERPARAMETER LAB ---
    st.markdown("---")
    st.markdown("### 🧬 Hyperparameter Lab")
    
    # Optimizer Imports
    try:
        from utils.optimizer import GridOptimizer
    except ImportError:
        GridOptimizer = None
        st.error("Optimizer Module not found")
        
    with st.expander("Configure Optimization Experiment"):
        st.info("Optimize Stop-Loss and Take-Profit settings for maximum return.")
        
        opt_col1, opt_col2 = st.columns(2)
        with opt_col1:
            opt_symbol = st.selectbox("Optimization Asset", state.get('symbols', ["R_75"]), key="opt_sym")
            opt_capital = st.number_input("Capital", value=1000.0, key="opt_cap")
            
        with opt_col2:
            sl_range = st.multiselect("Stop Loss Candidates (%)", [0.01, 0.02, 0.03, 0.05, 0.10], default=[0.01, 0.02, 0.05])
            tp_range = st.multiselect("Take Profit Candidates (%)", [0.02, 0.04, 0.06, 0.10, 0.20], default=[0.02, 0.04, 0.10])
            
        if st.button("🧪 Run Grid Search"):
            if GridOptimizer:
                df_opt = load_data(opt_symbol)
                if df_opt is not None:
                    with st.spinner(f"Running experiment on {len(sl_range)*len(tp_range)} combinations..."):
                        optimizer = GridOptimizer(df_opt, opt_capital)
                        
                        param_grid = {
                            'sl': sl_range,
                            'tp': tp_range,
                            'risk_per_trade': [0.02] # Fixed risk for now
                        }
                        
                        opt_results = optimizer.optimize(param_grid)
                        
                        st.success("Optimization Complete!")
                        st.dataframe(opt_results.style.format({
                            'Total Return %': '{:.2f}%',
                            'Win Rate': '{:.2f}%',
                            'Max Drawdown': '{:.2f}%',
                            'Sharpe': '{:.2f}'
                        }))
                        
                        best_run = opt_results.iloc[0]
                        st.balloons()
                        st.success(f"🏆 Best Config: SL={best_run['sl']*100}% | TP={best_run['tp']*100}% | Ret={best_run['Total Return %']:.2f}%")
                else:
                    st.error("No data for asset")
