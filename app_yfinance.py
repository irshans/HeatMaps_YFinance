import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from scipy.interpolate import interp1d
from datetime import datetime

# --- APP CONFIG ---
st.set_page_config(page_title="Free GEX Dashboard (YFinance)", page_icon="📊", layout="wide")

# --- STYLES ---
CUSTOM_COLORSCALE = [[0.0, '#FF3333'], [0.45, '#121212'], [0.5, '#121212'], [0.55, '#121212'], [1.0, '#00FF7F']]

# --- BLACK-SCHOLES MATH (The "Free" Greeks) ---
def calculate_gamma(S, K, T, r, sigma):
    """Calculates Option Gamma manually since YFinance doesn't provide it."""
    if T <= 0 or sigma <= 0: return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return gamma

def calculate_vega(S, K, T, r, sigma):
    """Calculates Option Vega manually."""
    if T <= 0 or sigma <= 0: return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    return vega

# --- DATA FETCHING ---
def fetch_yfinance_gex(ticker_symbol, exp_count, strike_window):
    # SPX on Yahoo is "^SPX"
    yf_ticker = f"^{ticker_symbol}" if ticker_symbol == "SPX" else ticker_symbol
    tk = yf.Ticker(yf_ticker)
    
    # 1. Get Spot Price
    history = tk.history(period="1d")
    if history.empty:
        st.error(f"Could not find ticker {yf_ticker}")
        return None, None
    S = history['Close'].iloc[-1]
    
    # 2. Get Expirations
    exps = tk.options[:exp_count]
    
    rows = []
    r = 0.045 # Approximate Risk-Free Rate (4.5%)
    
    progress_bar = st.progress(0)
    for i, exp in enumerate(exps):
        opt = tk.option_chain(exp)
        # Combine Calls and Puts
        calls = opt.calls[['strike', 'openInterest', 'impliedVolatility']].assign(type='C')
        puts = opt.puts[['strike', 'openInterest', 'impliedVolatility']].assign(type='P')
        chain = pd.concat([calls, puts])
        
        # Filter Strikes around Spot
        chain = chain[(chain['strike'] >= S * 0.95) & (chain['strike'] <= S * 1.05)]
        
        # Calculate Time to Expiry (in years)
        days_to_expiry = (datetime.strptime(exp, '%Y-%m-%d') - datetime.now()).days
        T = max(days_to_expiry, 0.5) / 365.0
        
        for _, row in chain.iterrows():
            K = row['strike']
            oi = row['openInterest'] if not np.isnan(row['openInterest']) else 10
            iv = row['impliedVolatility']
            
            if iv > 0:
                gamma = calculate_gamma(S, K, T, r, iv)
                vega = calculate_vega(S, K, T, r, iv)
                
                # Dealer GEX: -1 * Gamma * S^2 * 0.01 * OI * 100
                gex = -1 * gamma * (S**2) * 0.01 * oi * 100
                vex = -1 * (vega / S) * S * oi * 100 if S > 0 else 0
                
                fmt_date = datetime.strptime(exp, '%Y-%m-%d').strftime('%a %m/%d/%Y')
                
                rows.append({
                    'expiration': fmt_date,
                    'strike': K,
                    'gex': gex,
                    'vex': vex
                })
        progress_bar.progress((i + 1) / len(exps))
        
    return S, pd.DataFrame(rows)

# --- ZERO GAMMA CALC ---
def find_zero_gamma(df):
    agg = df.groupby('strike')['gex'].sum().sort_index()
    strikes, values = agg.index.values, agg.values
    if np.all(values > 0) or np.all(values < 0): return None
    try:
        f = interp1d(values, strikes, kind='linear', fill_value="extrapolate")
        zg = float(f(0))
        return zg if strikes.min() <= zg <= strikes.max() else None
    except: return None

# --- VISUALIZATION ---
def render_surface(df, spot, zero_gamma, val_col, title):
    pivot = df.pivot_table(index='strike', columns='expiration', values=val_col, aggfunc='sum').fillna(0)
    
    # Star Logic
    flat_idx = np.abs(pivot.values).argmax()
    row_idx, col_idx = np.unravel_index(flat_idx, pivot.shape)
    star_strike, star_exp = pivot.index[row_idx], pivot.columns[col_idx]
    
    text_vals = [[(f"⭐{val/1e6:.1f}M" if pivot.index[i] == star_strike and pivot.columns[j] == star_exp else f"{val/1e6:.1f}M") 
                  for j, val in enumerate(row)] for i, row in enumerate(pivot.values)]

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values, x=pivot.columns, y=pivot.index,
        colorscale=CUSTOM_COLORSCALE, zmid=0, text=text_vals, texttemplate="%{text}",
        textfont={"size": 11, "color": "white"}
    ))
    
    fig.update_yaxes(range=[pivot.index.min(), pivot.index.max()], autorange=False)
    if zero_gamma:
        fig.add_hline(y=zero_gamma, line_dash="dot", line_color="orange", annotation_text="ZERO GAMMA")
    
    fig.add_hline(y=spot, line_dash="dash", line_color="white", annotation_text=f"Spot: {spot:.2f}")
    fig.update_layout(title=title, template="plotly_dark", height=800, xaxis={'type': 'category'})
    return fig

# --- MAIN ---
def main():
    st.sidebar.header("📊 Free GEX (Yahoo)")
    ticker = st.sidebar.text_input("Ticker (e.g. SPX, AAPL)", "SPX").upper()
    exp_days = st.sidebar.slider("Expiries", 1, 8, 3)
    
    if st.sidebar.button("Run Analysis", type="primary"):
        with st.spinner("Fetching Data and Calculating Greeks..."):
            spot, df = fetch_yfinance_gex(ticker, exp_days, 15)
        
        if df is not None and not df.empty:
            total_gex = df['gex'].sum()
            zero_gamma = find_zero_gamma(df)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Total GEX", f"{total_gex/1e6:.2f}M")
            if zero_gamma: c2.metric("Zero Gamma", f"{zero_gamma:.2f}")
            c3.subheader("🟢 LONG GAMMA" if total_gex > 0 else "🔴 SHORT GAMMA")

            t1, t2 = st.tabs(["Gamma Exposure", "Vanna Exposure"])
            with t1: st.plotly_chart(render_surface(df, spot, zero_gamma, 'gex', "Dealer Gamma"), use_container_width=True)
            with t2: st.plotly_chart(render_surface(df, spot, None, 'vex', "Dealer Vanna"), use_container_width=True)
        else:
            st.error("Data not found. Try a common ticker like AAPL or TSLA.")

if __name__ == "__main__":
    main()