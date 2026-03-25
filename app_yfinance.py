import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from scipy.interpolate import interp1d
from datetime import datetime

# --- APP CONFIG ---
st.set_page_config(page_title="GEX Dashboard Pro", page_icon="📊", layout="wide")

# --- STYLES ---
CUSTOM_COLORSCALE = [[0.0, '#FF3333'], [0.45, '#121212'], [0.5, '#121212'], [0.55, '#121212'], [1.0, '#00FF7F']]

# --- BLACK-SCHOLES MATH ---
def calculate_gamma(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0: return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

def calculate_vega(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0: return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)

# --- DATA FETCHING ---
def fetch_yfinance_gex(ticker_symbol, exp_count, strike_count):
    yf_ticker = f"^{ticker_symbol}" if ticker_symbol == "SPX" else ticker_symbol
    tk = yf.Ticker(yf_ticker)
    
    history = tk.history(period="1d")
    if history.empty: return None, None
    S = history['Close'].iloc[-1]
    
    exps = tk.options[:exp_count]
    rows = []
    r = 0.045 
    
    progress_bar = st.progress(0)
    for i, exp in enumerate(exps):
        try:
            opt = tk.option_chain(exp)
            calls = opt.calls[['strike', 'openInterest', 'impliedVolatility']].assign(type='C')
            puts = opt.puts[['strike', 'openInterest', 'impliedVolatility']].assign(type='P')
            chain = pd.concat([calls, puts])
            
            # Filter to specific number of strikes around spot
            all_strikes = sorted(chain['strike'].unique())
            idx = np.searchsorted(all_strikes, S)
            start_idx = max(0, idx - strike_count)
            end_idx = min(len(all_strikes), idx + strike_count)
            valid_strikes = all_strikes[start_idx:end_idx]
            
            chain = chain[chain['strike'].isin(valid_strikes)]
            
            T = max((datetime.strptime(exp, '%Y-%m-%d') - datetime.now()).days, 0.5) / 365.0
            
            for _, row in chain.iterrows():
                K, oi, iv = row['strike'], row['openInterest'], row['impliedVolatility']
                if not np.isnan(oi) and iv > 0:
                    gamma = calculate_gamma(S, K, T, r, iv)
                    vega = calculate_vega(S, K, T, r, iv)
                    rows.append({
                        'expiration': datetime.strptime(exp, '%Y-%m-%d').strftime('%a %m/%d/%Y'),
                        'strike': K,
                        'gex': -1 * gamma * (S**2) * 0.01 * oi * 100,
                        'vex': -1 * (vega / S) * S * oi * 100 if S > 0 else 0
                    })
        except: continue
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
def render_surface(df, spot, zero_gamma, val_col, title, interval):
    pivot = df.pivot_table(index='strike', columns='expiration', values=val_col, aggfunc='sum').fillna(0)
    
    # Text Annotations for Highest Exposure
    flat_idx = np.abs(pivot.values).argmax()
    row_idx, col_idx = np.unravel_index(flat_idx, pivot.shape)
    star_strike, star_exp = pivot.index[row_idx], pivot.columns[col_idx]
    
    text_vals = [[(f"⭐{val/1e6:.1f}M" if pivot.index[i] == star_strike and pivot.columns[j] == star_exp else f"{val/1e6:.1f}M") 
                  for j, val in enumerate(row)] for i, row in enumerate(pivot.values)]

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values, x=pivot.columns, y=pivot.index,
        colorscale=CUSTOM_COLORSCALE, zmid=0, text=text_vals, texttemplate="%{text}",
        textfont={"size": 10, "color": "white"}
    ))
    
    # Apply Interval Slider logic to Axis
    fig.update_yaxes(
        tickmode='linear',
        tick0=pivot.index.min(),
        dtick=interval,
        range=[pivot.index.min(), pivot.index.max()],
        autorange=False,
        title="Strike Price"
    )

    if zero_gamma:
        fig.add_hline(y=zero_gamma, line_dash="dot", line_color="orange", line_width=3,
                      annotation_text=f"ZERO GAMMA: {zero_gamma:.1f}", annotation_position="bottom left")
    
    fig.add_hline(y=spot, line_dash="dash", line_color="white", line_width=2,
                  annotation_text=f"SPOT: {spot:.2f}", annotation_position="top left")
    
    fig.update_layout(
        title=title, 
        template="plotly_dark", 
        height=900, 
        xaxis={'type': 'category', 'title': 'Expiration Date'}
    )
    return fig

# --- MAIN ---
def main():
    st.sidebar.header("📊 Parameters")
    ticker = st.sidebar.text_input("Ticker", "SPX").upper()
    exp_days = st.sidebar.slider("Expirations", 1, 10, 3)
    strike_count = st.sidebar.slider("Strikes (+/- Spot)", 5, 50, 20)
    
    # NEW: Interval Slider
    interval = st.sidebar.select_slider(
        "Label Interval", 
        options=[1, 2, 5, 10, 25, 50], 
        value=5 if ticker == "SPX" else 1
    )
    
    if st.sidebar.button("Run Analysis", type="primary"):
        with st.spinner(f"Analyzing {ticker}..."):
            spot, df = fetch_yfinance_gex(ticker, exp_days, strike_count)
        
        if df is not None and not df.empty:
            total_gex = df['gex'].sum()
            zero_gamma = find_zero_gamma(df)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Net GEX", f"{total_gex/1e6:.2f}M")
            if zero_gamma:
                dist = ((spot/zero_gamma)-1)*100
                c2.metric("Zero Gamma", f"{zero_gamma:.1f}", f"{dist:.2f}% from Spot")
            c3.subheader("🟢 LONG GAMMA" if total_gex > 0 else "🔴 SHORT GAMMA")

            t1, t2 = st.tabs(["Gamma Exposure (GEX)", "Vanna Exposure (VEX)"])
            with t1:
                st.plotly_chart(render_surface(df, spot, zero_gamma, 'gex', f"{ticker} Gamma Surface", interval), use_container_width=True)
            with t2:
                st.plotly_chart(render_surface(df, spot, None, 'vex', f"{ticker} Vanna Surface", interval), use_container_width=True)
        else:
            st.error("No data found. Ensure ticker and internet connection are valid.")

if __name__ == "__main__":
    main()