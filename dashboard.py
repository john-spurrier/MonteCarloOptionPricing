import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import subprocess
import datetime
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="Option Pricing Dashboard", layout="wide")

st.title("📈 Monte Carlo Option Pricing Dashboard")

# --- Sidebar Inputs ---
ticker = st.sidebar.text_input("Stock Ticker", value="AAPL")
option_types = st.sidebar.multiselect("Option Types", ["Call", "Put", "Digital"], default=["Call"])
strike = st.sidebar.number_input("Strike Price", value=100.0)
num_simulations = st.sidebar.slider("Number of Simulations", 1000, 1000000, 100000, step=1000)

# --- Fetch stock object ---
try:
    stock = yf.Ticker(ticker)
    current_price = stock.info.get("regularMarketPrice", None)
    if current_price is None:
        raise ValueError("Could not fetch current price.")
    st.write(f"**Current Price for {ticker}:** ${current_price:.2f}")
except Exception as e:
    st.error(f"Could not fetch stock data: {e}")
    st.stop()

# --- Fetch expiration dates ---
try:
    expirations = stock.options
    if not expirations:
        raise ValueError("No available option expirations for this ticker.")
    expiry = st.sidebar.selectbox("Expiration Date", expirations)
except Exception as e:
    st.error(f"Could not fetch option expiration dates: {e}")
    st.stop()

# --- Save params.txt for C++ backend ---
hist = stock.history(period="1y")
returns = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
volatility = returns.std() * np.sqrt(252)

with open("params.txt", "w") as f:
    f.write(f"{current_price},{volatility}\n")

# --- Run C++ Monte Carlo Simulator for each option type ---
simulation_sizes = [1000, 5000, 10000, 50000, 100000, 200000, 500000, num_simulations]
results = {}

for option_type_str in option_types:
    option_type_flag = option_type_str.lower()
    prices = []
    lowers = []
    uppers = []
    for sim in simulation_sizes:
        cpp_command = ["./monte_carlo_option", str(sim), "--csv", option_type_flag, f"--K={strike}"]
        try:
            # st.write(f"Running: {cpp_command}")
            result = subprocess.run(cpp_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            # Show debug info in UI
            # if result.stderr:
            #     st.subheader(f"⚙️ Debug Output for {option_type_str}")
            #     st.code(result.stderr, language="bash")

            for line in result.stdout.splitlines():
                if "," in line:
                    parts = line.strip().split(",")
                    prices.append(float(parts[1]))
                    lowers.append(float(parts[2]))
                    uppers.append(float(parts[3]))
        except Exception as e:
            st.error(f"Error running Monte Carlo for {option_type_str}: {e}")
            st.stop()
    results[option_type_str] = (prices, lowers, uppers)

# --- Display Market Option Price ---
try:
    options_df = stock.option_chain(str(expiry))
    st.subheader("Market Option Chain")
    st.dataframe(options_df.calls if "Call" in option_types else options_df.puts)
    if "Call" in option_types or "Put" in option_types:
        opt_data = options_df.calls if "Call" in option_types else options_df.puts
        closest_row = opt_data.iloc[(opt_data['strike'] - strike).abs().argsort()[:1]]
        market_price = closest_row['lastPrice'].values[0]
        st.write(f"**Market Price for Closest Option:** ${market_price:.2f}")
    else:
        market_price = None
except Exception as e:
    st.warning(f"Could not load option chain: {e}")
    market_price = None

# --- Display Simulation Result and Plot ---
st.subheader("Monte Carlo Estimates and Convergence")
fig, ax = plt.subplots(figsize=(10, 5))
for option_type_str in option_types:
    prices, lowers, uppers = results[option_type_str]
    ax.plot(simulation_sizes, prices, marker='o', label=f'{option_type_str} Price')
    ax.fill_between(simulation_sizes, lowers, uppers, alpha=0.2, label=f'{option_type_str} 95% CI')
    st.write(f"**{option_type_str} Final Simulated Price:** ${prices[-1]:.4f}")
    st.write(f"**{option_type_str} 95% CI:** [{lowers[-1]:.4f}, {uppers[-1]:.4f}]")
    if market_price is not None and option_type_str in ["Call", "Put"]:
        st.write(f"**{option_type_str} Difference from Market:** ${prices[-1] - market_price:.4f}")

if market_price:
    ax.axhline(market_price, color='red', linestyle='--', label='Market Price')
ax.set_xscale("log")
ax.set_xlabel("Number of Simulations")
ax.set_ylabel("Option Price")
ax.set_title("Convergence of Monte Carlo Estimates")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# --- Export Results to CSV ---
combined_csv = []
for option_type_str in option_types:
    prices, lowers, uppers = results[option_type_str]
    for sim, price, low, high in zip(simulation_sizes, prices, lowers, uppers):
        combined_csv.append({
            "Option Type": option_type_str,
            "Simulations": sim,
            "Simulated Price": price,
            "Lower CI": low,
            "Upper CI": high
        })

csv_data = pd.DataFrame(combined_csv)
csv_buffer = io.StringIO()
csv_data.to_csv(csv_buffer, index=False)
st.download_button("Download Results as CSV", csv_buffer.getvalue(), file_name="monte_carlo_results.csv", mime="text/csv")