import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import subprocess
import datetime
import matplotlib.pyplot as plt
import io
import time

st.set_page_config(page_title="Option Pricing Dashboard", layout="wide")

st.title("📈 Monte Carlo Option Pricing Dashboard")

# --- Sidebar Inputs ---
ticker = st.sidebar.text_input("Stock Ticker", value="AAPL")
option_types = st.sidebar.multiselect("Option Types", ["Call", "Put", "Digital"], default=["Call"])
strike = st.sidebar.number_input("Strike Price", value=100.0)
num_simulations = st.sidebar.slider("Number of Simulations", 1000, 1000000, 100000, step=1000)

# --- Additional inputs ---
with st.sidebar.expander("Advanced Parameters"):
    vr_technique = st.selectbox(
        "Variance Reduction Technique",
        ["None", "Antithetic Variates", "Control Variates", "Both"],
        index=0,
        help="Select a variance reduction technique to improve simulation efficiency"
    )
    
    # Map UI selection to command-line args
    vr_mapping = {
        "None": "none",
        "Antithetic Variates": "antithetic",
        "Control Variates": "control", 
        "Both": "both"
    }
    
    benchmark_mode = st.checkbox(
        "Run Benchmarks", 
        value=False,
        help="Compare different variance reduction techniques"
    )

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

# === Run benchmarks if selected ===
if benchmark_mode:
    st.subheader("⚡ Performance Benchmarking")
    st.write("Comparing different variance reduction techniques...")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Execute benchmark mode for each option type
    benchmark_results = {}
    for i, option_type_str in enumerate(option_types):
        option_type_flag = option_type_str.lower()
        status_text.text(f"Running benchmarks for {option_type_str} options...")
        
        cpp_command = ["./monte_carlo_option", "10000", "--benchmark", option_type_flag, f"--K={strike}"]
        try:
            result = subprocess.run(cpp_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            benchmark_results[option_type_str] = result.stdout
            progress_bar.progress((i + 1) / len(option_types))
        except Exception as e:
            st.error(f"Error running benchmarks for {option_type_str}: {e}")
    
    progress_bar.progress(100)
    status_text.text("Benchmark completed!")
    
    # Parse and display benchmark results
    for option_type, result_text in benchmark_results.items():
        st.subheader(f"Benchmark Results - {option_type} Options")
        st.text(result_text)
        
        # Parse and convert to DataFrame for better visualization
        lines = result_text.strip().split('\n')
        header_index = next((i for i, line in enumerate(lines) if "Simulations" in line), None)
        
        if header_index is not None:
            # Skip separator lines and extract data rows
            data_rows = []
            columns = lines[header_index].split()
            
            for line in lines[header_index+2:]:
                if "---------" in line:
                    continue
                if line.strip():  # Skip empty lines
                    # Handle multi-word techniques
                    parts = line.split()
                    if len(parts) >= 6:
                        # Parse based on known pattern
                        if "Control" in line and "Variate" in line:
                            # Handle "Control Variate" as a single term
                            technique = "Control Variate"
                            sim_size = parts[0]
                            price = parts[3]
                            std_dev = parts[4]
                            abs_error = parts[5]
                            time_ms = parts[6]
                            efficiency = parts[7] if len(parts) > 7 else "N/A"
                        elif parts[1] == "Both":
                            # Handle "Both (Antithetic + Control)" case
                            technique = "Both"
                            sim_size = parts[0]
                            # Find values by position after parenthesis
                            rest_parts = ' '.join(parts[1:]).split(')')
                            if len(rest_parts) > 1:
                                value_parts = rest_parts[1].strip().split()
                                if len(value_parts) >= 4:
                                    price = value_parts[0]
                                    std_dev = value_parts[1]
                                    abs_error = value_parts[2]
                                    time_ms = value_parts[3]
                                    efficiency = value_parts[4] if len(value_parts) > 4 else "N/A"
                                else:
                                    continue
                            else:
                                continue
                        else:
                            # Normal case
                            sim_size = parts[0]
                            technique = parts[1]
                            price = parts[2]
                            std_dev = parts[3]
                            abs_error = parts[4]
                            time_ms = parts[5]
                            efficiency = parts[6] if len(parts) > 6 else "N/A"
                        
                        data_rows.append({
                            "Simulations": sim_size,
                            "Technique": technique,
                            "Price": price,
                            "StdDev": std_dev,
                            "AbsError": abs_error,
                            "Time(ms)": time_ms,
                            "Efficiency": efficiency
                        })
            
            if data_rows:
                df = pd.DataFrame(data_rows)
                df["Simulations"] = df["Simulations"].astype(int)
                df["Price"] = df["Price"].astype(float)
                df["StdDev"] = df["StdDev"].astype(float)
                df["AbsError"] = df["AbsError"].astype(float)
                df["Time(ms)"] = df["Time(ms)"].astype(float)
                
                # Create plots
                col1, col2 = st.columns(2)
                with col1:
                    fig1, ax1 = plt.subplots(figsize=(10, 6))
                    techniques = df["Technique"].unique()
                    sim_sizes = df["Simulations"].unique()
                    
                    for technique in techniques:
                        tech_data = df[df["Technique"] == technique]
                        ax1.plot(tech_data["Simulations"], tech_data["AbsError"], marker='o', label=f'{technique}')
                    
                    ax1.set_xlabel("Number of Simulations")
                    ax1.set_ylabel("Absolute Error")
                    ax1.set_title("Error vs. Simulation Size")
                    ax1.set_xscale("log")
                    ax1.set_yscale("log")
                    ax1.legend()
                    ax1.grid(True)
                    st.pyplot(fig1)
                
                with col2:
                    fig2, ax2 = plt.subplots(figsize=(10, 6))
                    for technique in techniques:
                        tech_data = df[df["Technique"] == technique]
                        ax2.plot(tech_data["Simulations"], tech_data["Time(ms)"], marker='o', label=f'{technique}')
                    
                    ax2.set_xlabel("Number of Simulations")
                    ax2.set_ylabel("Execution Time (ms)")
                    ax2.set_title("Performance vs. Simulation Size")
                    ax2.set_xscale("log")
                    ax2.legend()
                    ax2.grid(True)
                    st.pyplot(fig2)
                
                # Efficiency comparison
                st.subheader("Efficiency Comparison")
                st.write("Higher efficiency means better variance reduction per unit of computation time")
                
                # Create a normalized efficiency chart for the largest simulation size
                largest_sim = df["Simulations"].max()
                largest_sim_data = df[df["Simulations"] == largest_sim].copy()
                
                if not largest_sim_data.empty:
                    # Calculate normalized efficiency (relative to no variance reduction)
                    base_efficiency = largest_sim_data[largest_sim_data["Technique"] == "None"]["Time(ms)"].values[0]
                    largest_sim_data["Normalized_Efficiency"] = base_efficiency / largest_sim_data["Time(ms)"] * \
                                                                largest_sim_data[largest_sim_data["Technique"] == "None"]["AbsError"].values[0] / \
                                                                largest_sim_data["AbsError"]
                    
                    fig3, ax3 = plt.subplots(figsize=(10, 6))
                    bars = ax3.bar(largest_sim_data["Technique"], largest_sim_data["Normalized_Efficiency"])
                    
                    # Add value labels on top of bars
                    for bar in bars:
                        height = bar.get_height()
                        ax3.annotate(f'{height:.2f}',
                                    xy=(bar.get_x() + bar.get_width() / 2, height),
                                    xytext=(0, 3),  # 3 points vertical offset
                                    textcoords="offset points",
                                    ha='center', va='bottom')
                    
                    ax3.set_ylabel("Normalized Efficiency (higher is better)")
                    ax3.set_title(f"Relative Efficiency at {largest_sim} Simulations")
                    ax3.grid(True, axis='y')
                    st.pyplot(fig3)
                
                # Display the raw data
                st.dataframe(df)
    
    # Add download button for the benchmark data
    benchmark_csv = pd.DataFrame()
    for option_type, result_text in benchmark_results.items():
        lines = result_text.strip().split('\n')
        header_index = next((i for i, line in enumerate(lines) if "Simulations" in line), None)
        
        if header_index is not None:
            data_rows = []
            for line in lines[header_index+2:]:
                if "---------" in line:
                    continue
                if line.strip():
                    # Handle multi-word techniques
                    parts = line.split()
                    if len(parts) >= 6:
                        # Parse based on known pattern
                        if "Control" in line and "Variate" in line:
                            # Handle "Control Variate" as a single term
                            technique = "Control Variate"
                            sim_size = parts[0]
                            price = parts[3]
                            std_dev = parts[4]
                            abs_error = parts[5]
                            time_ms = parts[6]
                            efficiency = parts[7] if len(parts) > 7 else "N/A"
                        elif parts[1] == "Both":
                            # Handle "Both (Antithetic + Control)" case
                            technique = "Both"
                            sim_size = parts[0]
                            # Find values by position after parenthesis
                            rest_parts = ' '.join(parts[1:]).split(')')
                            if len(rest_parts) > 1:
                                value_parts = rest_parts[1].strip().split()
                                if len(value_parts) >= 4:
                                    price = value_parts[0]
                                    std_dev = value_parts[1]
                                    abs_error = value_parts[2]
                                    time_ms = value_parts[3]
                                    efficiency = value_parts[4] if len(value_parts) > 4 else "N/A"
                                else:
                                    continue
                            else:
                                continue
                        else:
                            # Normal case
                            sim_size = parts[0]
                            technique = parts[1]
                            price = parts[2]
                            std_dev = parts[3]
                            abs_error = parts[4]
                            time_ms = parts[5]
                            efficiency = parts[6] if len(parts) > 6 else "N/A"
                        
                        row_data = {
                            "Option Type": option_type,
                            "Simulations": sim_size,
                            "Technique": technique,
                            "Price": price,
                            "StdDev": std_dev,
                            "AbsError": abs_error,
                            "Time(ms)": time_ms,
                            "Efficiency": efficiency
                        }
                        data_rows.append(row_data)
            
            if data_rows:
                option_df = pd.DataFrame(data_rows)
                benchmark_csv = pd.concat([benchmark_csv, option_df])
    
    if not benchmark_csv.empty:
        csv_buffer = io.StringIO()
        benchmark_csv.to_csv(csv_buffer, index=False)
        st.download_button("Download Benchmark Results", csv_buffer.getvalue(), file_name="benchmark_results.csv", mime="text/csv")

# --- Run C++ Monte Carlo Simulator for each option type ---
if not benchmark_mode:
    simulation_sizes = [1000, 5000, 10000, 50000, 100000, 200000, 500000, num_simulations]
    results = {}

    for option_type_str in option_types:
        option_type_flag = option_type_str.lower()
        prices = []
        lowers = []
        uppers = []
        times = []
        
        for sim in simulation_sizes:
            cpp_command = ["./monte_carlo_option", str(sim), "--csv", option_type_flag, f"--K={strike}", f"--vr={vr_mapping[vr_technique]}"]
            try:
                result = subprocess.run(cpp_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                
                for line in result.stdout.splitlines():
                    if "," in line:
                        parts = line.strip().split(",")
                        if len(parts) >= 5:  # Make sure we have enough parts
                            prices.append(float(parts[1]))
                            lowers.append(float(parts[2]))
                            uppers.append(float(parts[3]))
                            times.append(float(parts[4]))
            except Exception as e:
                st.error(f"Error running Monte Carlo for {option_type_str}: {e}")
                st.stop()
        results[option_type_str] = (prices, lowers, uppers, times)

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
    st.write(f"Using Variance Reduction: **{vr_technique}**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Convergence plot
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        for option_type_str in option_types:
            prices, lowers, uppers, _ = results[option_type_str]
            ax1.plot(simulation_sizes, prices, marker='o', label=f'{option_type_str} Price')
            ax1.fill_between(simulation_sizes, lowers, uppers, alpha=0.2, label=f'{option_type_str} 95% CI')
            
        if market_price is not None and ("Call" in option_types or "Put" in option_types):
            ax1.axhline(market_price, color='red', linestyle='--', label='Market Price')
            
        ax1.set_xscale("log")
        ax1.set_xlabel("Number of Simulations")
        ax1.set_ylabel("Option Price")
        ax1.set_title("Convergence of Monte Carlo Estimates")
        ax1.legend()
        ax1.grid(True)
        st.pyplot(fig1)
    
    with col2:
        # Performance plot
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        for option_type_str in option_types:
            _, _, _, times = results[option_type_str]
            if len(times) == len(simulation_sizes):  # Ensure data alignment
                ax2.plot(simulation_sizes, times, marker='o', label=f'{option_type_str}')
            
        ax2.set_xscale("log")
        ax2.set_yscale("log")  # Log scale for better visualization of time differences
        ax2.set_xlabel("Number of Simulations")
        ax2.set_ylabel("Execution Time (ms)")
        ax2.set_title("Performance vs. Simulation Size")
        ax2.legend()
        ax2.grid(True)
        st.pyplot(fig2)
    
    # Display final results
    st.subheader("Final Results")
    result_df = pd.DataFrame()
    
    for option_type_str in option_types:
        prices, lowers, uppers, times = results[option_type_str]
        if len(prices) > 0:  # Only create data if results exist
            last_idx = len(prices) - 1
            
            # Calculate relative error if market price is available
            rel_error = ""
            if market_price is not None and option_type_str in ["Call", "Put"]:
                abs_error = abs(prices[last_idx] - market_price)
                rel_error = f"{abs_error:.4f} ({(abs_error / market_price * 100):.2f}%)"
            
            # Add to dataframe
            result_df = pd.concat([result_df, pd.DataFrame({
                "Option Type": [option_type_str],
                "Simulations": [simulation_sizes[last_idx]],
                "Price": [f"${prices[last_idx]:.4f}"],
                "95% CI": [f"[{lowers[last_idx]:.4f}, {uppers[last_idx]:.4f}]"],
                "Execution Time": [f"{times[last_idx]:.2f} ms"],
                "Difference from Market": [rel_error]
            })])
    
    if not result_df.empty:
        st.dataframe(result_df, use_container_width=True)
    
    # --- Export Results to CSV ---
    combined_csv = []
    for option_type_str in option_types:
        prices, lowers, uppers, times = results[option_type_str]
        for sim, price, low, high, time_ms in zip(simulation_sizes, prices, lowers, uppers, times):
            combined_csv.append({
                "Option Type": option_type_str,
                "Simulations": sim,
                "Variance Reduction": vr_technique,
                "Simulated Price": price,
                "Lower CI": low,
                "Upper CI": high,
                "Execution Time (ms)": time_ms
            })

    csv_data = pd.DataFrame(combined_csv)
    csv_buffer = io.StringIO()
    csv_data.to_csv(csv_buffer, index=False)
    st.download_button("Download Results as CSV", csv_buffer.getvalue(), file_name="monte_carlo_results.csv", mime="text/csv")