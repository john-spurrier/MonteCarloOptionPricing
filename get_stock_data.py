import yfinance as yf
import pandas as pd
import numpy as np

def fetch_data(ticker='AAPL', period='1y'):
    # Download historical price data
    data = yf.download(ticker, period=period)
    data = data[['Close']]  # Only keep the closing prices
    data.dropna(inplace=True)

    # Calculate log returns
    data['LogReturn'] = np.log(data['Close'] / data['Close'].shift(1))
    data.dropna(inplace=True)

    # Calculate volatility
    daily_std = data['LogReturn'].std()
    annual_volatility = daily_std * np.sqrt(252)

    # Save to CSV
    data.to_csv('stock_data.csv')

    print(f"Downloaded {len(data)} data points for {ticker}.")
    latest_price = data['Close'].iloc[-1].item()
    print(f"Latest Close Price: {latest_price:.2f}")

    print(f"Annualized Volatility: {annual_volatility:.4f}")

    # Also return key values for use in C++
    return latest_price, annual_volatility

if __name__ == "__main__":
    ticker = input("Enter stock ticker (e.g., AAPL): ").upper()
    S0, sigma = fetch_data(ticker)
    with open("params.txt", "w") as f:
        f.write(f"{S0},{sigma}\n")
