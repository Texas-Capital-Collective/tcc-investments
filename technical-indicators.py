import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Get historical stock prices using yfinance
def get_historical_prices(symbol, start_date, end_date):
    df = yf.download(symbol, start=start_date, end=end_date)
    
    if df.empty:
        print(f"❌ No data found for {symbol}.")
        return None
    
    df = df.reset_index()[["Date", "Close"]]  # Keep only Date & Close Price
    return df

# Calculate SMA
def get_sma(symbol, start_date, end_date, window):
    df = get_historical_prices(symbol, start_date, end_date)
    if df is None or df.empty:
        return None
    df["SMA"] = df["Close"].rolling(window=window).mean()
    return df

# Calculate Bollinger Bands to show volatility Overbought/Oversold
def get_bollinger_bands(symbol, start_date, end_date, window):
    df = get_historical_prices(symbol, start_date, end_date)
    df['SMA'] = df['Close'].rolling(window=20).mean()

    # Calculate the 20-period Standard Deviation (SD)
    df['SD'] = df['Close'].rolling(window=20).std()

    # Calculate the Upper Bollinger Band (UB) and Lower Bollinger Band (LB)
    df['UB'] = df['SMA'] + 2 * df['SD']
    df['LB'] = df['SMA'] - 2 * df['SD']
    
    return df


# Chande Momentum Oscillator to predict future momentum
def get_cmo(symbol, start_date, end_date, window=14):
    df = get_historical_prices(symbol, start_date, end_date)
    # Calculate the daily price change
    df['price_change'] = df['Close'].diff()
    
    # Separate the gains and losses
    df['gain'] = df['price_change'].where(df['price_change'] > 0, 0)
    df['loss'] = -df['price_change'].where(df['price_change'] < 0, 0)
    
    # Calculate the sum of gains and losses over the window period
    df['sum_gain'] = df['gain'].rolling(window=window).sum()
    df['sum_loss'] = df['loss'].rolling(window=window).sum()
    
    # Calculate the CMO
    df['CMO'] = 100 * (df['sum_gain'] - df['sum_loss']) / (df['sum_gain'] + df['sum_loss'])
    
    return df

# Fetch stock data
stock_symbol = "AAPL"
start_date = "2025-01-01"
end_date = "2025-03-01"

# Corrected the call to get_historical_prices by passing the required parameters
df = get_cmo(stock_symbol, start_date, end_date, window=14)

print(df['CMO'])

