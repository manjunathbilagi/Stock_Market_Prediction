import yfinance as yf
import ta

def get_stock_data(stock, months):
    return yf.download(stock, period=f"{months}mo")

def add_indicators(data):
    data['MA20'] = data['Close'].rolling(20).mean()
    data['MA50'] = data['Close'].rolling(50).mean()
    data['RSI'] = ta.momentum.RSIIndicator(close=data['Close'].squeeze()).rsi()
    data['Signal'] = 0
    data.loc[data['MA20'] > data['MA50'], 'Signal'] = 1
    data.loc[data['MA20'] < data['MA50'], 'Signal'] = -1
    return data
