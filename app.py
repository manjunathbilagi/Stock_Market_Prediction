import streamlit as st
import matplotlib.pyplot as plt
from utils import get_stock_data, add_indicators
from model import train_lstm_model, predict_prices
from sklearn.metrics import mean_squared_error
import numpy as np

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Stock Dashboard", layout="wide")

st.title("📈 Stock Market Prediction Dashboard")

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("⚙️ User Input")

stock_input = st.sidebar.text_input("Enter Stock Symbol", "AAPL")
months_input = st.sidebar.slider("Select Months", 1, 24, 6)

submit = st.sidebar.button("🚀 Submit")

# -----------------------------
# Instruction Message
# -----------------------------
if not submit:
    st.info("👈 Enter stock details and click Submit to view graphs")

# -----------------------------
# Main Logic (runs only on submit)
# -----------------------------
if submit:

    # Fetch Data
    data = get_stock_data(stock_input, months_input)

    if data.empty:
        st.error("❌ Invalid stock symbol!")
        st.stop()

    # Add Indicators
    data = add_indicators(data)

    # Show Current Price
    current_price = data['Close'].iloc[-1]
    if isinstance(current_price, (list, tuple)):
        current_price = current_price[0]
    if hasattr(current_price, "values"):
        current_price = current_price.values[0]
        current_price = float(current_price)
        st.metric("📌 Current Price", f"{current_price:.2f}")
    # Train Model
    with st.spinner("Training LSTM model... ⏳"):
        model, scaler, X, y = train_lstm_model(data)
        predicted, actual = predict_prices(model, scaler, X, y)

    # -----------------------------
    # Accuracy
    # -----------------------------
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    st.subheader("📊 Model Accuracy")
    st.write(f"RMSE: {rmse:.2f}")

    # -----------------------------
    # GRAPH 1: LSTM Prediction
    # -----------------------------
    st.subheader("📊 LSTM Prediction vs Actual")

    fig1, ax1 = plt.subplots()
    ax1.plot(actual, label="Actual")
    ax1.plot(predicted, label="Predicted")
    ax1.legend()
    st.pyplot(fig1)

    # -----------------------------
    # GRAPH 2: RSI
    # -----------------------------
    st.subheader("📉 RSI Indicator")

    fig2, ax2 = plt.subplots()
    ax2.plot(data['RSI'], label="RSI")
    ax2.axhline(70)
    ax2.axhline(30)
    ax2.legend()
    st.pyplot(fig2)

    # -----------------------------
    # GRAPH 3: Moving Averages
    # -----------------------------
    st.subheader("📈 Price + MA20 & MA50")

    fig3, ax3 = plt.subplots()
    ax3.plot(data['Close'], label="Price")
    ax3.plot(data['MA20'], label="MA20")
    ax3.plot(data['MA50'], label="MA50")
    ax3.legend()
    st.pyplot(fig3)

    # -----------------------------
    # GRAPH 4: Buy/Sell Signals
    # -----------------------------
    st.subheader("🟢 Buy / 🔴 Sell Signals")

    fig4, ax4 = plt.subplots()
    ax4.plot(data['Close'], label="Price")

    buy = data[data['Signal'] == 1]
    sell = data[data['Signal'] == -1]

    ax4.scatter(buy.index, buy['Close'], marker='^', label="Buy")
    ax4.scatter(sell.index, sell['Close'], marker='v', label="Sell")

    ax4.legend()
    st.pyplot(fig4)

    # -----------------------------
    # GRAPH 5: Monthly Trend
    # -----------------------------
    st.subheader("📅 Monthly Trend")

    monthly = data['Close'].resample('ME').mean()

    fig5, ax5 = plt.subplots()
    ax5.plot(monthly, label="Monthly Avg")
    ax5.legend()
    st.pyplot(fig5)

    # -----------------------------
    # AI Recommendation
    # -----------------------------
    st.subheader("🤖 AI Recommendation")

    latest_signal = data['Signal'].iloc[-1]

    if latest_signal == 1:
        st.success("BUY Signal 🟢")
    elif latest_signal == -1:
        st.error("SELL Signal 🔴")
    else:
        st.warning("HOLD ⚖️")