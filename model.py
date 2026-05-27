import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def train_lstm_model(data):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data[['Close']])
    X, y = [], []
    for i in range(60, len(scaled)):
        X.append(scaled[i-60:i])
        y.append(scaled[i])
    X, y = np.array(X), np.array(y)
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=2, batch_size=32, verbose=0)
    return model, scaler, X, y

def predict_prices(model, scaler, X, y):
    pred = model.predict(X)
    pred = scaler.inverse_transform(pred)
    actual = scaler.inverse_transform(y)
    return pred, actual
