import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import os

# Setup folder
os.makedirs('evaluation', exist_ok=True)

# Load full data
df = pd.read_csv('data/cleaned_earthquakes.csv')
df['time'] = pd.to_datetime(df['time'], errors='coerce')
df = df.dropna(subset=['time'])
df['year'] = df['time'].dt.year
yearly = df.groupby('year').size().reset_index(name='count')

# Split train/test
train = yearly[yearly['year'] <= 2021]
test = yearly[(yearly['year'] > 2021) & (yearly['year'] <= 2024)]

# ---------- Prophet ----------
prophet_df = train.rename(columns={'year': 'ds', 'count': 'y'})
prophet_df['ds'] = pd.to_datetime(prophet_df['ds'], format='%Y')

model = Prophet()
model.fit(prophet_df)

future = model.make_future_dataframe(periods=3, freq='Y')  # 2022-2024
forecast = model.predict(future)

prophet_forecast = forecast[['ds', 'yhat']].tail(3)
prophet_forecast['year'] = prophet_forecast['ds'].dt.year
prophet_forecast = prophet_forecast[['year', 'yhat']].set_index('year')

# ---------- LSTM ----------
scaler = MinMaxScaler()
scaled = scaler.fit_transform(train['count'].values.reshape(-1, 1))

def create_sequences(data, window=5):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i-window:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

window = 5
X, y_lstm = create_sequences(scaled, window)
X = X.reshape((X.shape[0], X.shape[1], 1))

model_lstm = Sequential()
model_lstm.add(LSTM(50, activation='relu', input_shape=(window, 1)))
model_lstm.add(Dense(1))
model_lstm.compile(optimizer='adam', loss='mse')
model_lstm.fit(X, y_lstm, epochs=100, verbose=0)

# Forecast next 3 years
last_seq = scaled[-window:]
preds = []
for _ in range(3):
    inp = last_seq.reshape((1, window, 1))
    pred = model_lstm.predict(inp, verbose=0)
    preds.append(pred[0][0])
    last_seq = np.append(last_seq[1:], [[pred[0][0]]], axis=0)

lstm_forecast = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
lstm_forecast = pd.DataFrame({'year': [2022, 2023, 2024], 'yhat': lstm_forecast}).set_index('year')

# ---------- Actual ----------
actual = test.set_index('year')['count']

# ---------- Evaluation ----------
def evaluate(preds, actuals, model_name):
    mae = mean_absolute_error(actuals, preds)
    rmse = np.sqrt(mean_squared_error(actuals, preds))
    r2 = r2_score(actuals, preds)
    print(f"📊 {model_name} Evaluation:")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.4f}")
    print("-" * 30)

evaluate(prophet_forecast['yhat'], actual, "Prophet")
evaluate(lstm_forecast['yhat'], actual, "LSTM")
