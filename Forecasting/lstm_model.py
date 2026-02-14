import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Create output folders if they don't exist
os.makedirs('tableau_exports', exist_ok=True)
os.makedirs('visualizations/charts_post_forecasting', exist_ok=True)

# Load and prepare data
df = pd.read_csv('data/cleaned_earthquakes.csv')
df['time'] = pd.to_datetime(df['time'], errors='coerce')
df = df.dropna(subset=['time'])
df['year'] = df['time'].dt.year
yearly_counts = df.groupby('year').size().reset_index(name='count')

# Normalize data
scaler = MinMaxScaler()
scaled_counts = scaler.fit_transform(yearly_counts['count'].values.reshape(-1, 1))

# Create time series sequences
def create_sequences(data, window_size=5):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

window_size = 5
X, y = create_sequences(scaled_counts, window_size)
X = X.reshape((X.shape[0], window_size, 1))  # LSTM expects 3D input

# Build and train model
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(window_size, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=100, verbose=0)

# Forecast next 2 years
forecast_steps = 2
last_sequence = scaled_counts[-window_size:]

predictions = []
for _ in range(forecast_steps):
    pred_input = last_sequence.reshape((1, window_size, 1))
    pred = model.predict(pred_input, verbose=0)[0][0]
    predictions.append(pred)
    last_sequence = np.append(last_sequence[1:], [[pred]], axis=0)

# Inverse transform predictions
predicted_counts = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

# Prepare result DataFrame
last_year = yearly_counts['year'].iloc[-1]
future_years = [last_year + i + 1 for i in range(forecast_steps)]

lstm_df = pd.DataFrame({
    'year': future_years,
    'predicted_count': predicted_counts.astype(int)
})
lstm_df.to_csv('tableau_exports/predicted_data_lstm.csv', index=False)

# Plot forecast
plt.figure(figsize=(10, 5))
plt.plot(yearly_counts['year'], yearly_counts['count'], label='Historical', linewidth=2)
plt.plot(future_years, predicted_counts, label='LSTM Forecast', linestyle='dashed', linewidth=2)
plt.xlabel('Year')
plt.ylabel('Earthquake Count')
plt.title('LSTM Earthquake Frequency Forecast (Next 2 Years)')
plt.legend()
plt.tight_layout()
plt.savefig('visualizations/charts_post_forecasting/lstm_forecast.png')
plt.close()

print("LSTM 2-year forecasting complete! Saved to tableau_exports and visualizations.")
