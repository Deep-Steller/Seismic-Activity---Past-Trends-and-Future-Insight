import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import os

# Setup folders
os.makedirs('forecasting', exist_ok=True)
os.makedirs('tableau_exports', exist_ok=True)
os.makedirs('visualizations/charts_post_forecasting', exist_ok=True)

# Load cleaned data
df = pd.read_csv('data/cleaned_earthquakes.csv')
df['time'] = pd.to_datetime(df['time'], errors='coerce')
df = df.dropna(subset=['time'])

# Group by year
df['year'] = df['time'].dt.year
yearly_counts = df.groupby('year').size().reset_index(name='count')

# Prepare for Prophet
prophet_df = yearly_counts.rename(columns={'year': 'ds', 'count': 'y'})
prophet_df['ds'] = pd.to_datetime(prophet_df['ds'], format='%Y')

# Fit Prophet model
model = Prophet()
model.fit(prophet_df)

# Forecast next 2 years
future = model.make_future_dataframe(periods=2, freq='YE')
forecast = model.predict(future)

# Export forecast
export_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
export_df.rename(columns={
    'ds': 'year',
    'yhat': 'predicted_count',
    'yhat_lower': 'lower_bound',
    'yhat_upper': 'upper_bound'
}, inplace=True)
export_df['year'] = export_df['year'].dt.year
export_df = export_df.tail(2)
export_df.to_csv('tableau_exports/predicted_data_prophet.csv', index=False)

# Plot forecast
fig = model.plot(forecast)
plt.title("Earthquake Frequency Forecast (Prophet) – Next 2 Years")
plt.xlabel("Year")
plt.ylabel("Predicted Earthquake Count")
plt.tight_layout()
plt.savefig("visualizations/charts_post_forecasting/prophet_forecast.png")
plt.close()

print("Prophet 2-year forecasting complete! Saved to tableau_exports and visualizations.")
