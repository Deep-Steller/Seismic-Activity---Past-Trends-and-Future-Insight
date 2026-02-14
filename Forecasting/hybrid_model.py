import pandas as pd
import os

# Setup folders
os.makedirs('forecasting', exist_ok=True)
os.makedirs('tableau_exports', exist_ok=True)
os.makedirs('visualizations/charts_post_forecasting', exist_ok=True)

# Load both forecast outputs
prophet_df = pd.read_csv('tableau_exports/predicted_data_prophet.csv')
lstm_df = pd.read_csv('tableau_exports/predicted_data_lstm.csv')

# Debug: See actual column names
print("Prophet Columns:", prophet_df.columns.tolist())
print("LSTM Columns:", lstm_df.columns.tolist())

# Merge based on 'year'
merged = pd.merge(prophet_df, lstm_df, on='year', suffixes=('_prophet', '_lstm'))

# Handle missing bound columns gracefully
lower_cols = [col for col in merged.columns if 'lower_bound' in col]
upper_cols = [col for col in merged.columns if 'upper_bound' in col]
print("Detected lower bound columns:", lower_cols)
print("Detected upper bound columns:", upper_cols)

# Compute average prediction
merged['avg_predicted_count'] = merged[['predicted_count_prophet', 'predicted_count_lstm']].mean(axis=1)
merged['avg_lower_bound'] = merged[lower_cols].mean(axis=1) if lower_cols else merged['avg_predicted_count']
merged['avg_upper_bound'] = merged[upper_cols].mean(axis=1) if upper_cols else merged['avg_predicted_count']

# Save combined results
hybrid_df = merged[['year', 'avg_predicted_count', 'avg_lower_bound', 'avg_upper_bound']].copy()
hybrid_df.columns = ['year', 'predicted_count', 'lower_bound', 'upper_bound']
hybrid_df.to_csv('tableau_exports/combined_forecast.csv', index=False)

print("Hybrid forecasting complete! Saved to tableau_exports/combined_forecast.csv")
