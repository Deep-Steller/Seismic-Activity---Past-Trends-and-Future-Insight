import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -----------------------------------------
# Load Data
# -----------------------------------------
df = pd.read_csv('data/cleaned_earthquakes.csv')
df['time'] = pd.to_datetime(df['time'], errors='coerce')
df = df.dropna(subset=['time'])

# Extract time-based fields
df['year'] = df['time'].dt.year.astype(int)
df['month'] = df['time'].dt.month_name()

# -----------------------------------------
# 1. Earthquake Frequency by Year
# -----------------------------------------
yearly_counts = df['year'].value_counts().sort_index()

plt.figure(figsize=(12, 5))
sns.lineplot(x=yearly_counts.index, y=yearly_counts.values)
plt.title('Earthquake Frequency by Year')
plt.xlabel('Year')
plt.ylabel('Number of Earthquakes')
plt.tight_layout()
plt.savefig('visualizations/charts_pre_forecasting/yearly_trend.png')
plt.close()

# -----------------------------------------
# 2. Earthquake Frequency by Month
# -----------------------------------------
monthly_counts = df['month'].value_counts().reindex([
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
])

plt.figure(figsize=(10, 5))
sns.barplot(x=monthly_counts.index, y=monthly_counts.values, palette='viridis')
plt.title('Earthquake Frequency by Month')
plt.xlabel('Month')
plt.ylabel('Number of Earthquakes')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('visualizations/charts_pre_forecasting/monthly_trend.png')
plt.close()

# -----------------------------------------
# 3. Top 10 Earthquake Locations
# -----------------------------------------
top_places = df['place'].value_counts().head(10)

plt.figure(figsize=(12, 6))
sns.barplot(x=top_places.values, y=top_places.index, palette='magma')
plt.title('Top 10 Earthquake Locations')
plt.xlabel('Number of Earthquakes')
plt.ylabel('Place')
plt.tight_layout()
plt.savefig('visualizations/charts_pre_forecasting/top_places.png')
plt.close()

# -----------------------------------------
# 4. Magnitude vs. Depth (Scatter Plot)
# -----------------------------------------
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='depth', y='mag', alpha=0.3)
plt.title('Magnitude vs. Depth')
plt.xlabel('Depth (km)')
plt.ylabel('Magnitude')
plt.tight_layout()
plt.savefig('visualizations/charts_pre_forecasting/mag_vs_depth.png')
plt.close()

# -----------------------------------------
# 5. Heatmap – Earthquake Activity by Year and Month
# -----------------------------------------
heatmap_data = df.groupby(['year', 'month']).size().unstack(fill_value=0).reindex(columns=[
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
])

plt.figure(figsize=(14, 6))
sns.heatmap(heatmap_data, cmap='YlGnBu', linewidths=0.2, cbar_kws={'label': 'Number of Earthquakes'})
plt.title('Earthquake Frequency Heatmap (Year vs. Month)')
plt.xlabel('Month')
plt.ylabel('Year')
plt.yticks(rotation=0)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('visualizations/charts_pre_forecasting/year_month_heatmap.png')
plt.close()

# -----------------------------------------
# 6. Export for Tableau
# -----------------------------------------
export_cols = ['time', 'latitude', 'longitude', 'depth', 'mag', 'place', 'type', 'year', 'month']
df[export_cols].to_csv('tableau_exports/pre_forecasting_data.csv', index=False)

print("EDA completed! All charts saved and Tableau data exported.")
