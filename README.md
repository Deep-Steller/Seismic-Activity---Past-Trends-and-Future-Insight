# 🌍 Seismic Activity: Past Trends and Future Insights

## 📌 Project Overview
This project analyzes historical global earthquake data (1900–2025) and forecasts future earthquake frequency (2025–2027) using machine learning models. The goal is to understand past seismic trends and evaluate forecasting performance using statistical and deep learning approaches.

## 🎯 Objective
To compare Prophet, LSTM, and Hybrid (Prophet + LSTM) models for forecasting global earthquake frequency and assess their predictive performance.

---

## 📊 Dataset
Source: Kaggle – Significant Earthquakes Dataset  
Time Range: 1900 – February 2025  

Key Features:
- Date
- Magnitude
- Depth
- Latitude & Longitude
- Location

---

## 🛠 Project Workflow

1. Data Cleaning & Preprocessing  
2. Exploratory Data Analysis (EDA)  
3. MySQL Database Storage  
4. Model Training (Prophet, LSTM, Hybrid)  
5. Model Evaluation (MAE, RMSE, R²)  
6. Tableau Dashboard Visualization  

---

## 🤖 Models Used

### 1. Prophet
Captures trend and seasonality patterns in time-series data.

### 2. LSTM (Long Short-Term Memory)
Deep learning model that learns time-dependent patterns.

### 3. Hybrid Model
Combines Prophet trend prediction with LSTM sequence learning.

---

## 📈 Model Evaluation Results

| Model | RMSE | MAE | R² |
|--------|--------|--------|--------|
| Prophet | 2216.66 | 1901.67 | -1.41 |
| LSTM | 2411.94 | 1991.61 | -1.86 |
| Hybrid | 2148.13 | 1871.22 | -1.22 |

Note: Negative R² reflects the high unpredictability of earthquake frequency.

---

## 📊 Tableau Dashboards

### Phase 1 – Pre-Forecasting
- Yearly Earthquake Trend
- Monthly Distribution
- Magnitude vs Depth
- Heatmap (Year vs Month)
- Top 10 Locations

### Phase 2 – Forecasting
- Prophet Forecast
- LSTM Forecast
- Hybrid Forecast
- Actual vs Predicted Comparison
- Anomaly Detection

---

## 🔮 Future Work
- Integrate live seismic feeds
- Add external geophysical features
- Explore Transformer/GRU models
- Build real-time alert dashboards

---

## 👥 Authors
Pradeepa Chakkaravarthy  
Logeswaran Selvapandian  
Puli Joshith Reddy
