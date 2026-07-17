import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
from pathlib import Path

# Page config
st.set_page_config(
    page_title="XAI Grid Carbon Intensity",
    page_icon="⚡",
    layout="wide"
)

# Load model
MODELS = Path("../src/models")

@st.cache_resource
def load_model():
    return joblib.load(MODELS / "xgboost_baseline.pkl")

@st.cache_resource
def load_explainer(_model):
    return shap.TreeExplainer(_model)

model = load_model()
explainer = load_explainer(model)

FEATURES = [
    "forecast", "u10", "t2m", "ssrd",
    "hour", "day_of_week", "month",
    "is_weekend", "season",
    "actual_lag1", "actual_lag2", "actual_lag24",
    "rolling_mean_24h"
]

# Header
st.title("⚡ Explainable AI — UK Grid Carbon Intensity")
st.markdown("**MSc Data Science Dissertation | London South Bank University**")
st.markdown("*Predicting and explaining UK electricity grid carbon intensity using XGBoost + SHAP*")
st.divider()

# Sidebar inputs
st.sidebar.header("Grid Conditions Input")
st.sidebar.markdown("Adjust values to see predictions and explanations")

forecast = st.sidebar.slider("National Grid Forecast (gCO2/kWh)", 0, 400, 175)
actual_lag1 = st.sidebar.slider("Carbon Intensity 1hr ago (gCO2/kWh)", 0, 400, 170)
actual_lag2 = st.sidebar.slider("Carbon Intensity 2hrs ago (gCO2/kWh)", 0, 400, 168)
actual_lag24 = st.sidebar.slider("Carbon Intensity 24hrs ago (gCO2/kWh)", 0, 400, 172)
rolling_mean_24h = st.sidebar.slider("24hr Rolling Mean (gCO2/kWh)", 0, 400, 170)

st.sidebar.subheader("Weather Conditions")
u10 = st.sidebar.slider("Wind Speed at 10m (m/s)", -15.0, 15.0, 2.0)
t2m = st.sidebar.slider("Temperature at 2m (°C)", -5.0, 35.0, 12.0)
ssrd = st.sidebar.slider("Solar Radiation (J/m²)", 0.0, 3000000.0, 500000.0)

st.sidebar.subheader("Time Features")
hour = st.sidebar.slider("Hour of Day", 0, 23, 12)
month = st.sidebar.slider("Month", 1, 12, 6)
day_of_week = st.sidebar.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 2)
is_weekend = 1 if day_of_week >= 5 else 0
season = {12:0,1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:3,10:3,11:3}[month]

# Build input
input_data = pd.DataFrame([[
    forecast, u10, t2m, ssrd, hour, day_of_week, month,
    is_weekend, season, actual_lag1, actual_lag2, actual_lag24, rolling_mean_24h
]], columns=FEATURES)

# Prediction
prediction = model.predict(input_data)[0]

# Carbon level
if prediction < 100:
    level = "🟢 Very Low"
    color = "green"
elif prediction < 150:
    level = "🟡 Low"
    color = "blue"
elif prediction < 200:
    level = "🟠 Moderate"
    color = "orange"
elif prediction < 250:
    level = "🔴 High"
    color = "red"
else:
    level = "🔴 Very High"
    color = "darkred"

# Display prediction
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Predicted Carbon Intensity", f"{prediction:.1f} gCO2/kWh")
with col2:
    st.metric("Carbon Level", level)
with col3:
    st.metric("National Grid Forecast", f"{forecast} gCO2/kWh")

st.divider()

# SHAP explanation
st.subheader("🔍 Why did the model predict this?")
st.markdown("SHAP values show how much each feature contributed to this prediction.")

shap_values = explainer.shap_values(input_data)
expected_value = explainer.expected_value

col_left, col_right = st.columns(2)

with col_left:
    fig, ax = plt.subplots(figsize=(8, 5))
    shap_df = pd.DataFrame({
        "Feature": FEATURES,
        "SHAP Value": shap_values[0],
        "Feature Value": input_data.iloc[0].values
    }).sort_values("SHAP Value")

    colors = ["red" if v > 0 else "steelblue" for v in shap_df["SHAP Value"]]
    ax.barh(shap_df["Feature"], shap_df["SHAP Value"], color=colors)
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP Value (impact on prediction)")
    ax.set_title(f"Feature Contributions\nBase: {expected_value:.1f} → Prediction: {prediction:.1f} gCO2/kWh")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col_right:
    st.subheader("📊 Feature Contribution Summary")
    shap_summary = shap_df.sort_values("SHAP Value", ascending=False)
    for _, row in shap_summary.iterrows():
        direction = "⬆️ pushes UP" if row["SHAP Value"] > 0 else "⬇️ pushes DOWN"
        st.write(f"**{row['Feature']}** = {row['Feature Value']:.2f} → {direction} by {abs(row['SHAP Value']):.2f}")

st.divider()

# Model info
st.subheader("📈 Model Performance")
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
col_m1.metric("R²", "0.9774")
col_m2.metric("RMSE", "10.02 gCO2/kWh")
col_m3.metric("MAE", "7.39 gCO2/kWh")
col_m4.metric("Training Carbon Cost", "0.0008 gCO2")

st.caption("Built by Ramya Shree Babu | MSc Data Science | LSBU | 2026")
