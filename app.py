import streamlit as st
import pandas as pd
import pickle
import glob
import os
import numpy as np

# Load models (from your GitHub-style path)
models = {}
for crop in ["maize", "beans"]:
    path = f"models/saved/{crop}_model.pkl"
    if os.path.exists(path):
        with open(path, "rb") as f:
            models[crop] = pickle.load(f)
        st.write(f"✅ Loaded {crop} model")
    else:
        st.warning(f"⚠️ Model missing: {path}")

# Load district list from your CSV (real data)
data_path = "/content/drive/MyDrive/KulimaBrain/"


files = glob.glob(os.path.join(data_path, "Uganda_Climate_*.csv"))

if len(files) == 0:
    raise ValueError(f"No CSV files found in {data_path}")


df_list = [pd.read_csv(f) for f in files]
climate_df = pd.concat(df_list, ignore_index=True)
# # Fix column names — your CSV has NO header, so we assign them explicitly
# climate_df.columns = ["ADM2_NAME", "date", "precipitation_mm", "temperature_c", "dewpoint_c"]
climate_df['date'] = pd.to_datetime(climate_df['date'])

DISTRICTS = sorted(climate_df['ADM2_NAME'].unique().tolist())

# Advice templates
ADVICE_TEMPLATES = {
    "plant_now": "✅ Good time to plant {crop}! Recent rain ({rain:.1f}mm) and warm soil.",
    "plant_but_prepare_for_dry_spell": "🌱 Plant {crop} now, but prepare for dry spell in next week.",
    "delay_planting_drought": "⚠️ Delay planting {crop} — drought conditions expected.",
    "monitor_conditions": "🔍 Keep monitoring {crop} — no clear signal yet."
}

st.title("🌱 AgriConsult Uganda: Farming Advisor")
st.markdown("Get AI-powered planting advice based on real weather data.")

col1, col2, col3 = st.columns(3)
with col1:
    district = st.selectbox("District", DISTRICTS)
with col2:
    date = st.date_input("Date", value=pd.to_datetime("2025-03-10"))
with col3:
    crop = st.selectbox("Crop", list(models.keys()))

if st.button("Get Advice"):
    if crop not in models:
        st.error("Model not loaded.")
    else:
        # Get past 30 days of observed weather for this district
        current_date = pd.to_datetime(date)
        district_data = climate_df[climate_df['ADM2_NAME'] == district].copy()
        start_date = current_date - pd.Timedelta(days=30)
        past_30d = district_data[
            (district_data['date'] >= start_date) & 
            (district_data['date'] <= current_date)
        ].sort_values('date')
        
        if len(past_30d) < 3:
            st.warning("⚠️ Not enough historical data for this district.")
        else:
            # Compute features (same as training)
            last_3d = past_30d.tail(3)
            rain_3d = last_3d['precipitation_mm'].sum()
            min_temp = last_3d['temperature_c'].min()
            avg_dew = past_30d['dewpoint_c'].mean()
            
            # Simulate forecast dry days (replace with Open-Meteo later)
            dry_days = 2  # placeholder — safe default
            
            # Predict using your trained model
            features = [[rain_3d, min_temp, avg_dew, dry_days]]
            pred_label = models[crop].predict(features)[0]
            
            # Extract base action (e.g., "plant_now_maize" → "plant_now")
            base_action = "_".join(pred_label.split("_")[:-1])
            advice = ADVICE_TEMPLATES.get(base_action, "❓ Unknown advice.").format(
                crop=crop, rain=rain_3d
            )
            
            st.success(advice)
            st.info(f"📍 {district} | 📅 {date} | 🌾 {crop} | Rain: {rain_3d:.1f}mm")
