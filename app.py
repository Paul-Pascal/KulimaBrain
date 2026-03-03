import streamlit as st
import pandas as pd
import pickle
import os
import numpy as np
from datetime import datetime, timedelta
import requests_cache
from retry_requests import retry
import openmeteo_requests

# ----------------------------
# 1. Load Models
# ----------------------------
models = {}
feature_cols = {}

for crop in ["maize", "beans"]:
    path = f"models/saved/{crop}_doy_model.pkl"
    if os.path.exists(path):
        with open(path, "rb") as f:
            data = pickle.load(f)
            models[crop] = data['model']
            feature_cols[crop] = data['features']
        st.write(f"✅ Loaded {crop} DOY model")
    else:
        st.warning(f"⚠️ Model missing: {path}")

# ----------------------------
# 2. District Coordinates
# ----------------------------
DISTRICT_COORDS = {
    "Abim": (2.7833, 33.8333),
    "Adjumani": (3.3833, 31.7833),
    # ... (your full list here - keep it)
    "Wakiso": (0.3833, 32.4667),
    "Yumbe": (3.4833, 31.2833),
    "Zombo": (3.2833, 30.7833)
}

DISTRICTS = sorted(DISTRICT_COORDS.keys())

st.title("🌱 AgriConsult Uganda: KulimaBrain")
st.markdown("Predicts planting day using climate trends + live weather.")

col1, col2, col3 = st.columns(3)
with col1:
    district = st.selectbox("District", DISTRICTS)
with col2:
    target_date = st.date_input("Target Year", value=datetime.today())
with col3:
    crop = st.selectbox("Crop", list(models.keys()))

if st.button("Get Advice"):
    if crop not in models:
        st.error("Model not loaded.")
        st.stop()

    target_year = target_date.year
    current_date = pd.to_datetime(datetime.today())

    # Get coordinates
    if district not in DISTRICT_COORDS:
        st.error(f"Coordinates not available for {district}")
        st.stop()
    lat, lon = DISTRICT_COORDS[district]

    # ----------------------------
    # 3. Get Real Weather Context from Open-Meteo
    # ----------------------------
    try:
        cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        openmeteo = openmeteo_requests.Client(session=retry_session)

        # Get recent weather (last 30 days) to estimate current conditions
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": ["precipitation_sum", "temperature_2m_min"],
            "past_days": 30,
            "forecast_days": 0,
            "timezone": "Africa/Kampala"
        }
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]
        daily = response.Daily()
        precip = daily.Variables(0).ValuesAsNumpy()
        temp_min = daily.Variables(1).ValuesAsNumpy()

        # Estimate features
        rain_3d = np.sum(precip[-3:]) if len(precip) >= 3 else 25.0
        min_temp_3d = np.min(temp_min[-3:]) if len(temp_min) >= 3 else 16.0

        # Simulate dry days in next 7 days (use climatology if needed)
        dry_days_next_7 = 2

    except Exception as e:
        st.warning(f"Using default weather: {e}")
        rain_3d = 25.0
        min_temp_3d = 16.0
        dry_days_next_7 = 2

    # ----------------------------
    # 4. Predict Planting DOY Using Model
    # ----------------------------
    month_guess = 3  # default to March
    features = [[target_year, month_guess, rain_3d, min_temp_3d, dry_days_next_7]]
    predicted_doy = int(models[crop].predict(features)[0])
    predicted_doy = max(30, min(330, predicted_doy))  # clamp to reasonable range

    predicted_date = pd.Timestamp(f"{target_year}-01-01") + pd.Timedelta(days=predicted_doy - 1)

    st.info(f"📅 Model predicts planting around {predicted_date.strftime('%B %d')}")

    # ----------------------------
    # 5. Refine with Open-Meteo (if near-term)
    # ----------------------------
    days_diff = (predicted_date - current_date).days
    if -7 <= days_diff <= 14:
        st.info("📡 Refining with Open-Meteo forecast...")

        try:
            # Get 14-day forecast around predicted date
            start_forecast = predicted_date - pd.Timedelta(days=7)
            end_forecast = predicted_date + pd.Timedelta(days=7)

            url = "https://api.open-meteo.com/v1/forecast"
            params = {
                "latitude": lat,
                "longitude": lon,
                "daily": ["precipitation_sum"],
                "start_date": start_forecast.strftime("%Y-%m-%d"),
                "end_date": end_forecast.strftime("%Y-%m-%d"),
                "timezone": "Africa/Kampala"
            }

            responses = openmeteo.weather_api(url, params=params)
            response = responses[0]
            daily = response.Daily()
            precip = daily.Variables(0).ValuesAsNumpy()

            # Find first suitable day
            final_date = predicted_date
            for i in range(2, len(precip)):
                if np.sum(precip[i-2:i+1]) >= 25:
                    final_date = start_forecast + pd.Timedelta(days=i)
                    break

            st.success(f"✅ Plant {crop} on {final_date.strftime('%Y-%m-%d')}")
            st.info(f"📍 {district} | 🌾 {crop}")
        except Exception as e:
            st.warning(f"Open-Meteo refinement failed: {e}. Using model prediction.")
            st.success(f"✅ Plant {crop} around {predicted_date.strftime('%Y-%m-%d')}")
    else:
        st.success(f"✅ Plant {crop} around {predicted_date.strftime('%Y-%m-%d')}")
        st.info(f"📍 {district} | 🌾 {crop} | Based on climate trends")
