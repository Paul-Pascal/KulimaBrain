import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
from datetime import datetime, timedelta
import requests
from io import BytesIO

def load_model_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return joblib.load(BytesIO(response.content))
    except Exception as e:
        st.error(f"Failed to load model from {url}: {str(e)}")
        return None


MAIZE_MODEL_URL = "https://github.com/Paul-Pascal/KulimaBrain/releases/download/v1.0/maize_doy_model.joblib"
BEANS_MODEL_URL = "https://github.com/Paul-Pascal/KulimaBrain/releases/download/v1.0/beans_doy_model.joblib"

models = {}
feature_cols = {}

for crop in ["maize", "beans"]:
    file_url = MAIZE_MODEL_URL if crop == "maize" else BEANS_MODEL_URL
    data = load_model_from_drive(file_url)
    if data is not None:
        models[crop] = data['model']
        feature_cols[crop] = data['features']
    else:
        st.warning(f"⚠️ Failed to load {crop} model from Google Drive")

DISTRICT_COORDS = {
    "Abim": (2.7833, 33.8333),
    "Adjumani": (3.3833, 31.7833),
    "Agago": (3.0833, 33.3333),
    "Alebtong": (2.3833, 32.8333),
    "Amolatar": (2.0833, 32.5833),
    "Amudat": (1.9833, 34.8333),
    "Amuria": (1.6833, 33.6333),
    "Amuru": (2.9833, 31.8333),
    "Apac": (2.0833, 32.0833),
    "Arua": (3.0333, 30.9167),
    "Budaka": (1.0833, 33.9833),
    "Bududa": (1.0833, 34.3833),
    "Bugiri": (0.5833, 33.7833),
    "Buhweju": (-0.4833, 30.1833),
    "Buikwe": (0.2833, 32.9833),
    "Bukedea": (1.3833, 33.8333),
    "Bukomansimbi": (-0.3833, 31.3833),
    "Bukwo": (1.3833, 34.7833),
    "Bulambuli": (1.2833, 34.5833),
    "Buliisa": (1.7833, 31.3833),
    "Bundibugyo": (0.7833, 30.0833),
    "Bushenyi": (-0.5833, 30.1833),
    "Busia": (0.5833, 34.1833),
    "Butaleja": (1.0833, 34.2833),
    "Butambala": (0.2833, 32.0833),
    "Butebo": (1.3833, 33.9833),
    "Buvuma": (0.0833, 32.9833),
    "Buyende": (1.2833, 33.1833),
    "Dokolo": (2.2833, 32.5833),
    "Gomba": (0.0833, 31.8833),
    "Gulu": (2.7747, 32.2990),
    "Hoima": (1.4333, 31.3500),
    "Ibanda": (-0.1833, 30.5833),
    "Iganga": (0.6833, 33.4833),
    "Isingiro": (-0.6833, 30.7833),
    "Jinja": (0.4333, 33.2000),
    "Kaabong": (3.5833, 33.8333),
    "Kabale": (-1.2500, 29.9833),
    "Kabarole": (0.7833, 30.2833),
    "Kaberamaido": (1.6833, 33.1833),
    "Kagadi": (0.8833, 30.8833),
    "Kakumiro": (1.1833, 31.1833),
    "Kalangala": (-0.4833, 32.1833),
    "Kaliro": (1.0833, 33.5833),
    "Kalungu": (-0.1833, 31.3833),
    "Kampala": (0.3136, 32.5811),
    "Kamuli": (1.0833, 33.1833),
    "Kamwenge": (0.3833, 30.3833),
    "Kanungu": (-1.0833, 29.7833),
    "Kapchorwa": (1.3833, 34.4500),
    "Kasese": (-0.1833, 30.0833),
    "Kassanda": (0.3833, 31.5833),
    "Katakwi": (2.0833, 33.8333),
    "Kayunga": (0.7833, 32.9833),
    "Kibaale": (1.0833, 31.1833),
    "Kiboga": (0.7833, 31.8833),
    "Kibuku": (1.3833, 33.8833),
    "Kiruhura": (-0.5833, 30.5833),
    "Kiryandongo": (1.7833, 31.8833),
    "Kisoro": (-1.2833, 29.6833),
    "Kitgum": (3.2833, 32.8333),
    "Koboko": (3.4833, 30.7833),
    "Kole": (2.3833, 32.5833),
    "Kotido": (2.7833, 33.8333),
    "Kumi": (1.4833, 33.9833),
    "Kwania": (2.1833, 32.5833),
    "Kween": (1.3833, 34.6833),
    "Kyankwanzi": (0.9833, 31.5833),
    "Kyegegwa": (0.5833, 30.5833),
    "Kyenjojo": (0.6833, 30.6833),
    "Kyotera": (-0.6833, 31.3833),
    "Lamwo": (3.5833, 32.5833),
    "Lira": (2.2500, 32.5833),
    "Luuka": (1.0833, 33.3833),
    "Luweero": (0.9333, 32.5000),
    "Lwengo": (-0.3833, 31.0833),
    "Lyantonde": (-0.2833, 31.0833),
    "Manafwa": (1.0833, 34.3833),
    "Maracha": (3.2833, 30.8833),
    "Masaka": (-0.3833, 31.7833),
    "Masindi": (1.6833, 31.7833),
    "Mayuge": (0.5833, 33.5833),
    "Mbale": (1.0833, 34.1833),
    "Mbarara": (-0.6167, 30.6500),
    "Mitooma": (-0.7833, 30.2833),
    "Mityana": (0.3833, 31.5833),
    "Moroto": (2.5833, 34.6833),
    "Moyo": (3.6833, 31.5833),
    "Mpigi": (0.2833, 32.3833),
    "Mubende": (0.5833, 31.3833),
    "Mukono": (0.3833, 32.7833),
    "Nabilatuk": (2.3833, 33.8333),
    "Nakapiripirit": (2.0833, 34.8333),
    "Nakaseke": (1.0833, 32.0000),
    "Nakasongola": (1.3833, 32.3833),
    "Namayingo": (0.2833, 33.8833),
    "Namisindwa": (1.0833, 34.4833),
    "Namutumba": (1.0833, 33.6833),
    "Napak": (2.3833, 34.8333),
    "Nebbi": (2.5833, 31.0833),
    "Ngora": (1.4833, 33.8833),
    "Ntoroko": (1.7833, 30.5833),
    "Ntungamo": (-0.9833, 30.2833),
    "Nwoya": (2.5833, 32.0833),
    "Obongi": (3.3833, 31.5833),
    "Omoro": (2.9833, 32.3833),
    "Otuke": (2.3833, 33.3833),
    "Oyam": (2.3833, 32.3833),
    "Pader": (3.0833, 33.0833),
    "Pakwach": (2.3833, 31.5833),
    "Pallisa": (1.3833, 33.7833),
    "Rakai": (-0.7833, 31.3833),
    "Rubanda": (-1.0833, 29.9833),
    "Rubirizi": (-0.1833, 30.2833),
    "Rukiga": (-1.2833, 29.8833),
    "Rukungiri": (-0.8833, 29.9833),
    "Sembabule": (-0.2833, 31.3833),
    "Serere": (1.4833, 33.5833),
    "Sheema": (-0.5833, 30.2833),
    "Sironko": (1.2833, 34.2833),
    "Soroti": (1.7167, 33.6333),
    "Tororo": (0.7833, 34.1833),
    "Wakiso": (0.3833, 32.4667),
    "Yumbe": (3.4833, 31.2833),
    "Zombo": (3.2833, 30.7833)
}

DISTRICTS = sorted(DISTRICT_COORDS.keys())

st.title("🌱 AgroConsult Uganda: KulimaBrain")
st.markdown("Taking Uganda's Agriculture to greater heights.")

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

    if district not in DISTRICT_COORDS:
        st.error(f"Coordinates not available for {district}")
        st.stop()
    lat, lon = DISTRICT_COORDS[district]

    try:
        import requests_cache
        from retry_requests import retry
        import openmeteo_requests
        cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        openmeteo = openmeteo_requests.Client(session=retry_session)

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

        rain_3d = np.sum(precip[-3:]) if len(precip) >= 3 else 25.0
        min_temp_3d = np.min(temp_min[-3:]) if len(temp_min) >= 3 else 16.0

        dry_days_next_7 = 2

    except Exception as e:
        st.warning(f"Using default weather: {e}")
        rain_3d = 25.0
        min_temp_3d = 16.0
        dry_days_next_7 = 2

    month_guess = 3
    features = [[target_year, month_guess, rain_3d, min_temp_3d, dry_days_next_7]]
    predicted_doy = int(models[crop].predict(features)[0])
    predicted_doy = max(30, min(330, predicted_doy))

    predicted_date = pd.Timestamp(f"{target_year}-01-01") + pd.Timedelta(days=predicted_doy - 1)

    st.info(f"📅 Model predicts planting around {predicted_date.strftime('%B %d')}")

    days_diff = (predicted_date - current_date).days
    if -7 <= days_diff <= 14:
        st.info("📡 Refining with Open-Meteo forecast...")

        try:
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
