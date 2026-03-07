import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime
import requests
from io import BytesIO

# ✅ SAFE MODEL LOADING
@st.cache_resource
def load_models():
    urls = {
        "maize": "https://github.com/Paul-Pascal/KulimaBrain/releases/download/v1.0/maize_doy_model.joblib",
        "beans": "https://github.com/Paul-Pascal/KulimaBrain/releases/download/v1.0/beans_doy_model.joblib"
    }
    models = {}
    for crop, url in urls.items():
        resp = requests.get(url)
        resp.raise_for_status()
        models[crop] = joblib.load(BytesIO(resp.content))['model']
    return models

models = load_models()

# Districts
DISTRICTS = [
    "Wakiso", "Mbarara", "Gulu", "Mbale", "Kampala", "Luwero", "Masaka", "Jinja"
]  # Add more if needed

# UI
st.title("🌱 AgroConsult Uganda: KulimaBrain")
district = st.selectbox("District", DISTRICTS)
target_date = st.date_input("Target Date", value=datetime.today())
crop = st.selectbox("Crop", list(models.keys()))

# Prediction
if st.button("Get Advice"):
    try:
        features = [[
            int(target_date.year),
            int(target_date.month),
            25.0,  # rain_3d
            16.0,  # min_temp
            2.0    # dry_days
        ]]
        doy = int(models[crop].predict(features)[0])
        doy = np.clip(doy, 30, 330)
        pred_date = pd.Timestamp(f"{target_date.year}-01-01") + pd.Timedelta(days=int(doy)-1)
        st.success(f"✅ Plant {crop} around {pred_date.strftime('%Y-%m-%d')}")
    except Exception as e:
        st.error(f"❌ Failed: {str(e)}")
