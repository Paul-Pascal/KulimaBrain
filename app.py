import streamlit as st
import pandas as pd
import pickle
import os

# Load models
models = {}
for crop in ["maize", "beans"]:
    path = f"models/saved/{crop}_model.pkl"
    if os.path.exists(path):
        with open(path, "rb") as f:
            models[crop] = pickle.load(f)

# Load district list from your data
climate_df = pd.read_csv("/content/Uganda_Climate_2000.csv")
DISTRICTS = sorted(climate_df['ADM2_NAME'].unique().tolist())

st.title("🌱 AgriAI Uganda: Farming Advisor")
district = st.selectbox("District", DISTRICTS)
date = st.date_input("Date", value=pd.to_datetime("2025-03-10"))
crop = st.selectbox("Crop", list(models.keys()))

if st.button("Get Advice"):
    if crop not in models:
        st.error("Model not loaded.")
    else:
        # Simulate prediction (replace with real logic later)
        st.success(f"✅ Good time to plant {crop}! Recent rain: 28.5mm, warm soil.")
        st.info(f"📍 {district} | 📅 {date} | 🌾 {crop}")
