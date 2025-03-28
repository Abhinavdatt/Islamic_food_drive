import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load the Random Forest model
with open('random_forest_model.pkl', 'rb') as file:
    model = pickle.load(file)

# App title and description
st.title("🍱 Islamic Food Drive - Pickup Prediction")
st.markdown("This app predicts the number of daily food pickups using past data.")

# Input form
st.subheader("📋 Enter Daily Data")
lag_7 = st.number_input("Pickups 7 Days Ago (lag_7)", min_value=0, step=1)
lag_14 = st.number_input("Pickups 14 Days Ago (lag_14)", min_value=0, step=1)
family_size = st.slider("Average Family Size", min_value=1, max_value=10, value=4)

# Prediction
if st.button("📈 Predict Number of Pickups"):
    input_df = pd.DataFrame([{
        'lag_7': lag_7,
        'lag_14': lag_14,
        'Family_size': family_size
    }])

    prediction = model.predict(input_df)[0]
    prediction_rounded = int(round(prediction))

    st.success(f"🔮 Predicted Number of Pickups: **{prediction_rounded}**")
