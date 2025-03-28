import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load LSTM model
@st.cache_resource
def load_lstm_model():
    return load_model("best_lstm_model.h5")

model = load_lstm_model()

# Load and cache dataset
@st.cache_data
def load_data():
    return pd.read_csv("Islamic_Food_Drive_cleaned_data.csv", parse_dates=["Pickup_date", "collect_scheduled_date"])

data = load_data()

# -----------------------------
# Updated Predict Page for LSTM
def predict():
    st.title("🔮 Predict Pickups using LSTM Model")

    st.write("Enter recent lag values and average family size:")

    lag_7 = st.slider("Pickups 7 Days Ago", 0, 100, 30)
    lag_14 = st.slider("Pickups 14 Days Ago", 0, 100, 28)
    family_size = st.slider("Average Family Size", 1, 10, 4)

    if st.button("📈 Predict"):
        input_array = np.array([[lag_7, lag_14, family_size]])
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_input = scaler.fit_transform(input_array)
        reshaped_input = np.reshape(scaled_input, (scaled_input.shape[0], 1, scaled_input.shape[1]))
        prediction_scaled = model.predict(reshaped_input)
        prediction = scaler.inverse_transform(prediction_scaled)
        predicted_value = int(np.round(prediction[0][0]))

        st.success(f"📦 Estimated Pickups: **{predicted_value}**")

