import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib
import os

from huggingface_hub import HfApi, login

HF_TOKEN = os.getenv("HF_TOKEN")  # Optional if repo is public

model_path = hf_hub_download(
    repo_id="tam3222/Engine_Model",
    filename="models/RandomForest.pkl",  
    repo_type="model",
    token=HF_TOKEN  # can be None for public repos
)

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Customer Conversion Prediction
st.title("Predictive Maintenance App")
st.write("This app predicts the engine condition.")
st.write("Please enter the customer details below:")

# Collect user input
engine_rpm = st.number_input("Engine_RPM", min_value=0, value=700)
lub_oil_pressure = st.number_input("Pressure of the lubricating oil in the engine(kPa)", min_value=0, value=3)
fuel_pressure = st.number_input("Pressure at which fuel is supplied to the engine(kPa)", min_value=0, value=3)
coolant_pressure = st.number_input("Coolant_Pressure(kPa)", min_value=0, value=2)
lub_oil_temperature = st.number_input("Temperature of the lubricating oil (°C) ", value=75)
coolant_temperature = st.number_input("Coolant_Temperature (°C)", value=78)

# Prepare input DataFrame
input_data = pd.DataFrame([{
    'engine_rpm': engine_rpm,
    'lub_oil_pressure': lub_oil_pressure,
    'fuel_pressure': fuel_pressure,
    'coolant_pressure': coolant_pressure,
    'lub_oil_temperature': lub_oil_temperature,
    'coolant_temperature': coolant_temperature,
}])

# Classification threshold
classification_threshold = 0.45

# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "Faulty" if prediction == 1 else "Healthy"
    st.write(f"Based on the information provided, the engine is {result}.")
