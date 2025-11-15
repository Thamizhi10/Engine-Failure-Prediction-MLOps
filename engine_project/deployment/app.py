import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib
import os

from huggingface_hub import HfApi, login
login(os.getenv("HF_TOKEN"))

# Download the model from the Model Hub
model_path = hf_hub_download(
    repo_id="tam3222/tourism",
    filename="best_engine_prediction_model_v1.joblib"
)

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Customer Conversion Prediction
st.title("Predictive Maintenance App")
st.write("This app predicts the engine condition.")
st.write("Please enter the customer details below:")

# Collect user input
Engine_RPM = st.number_input("Engine_RPM", min_value=0, value=700)
Lub_Oil_Pressure = st.number_input("Pressure of the lubricating oil in the engine(kPa)", min_value=0, value=3)
Fuel_Pressure = st.number_input("Pressure at which fuel is supplied to the engine(kPa)", min_value=0, value=3)
Coolant_Pressure = st.number_input("Coolant_Pressure(kPa)", min_value=0, value=2)
Lub_Oil_Temperature = st.number_input("Temperature of the lubricating oil (°C) ", value=75)
Coolant_Temperature = st.number_input("Coolant_Temperature (°C)", value=78)

# Prepare input DataFrame
input_data = pd.DataFrame([{
    'Engine_RPM': Engine_RPM,
    'Lub_Oil_Pressure': Lub_Oil_Pressure,
    'Fuel_Pressure': Fuel_Pressure,
    'Coolant_Pressure': Coolant_Pressure,
    'Lub_Oil_Temperature': Lub_Oil_Temperature,
    'Coolant_Temperature': Coolant_Temperature,
}])

# Classification threshold
classification_threshold = 0.45

# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "Faulty" if prediction == 1 else "Healthy"
    st.write(f"Based on the information provided, the engine is {result}.")
