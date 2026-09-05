import streamlit as st
import joblib

st.title("DeliveryGuard")
st.write("Food Delivery Time Prediction System")

model_path = "models/deliveryguard_model.pkl"
model = joblib.load(model_path)

st.success("Model loaded successfully!")
