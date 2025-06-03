import streamlit as st
import pandas as pd
import pickle
import numpy as np
import xgboost as xgb

# Load model and encoders
model = xgb.XGBClassifier()
model.load_model('model.json')  # Load JSON model
le_income = pickle.load(open('le_income.pkl', 'rb'))
le_payment = pickle.load(open('le_payment.pkl', 'rb'))

# Streamlit app
st.title("SmartFee: Predictive Payment Management")

st.header("Enter Student Details")
income_level = st.selectbox("Income Level", ['low', 'medium', 'high'])
attendance_rate = st.slider("Attendance Rate (%)", 50, 100, 75)
academic_score = st.slider("Academic Score", 50, 100, 75)
payment_history = st.selectbox("Payment History", ['on_time', 'late', 'missed'])

# Preprocess inputs
income_encoded = le_income.transform([income_level])[0]
payment_encoded = le_payment.transform([payment_history])[0]
input_data = np.array([[income_encoded, attendance_rate / 100, academic_score, payment_encoded]])

# Predict
if st.button("Predict Default Risk"):
    prob = model.predict_proba(input_data)[0][1]
    st.write(f"Default Risk Probability: {prob:.2%}")

    # Suggest payment plan and reminder
    if prob > 0.7:
        st.write("**Suggested Plan**: Offer flexible payment plan (e.g., monthly installments).")
        st.write("**Reminder**: Send urgent SMS reminder in 3 days.")
    elif prob > 0.3:
        st.write("**Suggested Plan**: Standard payment plan with slight flexibility.")
        st.write("**Reminder**: Send email reminder in 5 days.")
    else:
        st.write("**Suggested Plan**: Standard payment plan.")
        st.write("**Reminder**: Send polite email reminder in 7 days.")