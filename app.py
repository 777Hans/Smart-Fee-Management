import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load('model.pkl')

# Streamlit app
st.title("SmartFee: Fee Default Prediction")

# User inputs
st.header("Enter Student Details")
income_level = st.selectbox("Income Level", ['low', 'medium', 'high'])
attendance_rate = st.slider("Attendance Rate (%)", 50.0, 100.0, 75.0) / 100.0
academic_score = st.slider("Academic Score", 50.0, 100.0, 75.0)
payment_history = st.selectbox("Payment History", ['on_time', 'late', 'missed'])

# Create input DataFrame
input_data = pd.DataFrame({
    'income_level': [income_level],
    'attendance_rate': [attendance_rate],
    'academic_score': [academic_score],
    'payment_history': [payment_history]
})

# Add interaction features
input_data['low_income_missed'] = ((input_data['income_level'] == 'low') & (input_data['payment_history'] == 'missed')).astype(int)
input_data['low_income_low_attendance'] = ((input_data['income_level'] == 'low') & (input_data['attendance_rate'] < 0.55)).astype(int)
input_data['missed_low_attendance'] = ((input_data['payment_history'] == 'missed') & (input_data['attendance_rate'] < 0.55)).astype(int)

# Predict
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    result = "Will Default" if prediction else "Will Not Default"
    st.subheader("Prediction Result")
    st.write(f"**Prediction**: {result}")
    st.write(f"**Default Probability**: {probability:.2%}")