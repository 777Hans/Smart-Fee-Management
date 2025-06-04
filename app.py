import streamlit as st
import pandas as pd
import pickle
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(page_title="SmartFee: Predictive Payment Management", layout="wide")

# Sidebar for navigation
st.sidebar.title("SmartFee Navigation")
st.sidebar.markdown("Use the app to predict fee default risk and get payment recommendations.")
page = st.sidebar.selectbox("Choose a feature", ["Single Prediction", "Batch Prediction", "Model Insights"])

# Load model and encoders
@st.cache_resource
def load_model_and_encoders():
    model = xgb.XGBClassifier()
    model.load_model('model.json')
    le_income = pickle.load(open('le_income.pkl', 'rb'))
    le_payment = pickle.load(open('le_payment.pkl', 'rb'))
    return model, le_income, le_payment

model, le_income, le_payment = load_model_and_encoders()

# Single Prediction Page
if page == "Single Prediction":
    st.title("SmartFee: Predictive Payment Management")
    st.markdown("Enter student details to predict fee default risk and receive tailored payment plans.")

    # Input form with validation
    with st.form("prediction_form"):
        st.header("Student Details")
        col1, col2 = st.columns(2)
        with col1:
            income_level = st.selectbox("Income Level", ['low', 'medium', 'high'], help="Select the student's income level.")
            attendance_rate = st.slider("Attendance Rate (%)", 50, 100, 75, help="Adjust the student's attendance rate.")
        with col2:
            academic_score = st.slider("Academic Score", 50, 100, 75, help="Adjust the student's academic score.")
            payment_history = st.selectbox("Payment History", ['on_time', 'late', 'missed'], help="Select the student's payment history.")
        
        submitted = st.form_submit_button("Predict Default Risk")
        
        if submitted:
            try:
                # Preprocess inputs
                income_encoded = le_income.transform([income_level])[0]
                payment_encoded = le_payment.transform([payment_history])[0]
                input_data = np.array([[income_encoded, attendance_rate / 100, academic_score, payment_encoded]])
                
                # Predict
                prob = model.predict_proba(input_data)[0][1]
                st.success(f"Default Risk Probability: {prob:.2%}")

                # Recommendations
                st.subheader("Recommendations")
                if prob > 0.7:
                    st.write("**Suggested Plan**: Offer flexible payment plan (e.g., monthly installments).")
                    st.write("**Reminder**: Send urgent SMS reminder in 3 days.")
                elif prob > 0.3:
                    st.write("**Suggested Plan**: Standard payment plan with slight flexibility.")
                    st.write("**Reminder**: Send email reminder in 5 days.")
                else:
                    st.write("**Suggested Plan**: Standard payment plan.")
                    st.write("**Reminder**: Send polite email reminder in 7 days.")

                # Bar chart for income level comparison
                st.subheader("Default Risk by Income Level")
                income_levels = ['low', 'medium', 'high']
                probs = []
                for inc in income_levels:
                    inc_encoded = le_income.transform([inc])[0]
                    sample_data = np.array([[inc_encoded, attendance_rate / 100, academic_score, payment_encoded]])
                    prob = model.predict_proba(sample_data)[0][1]
                    probs.append(prob)
                fig, ax = plt.subplots()
                sns.barplot(x=income_levels, y=probs, palette="Blues_d")
                ax.set_ylabel("Default Risk Probability")
                ax.set_xlabel("Income Level")
                st.pyplot(fig)

            except Exception as e:
                st.error(f"Error processing inputs: {str(e)}")

# Batch Prediction Page
elif page == "Batch Prediction":
    st.title("Batch Prediction")
    st.markdown("Upload a CSV file with student data to predict default risks for multiple students.")

    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            required_columns = ['income_level', 'attendance_rate', 'academic_score', 'payment_history']
            if not all(col in df.columns for col in required_columns):
                st.error("CSV must contain columns: income_level, attendance_rate, academic_score, payment_history")
            else:
                # Preprocess data
                df['income_level'] = le_income.transform(df['income_level'])
                df['payment_history'] = le_payment.transform(df['payment_history'])
                df['attendance_rate'] = df['attendance_rate'] / 100
                X = df[required_columns]
                
                # Predict
                probs = model.predict_proba(X)[:, 1]
                df['default_risk'] = probs
                df['recommendation'] = df['default_risk'].apply(
                    lambda x: "Flexible plan, urgent SMS in 3 days" if x > 0.7 else
                              "Standard plan, email in 5 days" if x > 0.3 else
                              "Standard plan, polite email in 7 days"
                )
                
                st.write("Prediction Results:")
                st.dataframe(df[['income_level', 'attendance_rate', 'academic_score', 'payment_history', 'default_risk', 'recommendation']])
                
                # Download results
                csv = df.to_csv(index=False)
                st.download_button("Download Results", csv, "predictions.csv", "text/csv")
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# Model Insights Page
elif page == "Model Insights":
    st.title("Model Insights")
    st.markdown("Explore the model's performance and feature importance.")
    
    # Model accuracy (from training)
    st.subheader("Model Performance")
    st.write("Model Accuracy: 74% (based on test set evaluation during training)")
    
    # Feature importance
    st.subheader("Feature Importance")
    feature_names = ['Income Level', 'Attendance Rate', 'Academic Score', 'Payment History']
    importance = model.feature_importances_
    fig, ax = plt.subplots()
    sns.barplot(x=importance, y=feature_names, palette="Greens_d")
    ax.set_xlabel("Importance Score")
    st.pyplot(fig)