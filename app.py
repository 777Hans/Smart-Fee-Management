import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import base64

# Set page config for wide layout and custom theme
st.set_page_config(page_title="SmartFee: Fee Default Prediction", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for enhanced styling
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stButton>button { background-color: #4CAF50; color: white; border-radius: 10px; }
    .stSlider { background-color: #e6f3ff; padding: 10px; border-radius: 10px; }
    .stSelectbox { background-color: #e6f3ff; padding: 10px; border-radius: 10px; }
    .sidebar .sidebar-content { background-color: #ffffff; }
    .reportview-container .main .block-container { padding: 2rem; }
    h1, h2, h3 { color: #2c3e50; }
    .stAlert { background-color: #dff0d8; color: #3c763d; border-radius: 10px; }
    .prediction-card { background-color: #ffffff; padding: 20px; border-radius: 15px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin-bottom: 20px; }
    .explanation-box { background-color: #f9f9f9; padding: 15px; border-left: 5px solid #4CAF50; border-radius: 5px; }
    </style>
""", unsafe_allow_html=True)

# Load model
model = joblib.load('model.pkl')

# Function to get base64 image for sidebar logo
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Sidebar with logo and info
try:
    logo_base64 = get_base64_image("logo.png")  # Ensure logo.png exists in the same directory
    st.sidebar.markdown(f'<img src="data:image/png;base64,{logo_base64}" width="100%">', unsafe_allow_html=True)
except FileNotFoundError:
    st.sidebar.image("https://via.placeholder.com/150", caption="SmartFee Logo")  # Fallback placeholder

st.sidebar.title("SmartFee Dashboard")
st.sidebar.info("Enter student details to predict fee default risk. Adjust sliders and dropdowns to see real-time changes in predictions.")

# Main app
st.title("SmartFee: Fee Default Prediction")
st.markdown("Predict whether a student is likely to default on fee payments based on their profile. Explore interactive visualizations and insights below.")

# Layout: Inputs and Outputs in two columns
col1, col2 = st.columns([1, 1])

# User inputs in left column
with col1:
    st.header("Enter Student Details")
    income_level = st.selectbox("Income Level", ['low', 'medium', 'high'], help="Select the student's household income level.")
    attendance_rate = st.slider("Attendance Rate (%)", 50.0, 100.0, 75.0, help="Percentage of classes attended.") / 100.0
    academic_score = st.slider("Academic Score", 50.0, 100.0, 75.0, help="Student's academic performance score.")
    payment_history = st.selectbox("Payment History", ['on_time', 'late', 'missed'], help="Past payment behavior.")

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

# Prediction and visualizations in right column
with col2:
    st.header("Prediction & Insights")
    if st.button("Predict", key="predict_button"):
        with st.spinner("Calculating prediction..."):
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]
            result = "Will Default" if prediction else "Will Not Default"

            # Prediction card
            st.markdown(f"""
                <div class="prediction-card">
                    <h3>Prediction Result</h3>
                    <p><strong>Prediction:</strong> {result}</p>
                    <p><strong>Default Probability:</strong> {probability:.2%}</p>
                </div>
            """, unsafe_allow_html=True)

            # Probability gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                title={'text': "Default Probability (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#4CAF50" if probability < 0.5 else "#e74c3c"},
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig_gauge.update_layout(height=250)
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Feature importance (simplified for display)
            feature_names = model.named_steps['preprocessor'].get_feature_names_out()
            importances = model.named_steps['classifier'].feature_importances_
            feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
            feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).head(5)

            fig_importance = px.bar(
                feature_importance_df,
                x='Importance',
                y='Feature',
                title="Top 5 Factors Influencing Prediction",
                color='Importance',
                color_continuous_scale='Viridis'
            )
            fig_importance.update_layout(yaxis={'tickfont': {'size': 10}}, height=300)
            st.plotly_chart(fig_importance, use_container_width=True)

# Explanation section
st.header("Why This Prediction?")
st.markdown("""
    <div class="explanation-box">
        The prediction is based on the following key factors:
        <ul>
            <li><strong>Income Level:</strong> Lower income levels may increase default risk due to financial constraints.</li>
            <li><strong>Attendance Rate:</strong> Poor attendance (below 55%) can correlate with higher default risk.</li>
            <li><strong>Academic Score:</strong> Lower scores may indicate disengagement, impacting payment likelihood.</li>
            <li><strong>Payment History:</strong> Past missed payments strongly predict future defaults.</li>
            <li><strong>Interactions:</strong> Combinations like low income and missed payments amplify risk.</li>
        </ul>
        The model uses these factors to calculate a probability of default, shown in the gauge above.
    </div>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
    <hr>
    <p style="text-align: center; color: #7f8c8d;">© 2025 SmartFee | Powered by Streamlit & XGBoost</p>
""", unsafe_allow_html=True)