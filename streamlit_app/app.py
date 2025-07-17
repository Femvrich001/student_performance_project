import streamlit as st
import os
import joblib
import numpy as np
import pandas as pd

# Set the base directory (project root)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # One level up from app.py

# Load model and feature columns
model_path = os.path.join(BASE_DIR, '..', 'models', 'logistic_model.pkl')
features_path = os.path.join(BASE_DIR, '..', 'models', 'feature_columns.pkl')

model = joblib.load(model_path)
feature_columns = joblib.load(features_path)

# App Title
st.title("🎓 Student Performance Risk Predictor")
st.markdown("This tool helps lecturers identify students at academic risk early based on their profile and performance.")

# Input Form
st.write("### 📋 Enter student details to predict if they are at academic risk:")

gender = st.selectbox("Gender", ["Male", "Female"])
state_of_origin = st.selectbox("State of Origin Zone", ["Southwest", "Southeast", "Southsouth", "Northcentral", "Northeast", "Northwest"])
residence_type = st.selectbox("Residence Type", ["Hostel", "Off-campus", "With Family"])
parental_education = st.selectbox("Parental Education", ["None", "Primary", "Secondary", "Tertiary"])
internet_access = st.selectbox("Internet Access", ["Yes", "No"])
has_scholarship = st.selectbox("Has Scholarship?", ["Yes", "No"])
waec_score = st.slider("WAEC Score", 180, 400, 250)
jamb_score = st.slider("JAMB Score", 150, 320, 200)
attendance_rate = st.slider("Attendance Rate (%)", 50.0, 100.0, 75.0)
study_hours = st.slider("Study Hours per Week", 0, 40, 10)
assignment_avg = st.slider("Assignment Score Avg", 40, 100, 60)
class_participation = st.selectbox("Class Participation", ["Low", "Medium", "High"])
cgpa100 = st.slider("CGPA100", 0.0, 4.0, 2.0)
cgpa200 = st.slider("CGPA200", 0.0, 4.0, 2.0)

# Encode inputs (same as training encoding logic)
input_data = np.array([[
    1 if gender == "Male" else 0,
    ["Southwest", "Southeast", "Southsouth", "Northcentral", "Northeast", "Northwest"].index(state_of_origin),
    ["Hostel", "Off-campus", "With Family"].index(residence_type),
    ["None", "Primary", "Secondary", "Tertiary"].index(parental_education),
    1 if internet_access == "Yes" else 0,
    1 if has_scholarship == "Yes" else 0,
    waec_score,
    jamb_score,
    attendance_rate,
    study_hours,
    assignment_avg,
    ["Low", "Medium", "High"].index(class_participation),
    cgpa100,
    cgpa200
]])

# Predict Button
if st.button("🔍 Predict Risk"):
    prediction = model.predict(input_data)[0]
    result = "🚨 At Risk" if prediction == 1 else "✅ Not at Risk"
    st.subheader(f"Prediction: {result}")