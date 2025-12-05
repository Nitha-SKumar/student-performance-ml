import streamlit as st
import numpy as np
from src.save_load import load_model

# Load the best model
model = load_model("models/best_student_model.pkl")

st.title("🎓 Student Performance Predictor")
st.write("Enter the details below to predict the student's final exam score.")

# Input fields
hours_studied = st.number_input("Hours Studied Per Day", min_value=0.0, max_value=12.0, step=0.5)
attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0, step=1.0)
previous_score = st.number_input("Previous Exam Score", min_value=0.0, max_value=100.0, step=1.0)
sleep_hours = st.number_input("Sleep Hours Per Day", min_value=0.0, max_value=12.0, step=0.5)
internet_usage = st.number_input("Internet Usage (hours/day)", min_value=0.0, max_value=10.0, step=0.5)
parent_involvement = st.slider("Parent Involvement (1 = Low, 5 = High)", 1, 5)

# Predict button
if st.button("Predict Score"):
    # Prepare features
    input_features = np.array([[ 
        hours_studied,
        attendance,
        previous_score,
        sleep_hours,
        internet_usage,
        parent_involvement
    ]])

    # Prediction
    predicted_score = model.predict(input_features)[0]

    st.success(f"📘 Predicted Final Score: **{predicted_score:.2f}**")