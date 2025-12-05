from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from src.save_load import load_model

# Load trained model
model = load_model("models/best_student_model.pkl")

app = FastAPI(title="Student Performance Prediction API")

# Request Body Schema
class StudentInput(BaseModel):
    hours_studied: float
    attendance: float
    previous_score: float
    sleep_hours: float
    internet_usage: float
    parent_involvement: float

@app.get("/")
def home():
    return {"message": "Welcome to the Student Performance Prediction API"}

@app.post("/predict")
def predict_score(data: StudentInput):
    # Convert input to numpy array
    features = np.array([[
        data.hours_studied,
        data.attendance,
        data.previous_score,
        data.sleep_hours,
        data.internet_usage,
        data.parent_involvement
    ]])

    # Model prediction
    prediction = model.predict(features)[0]

    return {
        "predicted_final_score": round(prediction, 2)
    }
    