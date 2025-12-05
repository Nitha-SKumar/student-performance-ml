import numpy as np
from src.save_load import load_model

model = load_model("models/best_student_model.pkl")

# Input feature values
hours_studied = 5
attendance = 87
previous_score = 70
sleep_hours = 7
internet_usage = 2
parent_involvement = 4

input_features = np.array([[ 
    hours_studied,
    attendance,
    previous_score,
    sleep_hours,
    internet_usage,
    parent_involvement
]])

predicted_score = model.predict(input_features)

print("Predicted Final Score:", round(predicted_score[0], 2))