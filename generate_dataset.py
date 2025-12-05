import numpy as np
import pandas as pd
import os

# Ensure the data folder exists
os.makedirs("data", exist_ok=True)

# Number of rows
n = 1000

np.random.seed(42)

hours_studied = np.random.uniform(0, 10, n)
attendance = np.random.uniform(50, 100, n)
previous_score = np.random.uniform(0, 100, n)
sleep_hours = np.random.uniform(4, 10, n)
internet_usage = np.random.uniform(0, 6, n)
parent_involvement = np.random.randint(1, 6, n)

# Generate final exam score using a realistic formula
noise = np.random.normal(0, 8, n)

final_score = (
    hours_studied * 5
    + attendance * 0.3
    + previous_score * 0.4
    + sleep_hours * 2
    - internet_usage * 1.5
    + parent_involvement * 3
    + noise
)

# Clip score between 0 and 100
final_score = np.clip(final_score, 0, 100)

# Combine into DataFrame
df = pd.DataFrame({
    "hours_studied": hours_studied,
    "attendance": attendance,
    "previous_score": previous_score,
    "sleep_hours": sleep_hours,
    "internet_usage": internet_usage,
    "parent_involvement": parent_involvement,
    "final_score": final_score
})

# Save CSV
df.to_csv("data/students.csv", index=False)

print("Dataset created successfully → data/students.csv")