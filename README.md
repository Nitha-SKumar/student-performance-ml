Student Performance Predictor (ML Project)

A complete end-to-end Machine Learning project that predicts a student’s final score based on study behavior, attendance, sleep, and lifestyle factors.
This project includes data preprocessing, model training, model comparison, hyperparameter tuning, API deployment, Streamlit UI, and a fully modular ML pipeline.

⸻

Features
	•	Predicts student final exam scores
	•	Modular ML code (clean & scalable)
	•	Linear Regression vs Random Forest comparison
	•	Automatic best-model selection
	•	Hyperparameter tuning with GridSearchCV
	•	FastAPI backend for real-time predictions
	•	Streamlit UI for a user-friendly interface
	•	Saves & loads models using joblib
	•	Ready for CI/CD & deployment

⸻

Project Structure

student_ml_clean/
│── api.py
│── main.py
│── predict_one.py
│── streamlit_app.py
│── data/
│── models/
│── src/
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   ├── save_load.py
│   ├── tuning.py
│   └── models_compare.py
│── .gitignore
│── README.md

⸻

ML Pipeline

Raw Data → Preprocessing → Train/Test Split → Model Training → Hyperparameter Tuning → Model Comparison → Save Best Model → API/Streamlit Prediction

⸻

Model Performance
| Model                  | MAE   | MSE    | R²    |
|------------------------|-------|--------|-------|
| Linear Regression      | ~6.00 | ~55.87 | ~0.79 |
| **Random Forest (Selected)** | **~5.66** | **~51.67** | **~0.81** |
Hyperparameter Tuning Summary

GridSearchCV tested combinations of:
	•	n_estimators
	•	max_depth
	•	max_features
	•	min_samples_split

Best parameters found:

n_estimators = 200
max_depth = None
max_features = sqrt
min_samples_split = 2

⸻

FastAPI — Usage

Start API

uvicorn api:app –reload

Example JSON Input

{
“hours_studied”: 5,
“attendance”: 87,
“previous_score”: 70,
“sleep_hours”: 7,
“internet_usage”: 2,
“parent_involvement”: 4
}

Example Output

{ “predicted_score”: 92.5 }

API docs available at:
http://127.0.0.1:8000/docs

⸻

Streamlit UI

Start Streamlit App

streamlit run streamlit_app.py

User enters study habits → Model returns predicted final score.

⸻

How to Run the Project

1. Clone the repository

git clone https://github.com/Nitha-SKumar/student-performance-ml.git
cd student-performance-ml

2. Install dependencies

pip install -r requirements.txt

3. Train the model

python main.py

4. Make a prediction

python predict_one.py

5. Run API

uvicorn api:app –reload

6. Launch Streamlit UI

streamlit run streamlit_app.py

⸻
 Future Improvements
	•	Add SHAP explainability
	•	Add validation & monitoring
	•	Deploy to Render/Railway/Streamlit Cloud
	•	Add Docker support
	•	Add CI/CD with GitHub Actions
	•	Integrate database
	•	Add XGBoost & LightGBM models

⸻

Technologies Used

Python
Pandas
NumPy
Scikit-Learn
RandomForestRegressor
GridSearchCV
FastAPI
Uvicorn
Streamlit
Joblib
Git & GitHub

