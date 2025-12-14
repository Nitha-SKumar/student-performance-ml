# Student Performance Prediction — Machine Learning Project

![CI](https://github.com/Nitha-SKumar/student-performance-ml/actions/workflows/ci.yml/badge.svg)

## 📌 Project Overview
This project builds an end-to-end Machine Learning pipeline to predict a student’s final score based on multiple academic and lifestyle factors.

The project demonstrates professional ML engineering practices including modular code, testing, and continuous integration.

---

## 🎯 Problem Statement
Given student-related features such as:
- Hours studied
- Attendance percentage
- Previous exam score
- Sleep hours
- Internet usage
- Parent involvement

Predict the **final academic score** using regression models.

---

## 🧠 Solution Approach
The solution follows a structured ML pipeline:
1. Load and validate data
2. Clean and preprocess features
3. Train regression models
4. Evaluate performance using standard metrics
5. Persist trained models
6. Validate pipeline with automated tests
7. Run CI pipeline on every push

---

## 🗂️ Project Structure
```text
student-performance-ml/
│── main.py
│── README.md
│── pytest.ini
│── .gitignore
│── data/
│   └── students.csv
│── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocess.py
│   └── save_load.py
│── tests/
│   └── test_pipeline.py
│── .github/
│   └── workflows/
│       └── ci.yml