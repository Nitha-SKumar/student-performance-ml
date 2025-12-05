from src.data_loader import load_data
from src.preprocess import preprocess
from src.train import train_model
from src.evaluate import evaluate
from src.save_load import save_model
from sklearn.model_selection import train_test_split
from src.models_compare import compare_models

def main():
    # Load data
    df = load_data("data/students.csv")
    if df is None:
        return

    # Preprocess clean data
    df = preprocess(df)

    # Check if final_score exists
    if "final_score" not in df.columns:
        print("ERROR: final_score column missing.")
        print("Columns:", df.columns)
        return

    # Features and target
    X = df.drop("final_score", axis=1)
    y = df["final_score"]

    # Compare Linear Regression vs RandomForest
    best_model = compare_models(X, y)

    # Save the best model
    save_model(best_model, "models/best_student_model.pkl")
    print("Best model saved successfully.")

if __name__ == "__main__":
    main()