from src.data_loader import load_data
from src.preprocess import preprocess
from src.save_load import save_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def main():
    # Load data
    df = load_data("data/students.csv")
    if df is None:
        return

    # Preprocess
    df = preprocess(df)
    if df is None:
        return

    # Split features and target
    X = df.drop("final_score", axis=1)
    y = df["final_score"]

    # Train model
    model = LinearRegression()
    model.fit(X, y)

    # Predict
    predictions = model.predict(X)

    # Evaluate
    mae = mean_absolute_error(y, predictions)
    mse = mean_squared_error(y, predictions)
    r2 = r2_score(y, predictions)

    print("Model Evaluation Results")
    print(f"MAE: {mae}")
    print(f"MSE: {mse}")
    print(f"R²: {r2}")

    # Save model
    save_model(model, "models/linear_model.pkl")

if __name__ == "__main__":
    main()