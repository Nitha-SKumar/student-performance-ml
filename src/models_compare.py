from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.tuning import tune_random_forest

def compare_models(X, y):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -----------------------------
    # 1. LINEAR REGRESSION PIPELINE
    # -----------------------------
    lr_model = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression())
    ])
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)

    lr_mae = mean_absolute_error(y_test, lr_preds)
    lr_mse = mean_squared_error(y_test, lr_preds)
    lr_r2 = r2_score(y_test, lr_preds)

   # -----------------------------
# 2. RANDOM FOREST with TUNING
# -----------------------------
    print("\n🔧 Running hyperparameter tuning for RandomForest...")
    rf_model = tune_random_forest(X_train, y_train)

    # Predict
    rf_preds = rf_model.predict(X_test)

    # Evaluate
    rf_mae = mean_absolute_error(y_test, rf_preds)
    rf_mse = mean_squared_error(y_test, rf_preds)
    rf_r2 = r2_score(y_test, rf_preds)

    print("\nRandom Forest Performance:")
    print(f"MAE: {rf_mae:.4f}")
    print(f"MSE: {rf_mse:.4f}")
    print(f"R²:  {rf_r2:.4f}")
    print("\nRandom Forest Performance:")
    print(f"MAE: {rf_mae:.4f}")
    print(f"MSE: {rf_mse:.4f}")
    print(f"R²:  {rf_r2:.4f}")

    print("----------------------------------")

    # Choose best model
    best_model = rf_model if rf_r2 > lr_r2 else lr_model
    print("\n🏆 BEST MODEL SELECTED:", "RandomForest" if best_model == rf_model else "Linear Regression")

    return best_model