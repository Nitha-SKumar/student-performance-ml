from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

def tune_random_forest(X_train, y_train):
    """
    Tune hyperparameters of RandomForest using GridSearchCV.
    """
    rf = RandomForestRegressor(random_state=42)

    param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20, 30],
    "max_features": ["sqrt", "log2"],   # FIXED: removed "auto"
    "min_samples_split": [2, 5, 10]
    }

    grid = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=3,
        scoring="r2",
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_train, y_train)

    print("\n🔍 BEST PARAMETERS FOUND:")
    print(grid.best_params_)

    print("\n🏆 BEST CV R² SCORE:", grid.best_score_)

    # Return the best model found
    return grid.best_estimator_