import joblib
import os

def save_model(model, file_path):
    """
    Save trained model to disk.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    joblib.dump(model, file_path)
    print("Model saved successfully.")

def load_model(file_path):
    """
    Load trained model from disk.
    """
    model = joblib.load(file_path)
    print("Model loaded successfully.")
    return model