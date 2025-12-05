import joblib

def save_model(model, path):
    """
    Save trained model to disk.
    """
    joblib.dump(model, path)

def load_model(path):
    """
    Load model from disk.
    """
    return joblib.load(path)