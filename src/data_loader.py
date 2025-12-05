import pandas as pd

def load_data(path):
    """
    Loads a CSV file and returns a pandas DataFrame.
    """
    try:
        df = pd.read_csv(path)
        print("Data loaded successfully.")
        return df
    except Exception as e:
        print("Error loading data:", e)
        return None