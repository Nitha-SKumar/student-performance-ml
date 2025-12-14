import pandas as pd

def load_data(file_path):
    """
    Load CSV data into a pandas DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        print("Data loaded successfully.")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None