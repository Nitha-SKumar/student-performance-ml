def preprocess(df):
    """
    Basic preprocessing for the dataset.
    """
    if df is None:
        print("No data to preprocess.")
        return None

    # Remove rows with missing values
    df = df.dropna()

    return df