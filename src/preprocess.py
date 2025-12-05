def preprocess(df):
    """
    Basic cleaning of dataset.
    """
    df = df.dropna()   # remove missing values for now
    return df