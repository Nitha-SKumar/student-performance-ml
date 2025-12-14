from src.data_loader import load_data
from src.preprocess import preprocess

def test_load_data():
    df = load_data("data/students.csv")
    assert df is not None
    assert not df.empty

def test_preprocess():
    df = load_data("data/students.csv")
    df_processed = preprocess(df)
    assert df_processed is not None
    assert not df_processed.empty