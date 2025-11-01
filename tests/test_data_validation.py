import pandas as pd
import pytest

def test_data_integrity():
    df = pd.read_csv("data/data.csv")
    assert not df.isnull().values.any(), "Dataset contains missing values"

def test_expected_columns():
    df = pd.read_csv("data/data.csv")
    expected_cols = {"sepal_length", "sepal_width", "petal_length", "petal_width", "species"}
    assert expected_cols.issubset(df.columns), f"Missing columns: {expected_cols - set(df.columns)}"

def test_model_file_exists():
    import os
    assert os.path.exists("models/model.joblib"), "Model file missing!"