import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
import pytest

def test_model_accuracy():
    model = joblib.load("models/model.joblib")
    df = pd.read_csv("data/data.csv")
    X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    y = df["species"]
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    assert acc >= 0.6, f"Model accuracy below threshold: {acc}"