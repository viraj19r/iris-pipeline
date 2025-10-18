import os
import time
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics


def train_and_store(data_path: str):
    """Train a DecisionTreeClassifier on Iris data and save artifacts to GCS."""

    df = pd.read_csv(data_path)

    print(f"Dataset shape: {df.shape}")

    # Split data
    X = df[['sepal_length','sepal_width','petal_length','petal_width']]
    y = df['species']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, stratify=y, random_state=42
    )

    # Train model
    model = DecisionTreeClassifier(max_depth=3, random_state=1)
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)
    acc = metrics.accuracy_score(y_test, preds)
    print(f"Accuracy: {acc:.3f}")

    # Save locally
    model_path = "./models/model.joblib"
    joblib.dump(model, model_path)

    return model_path, X_test, y_test


if __name__ == "__main__":
    data_path = "data/data.csv"
    model_path, X_test, y_test = train_and_store(data_path)
    print(f"Model saved to: {model_path}")
    