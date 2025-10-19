import json
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def run_inference(data_path: str, model_path: str, metrics_path: str = "metrics.json"):
    """Run inference using the trained DecisionTree model and save evaluation metrics."""

    # Load model
    print(f"🔹 Loading model from {model_path} ...")
    model = joblib.load(model_path)

    # Load data
    print(f"🔹 Loading data from {data_path} ...")
    df = pd.read_csv(data_path)

    X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y_true = df['species']

    # Make predictions
    print("🔹 Running inference ...")
    y_pred = model.predict(X)

    # Compute metrics
    accuracy = round(accuracy_score(y_true, y_pred), 4)
    print(f"✅ Accuracy: {accuracy}")

    report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred).tolist()

    # Prepare metrics dictionary
    metrics_dict = {
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": cm
    }

    # Save metrics to JSON file
    with open(metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)

    print(f"📊 Metrics saved to {metrics_path}")


if __name__ == "__main__":
    data_path = "data/data.csv"
    model_path = "models/model.joblib"
    run_inference(data_path, model_path)
