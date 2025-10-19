import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
# Suppress specific scikit-learn warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=Warning)  # catches InconsistentVersionWarning


def run_inference(data_path: str, model_path: str):
    """Run inference using the trained DecisionTree model and print evaluation metrics."""

    # Load model
    model = joblib.load(model_path)

    # Load data
    df = pd.read_csv(data_path)

    X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y_true = df['species']

    # Make predictions
    print(" Running inference ...")
    y_pred = model.predict(X)

    # Compute metrics
    accuracy = round(accuracy_score(y_true, y_pred), 4)
    print(f" ### Accuracy: {accuracy}\n")

    print("Classification Report:")
    print(classification_report(y_true, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    data_path = "data/data.csv"
    model_path = "models/model.joblib"
    run_inference(data_path, model_path)
