import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import warnings

# Suppress specific scikit-learn warnings for cleaner logs
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=Warning)

app = FastAPI(
    title="Iris Species Predictor API",
    description="An API to predict the species of an Iris flower based on its measurements.",
    version="1.0.0"
)

# Define the input data model using Pydantic for validation
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

    class Config:
        schema_extra = {
            "example": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2,
            }
        }

# Load the trained model
model = joblib.load("models/model.joblib")

@app.post("/predict")
def predict(iris: IrisFeatures):
    """Predict the Iris species from input features."""
    data = pd.DataFrame([iris.dict()])
    prediction = model.predict(data)[0]
    return {"species": prediction}
