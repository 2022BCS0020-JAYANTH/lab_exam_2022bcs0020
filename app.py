import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_PATH = "results/model.joblib"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

app = FastAPI()

class WineFeatures(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

@app.get("/")
def home():
    return {"message": "Wine Quality API is running"}

@app.post("/predict")
def predict(features: WineFeatures):
    X = pd.DataFrame([{
        "fixed acidity": features.fixed_acidity,
        "volatile acidity": features.volatile_acidity,
        "citric acid": features.citric_acid,
        "residual sugar": features.residual_sugar,
        "chlorides": features.chlorides,
        "free sulfur dioxide": features.free_sulfur_dioxide,
        "total sulfur dioxide": features.total_sulfur_dioxide,
        "density": features.density,
        "pH": features.pH,
        "sulphates": features.sulphates,
        "alcohol": features.alcohol
    }])

    prediction = model.predict(X)[0]
    wine_quality = int(round(prediction))

    return {
        "name": "batch1_2022BCS0020",
        "wine_quality": wine_quality
    }