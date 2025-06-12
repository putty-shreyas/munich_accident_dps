from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load model & encoders
model = joblib.load("models/lgbm_model.joblib")
encoders = joblib.load("models/label_encoders.joblib")

app = FastAPI()

@app.get("/")
def read_root():
    return {
        "message": "🚦 Welcome to the Munich Traffic Accident Forecast API!",
        "usage": "POST a JSON to /predict with 'year' and 'month'"
    }

class RequestInput(BaseModel):
    year: int
    month: int

@app.post("/predict")
def predict(data: RequestInput):
    year = data.year
    month = data.month

    # Encode categorical inputs
    category_encoded = encoders['Category'].transform(['Alkoholunfälle'])[0]
    type_encoded = encoders['Accident_type'].transform(['insgesamt'])[0]

    input_array = np.array([[category_encoded, type_encoded, year, month]])
    prediction = model.predict(input_array)[0]

    return {"prediction": round(prediction, 2)}