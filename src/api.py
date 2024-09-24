from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os

# Initialize FastAPI app
app = FastAPI()

# Define the request body model using Pydantic
class PredictionInput(BaseModel):
    Customers: int
    StoreType: str
    CompetitionDistance: float
    Promo: int
    Store: int
    DayOfWeek: int
    IsWeekend: int

# Define a root endpoint to check the API status
@app.get("/")
def read_root():
    return {"message": "API is up and running!"}

# Endpoint to make predictions
@app.post("/predict")
def predict_sales(data: PredictionInput):
    # Load the model lazily
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "../notebook/model/sales_model_2024-09-23-21-04-57.pkl")
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {e}")

    # Convert input data to DataFrame
    input_data = pd.DataFrame([data.dict()])

    try:
        # Make predictions using the loaded model
        prediction = model.predict(input_data)
        return {"prediction": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
