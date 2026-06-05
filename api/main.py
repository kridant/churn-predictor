from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# Load model and scaler
_BASE_DIR = Path(__file__).resolve().parent
_MODELS_DIR = _BASE_DIR.parent / "models"

with open(_MODELS_DIR / "churn_model.pkl", "rb") as f:
    model = pickle.load(f)

with open(_MODELS_DIR / "scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Initialize FastAPI app
app = FastAPI(title="Customer Churn Predictor API")

# Define input schema
class CustomerData(BaseModel):
    tenure: float
    MonthlyCharges: float
    TotalCharges: float
    gender: int
    SeniorCitizen: int
    Partner: int
    Dependents: int
    PhoneService: int
    PaperlessBilling: int
    MultipleLines_Yes: int
    MultipleLines_No_phone_service: int = 0
    InternetService_Fiber_optic: int = 0
    InternetService_No: int = 0
    OnlineSecurity_Yes: int = 0
    OnlineSecurity_No_internet_service: int = 0
    OnlineBackup_Yes: int = 0
    OnlineBackup_No_internet_service: int = 0
    DeviceProtection_Yes: int = 0
    DeviceProtection_No_internet_service: int = 0
    TechSupport_Yes: int = 0
    TechSupport_No_internet_service: int = 0
    StreamingTV_Yes: int = 0
    StreamingTV_No_internet_service: int = 0
    StreamingMovies_Yes: int = 0
    StreamingMovies_No_internet_service: int = 0
    Contract_One_year: int = 0
    Contract_Two_year: int = 0
    PaymentMethod_Credit_card_automatic: int = 0
    PaymentMethod_Electronic_check: int = 0
    PaymentMethod_Mailed_check: int = 0

@app.get("/")
def home():
    return {"message": "Churn Predictor API is running"}

@app.post("/predict")
def predict(customer: CustomerData):
    # Convert input to dataframe
    input_dict = customer.dict()

    # Fix column names - replace underscores back to spaces to match training
    rename_map = {
        "MultipleLines_No_phone_service": "MultipleLines_No phone service",
        "InternetService_Fiber_optic": "InternetService_Fiber optic",
        "InternetService_No": "InternetService_No",
        "OnlineSecurity_No_internet_service": "OnlineSecurity_No internet service",
        "OnlineBackup_No_internet_service": "OnlineBackup_No internet service",
        "DeviceProtection_No_internet_service": "DeviceProtection_No internet service",
        "TechSupport_No_internet_service": "TechSupport_No internet service",
        "StreamingTV_No_internet_service": "StreamingTV_No internet service",
        "StreamingMovies_No_internet_service": "StreamingMovies_No internet service",
        "Contract_One_year": "Contract_One year",
        "Contract_Two_year": "Contract_Two year",
        "PaymentMethod_Credit_card_automatic": "PaymentMethod_Credit card (automatic)",
        "PaymentMethod_Electronic_check": "PaymentMethod_Electronic check",
        "PaymentMethod_Mailed_check": "PaymentMethod_Mailed check",
    }

    renamed_dict = {}
    for key, value in input_dict.items():
        new_key = rename_map.get(key, key)
        renamed_dict[new_key] = value

    data = pd.DataFrame([renamed_dict])
    
    # Scale numerical columns
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges',
                     'ChargePerDay', 'TotalServices', 'ChargeRatio']
    
    # Add engineered features
    data['ChargePerDay'] = data['MonthlyCharges'] / (data['tenure'] + 1)
    data['TotalServices'] = (data['PhoneService'] + 
                            data['OnlineSecurity_Yes'] +
                            data['OnlineBackup_Yes'] + 
                            data['DeviceProtection_Yes'] +
                            data['TechSupport_Yes'] + 
                            data['StreamingTV_Yes'] +
                            data['StreamingMovies_Yes'])
    data['IsNewCustomer'] = (data['tenure'] < 12).astype(int)
    data['ChargeRatio'] = data['MonthlyCharges'] / (data['TotalCharges'] + 1)
    
    # Scale numerical columns
    data[numerical_cols] = scaler.transform(data[numerical_cols])

    # Match training feature order before prediction
    data = data[model.feature_names_in_]
    
    # Get prediction
    churn_prob = model.predict_proba(data)[0][1]
    churn_prediction = int(churn_prob >= 0.5)
    
    # Risk level
    if churn_prob >= 0.7:
        risk = "High"
    elif churn_prob >= 0.4:
        risk = "Medium"
    else:
        risk = "Low"
    
    return {
        "churn_probability": round(float(churn_prob), 4),
        "churn_prediction": "Yes" if churn_prediction == 1 else "No",
        "risk_level": risk
    }