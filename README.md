# Customer Churn Prediction System

## Problem Statement

Telecom companies lose significant revenue when customers cancel subscriptions. 

This system predicts which customers are likely to churn, enabling targeted 

retention strategies before customers leave.

## Results

| Model | AUC-ROC | F1 Score |

|-------|---------|----------|

| Logistic Regression | 0.8472 | 0.5882 |

| Random Forest | 0.8211 | 0.5385 |

| XGBoost (Tuned) | 0.8458 | 0.5710 |

**Winner: Logistic Regression** — outperformed complex models due to 

largely linear relationships in the dataset and effective feature engineering.

## Key Business Insights

- Month-to-month contract customers churn 3x more than 2-year contract customers

- Fiber optic internet customers have highest churn rate due to premium pricing

- New customers (tenure < 12 months) are the highest risk segment

- Customers using more services churn significantly less

## Feature Engineering

Created 4 domain-driven features that improved model performance:

- **ChargePerDay** — MonthlyCharges / (tenure + 1)

- **TotalServices** — count of subscribed services

- **IsNewCustomer** — binary flag for tenure < 12 months

- **ChargeRatio** — MonthlyCharges / (TotalCharges + 1)

Two engineered features ranked in top 10 by SHAP importance.

## Tech Stack

- **ML:** Scikit-learn, XGBoost, SHAP

- **Data:** Pandas, NumPy

- **API:** FastAPI, Uvicorn

- **Visualization:** Matplotlib, Seaborn



## Project Structure

churn-predictor/ 

├── data/ # Raw dataset 

├── notebooks/ # EDA and modeling (Google Colab) 

├── src/ # Preprocessing and training scripts 

├── models/ # Saved model and scaler 

├── api/ # FastAPI endpoint 

└── [README.md](http://README.md)



## API Usage

### Run locally

```bash

uvicorn api.main:app --reload

```

### Predict endpoint

**POST** `http://127.0.0.1:8000/predict`

Example request:

```json

{

  "tenure": 2,

  "MonthlyCharges": 94.0,

  "TotalCharges": 188.0,

  "InternetService_Fiber_optic": 1,

  "Contract_Two_year": 0

}

```

Example response:

```json

{

  "churn_probability": 0.7231,

  "churn_prediction": "Yes",

  "risk_level": "High"

}

```

## Dataset

IBM Telco Customer Churn Dataset — 7,043 customers, 21 features