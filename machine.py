from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# Importing sklearn for model and scaler
from sklearn.preprocessing import StandardScaler

# Import qiskit libraries (no changes needed for these)
from qiskit import *
from qiskit_machine_learning.algorithms import QSVC

import warnings
warnings.filterwarnings('ignore')

# Load trained model and scaler once at startup
qsvc_model = joblib.load("Myqsvm_model.pkl")
scaler = joblib.load("scaler.pkl")

# FastAPI initialization
app = FastAPI()

# Define the Pydantic model for input validation
class LoanData(BaseModel):
    age: float
    person_income: float
    person_home_ownership: int
    person_emp_length	: int
    loan_intent: int
    loan_grade:int
    loan_amnt: float
    loan_int_rate:int
    loan_percent_income: float
    cb_person_default_on_file: int
    cb_person_cred_hist_length: int

# Define the POST route for loan prediction
@app.post("/predict")
async def predict_loan_outcome(data: LoanData):
    try:
        # Convert the input data into a numpy array for prediction
        loan_data = np.array([[data.age, data.person_income, data.person_home_ownership, data.person_emp_length,
        data.loan_intent, data.loan_grade, data.loan_amnt, data.loan_int_rate, data.loan_percent_income, data.cb_person_default_on_file, data.cb_person_cred_hist_length]])

        # Scale the input data using the scaler
        customer_data_scaled = scaler.transform(loan_data)
        
        # Predict the loan outcome using the trained model
        prediction = qsvc_model.predict(customer_data_scaled)
        
        # Return the prediction (0 for default, 1 for non-default) as a JSON response
        return {"prediction": int(prediction[0])}

    except Exception as e:
        # Handle any errors that occur during prediction
        raise HTTPException(status_code=400, detail=str(e))

# To run the application, use the command `uvicorn filename:app --reload`
