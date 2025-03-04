from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from sklearn.impute import SimpleImputer  # Import SimpleImputer
from fastapi.middleware.cors import CORSMiddleware


# Load the trained QSVC model
model = joblib.load("Myqsvm_model.pkl")

# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to match your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request model for input validation
class LoanApplicant(BaseModel):
    age: float
    person_income: float
    person_home_ownership: int
    person_emp_length: int
    loan_intent: int
    loan_grade: int
    loan_amnt: float
    loan_int_rate: float
    loan_percent_income: float
    cb_person_default_on_file: int
    cb_person_cred_hist_length: int

@app.get("/")
def home():
    return {"message": "Quantum SVM Loan Risk Prediction API is running!"}

@app.post("/predict")
def predict_loan_risk(applicant: LoanApplicant):
    """
    Accepts structured loan applicant details and returns loan risk prediction.
    """
    try:
        # Convert input data into a NumPy array for prediction
        input_features = np.array([
            applicant.age,
            applicant.person_income,
            applicant.person_home_ownership,
            applicant.person_emp_length,
            applicant.loan_intent,
            applicant.loan_grade,
            applicant.loan_amnt,
            applicant.loan_int_rate,
            applicant.loan_percent_income,
            applicant.cb_person_default_on_file,
            applicant.cb_person_cred_hist_length
        ]).reshape(1, -1)

        # Create an imputer to handle missing values (strategy: mean)
        imputer = SimpleImputer(strategy="mean")
        
        # Impute missing values in the input features
        input_features_imputed = imputer.fit_transform(input_features)

        # Make prediction using the trained model
        prediction = model.predict(input_features_imputed)

        # Return result along with the original input features
        return {
            "input_data": {
                "age": applicant.age,
                "income": applicant.person_income,
                "home_ownership": applicant.person_home_ownership,
                "emp_length": applicant.person_emp_length,
                "loan_intent": applicant.loan_intent,
                "loan_grade": applicant.loan_grade,
                "loan_amnt": applicant.loan_amnt,
                "loan_int_rate": applicant.loan_int_rate,
                "loan_percent_income": applicant.loan_percent_income,
                "cb_person_default_on_file": applicant.cb_person_default_on_file,
                "cb_person_cred_hist_length": applicant.cb_person_cred_hist_length
            },
            "prediction": "High Risk" if prediction[0] == 1 else "Low Risk"
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# To run the FastAPI se
