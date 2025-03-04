from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
#sklearn
from sklearn.svm import SVC 
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import learning_curve
from qiskit import *
from qiskit.circuit import Parameter
from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap, TwoLocal, RealAmplitudes
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit_machine_learning.algorithms import QSVC, NeuralNetworkClassifier 
#from qiskit.aqua.algorithms import VQC
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit.algorithms.optimizers import SPSA, L_BFGS_B, COBYLA
from qiskit_machine_learning.neural_networks import CircuitQNN
from qiskit_machine_learning.exceptions import QiskitMachineLearningError

from typing import Union
from qiskit import BasicAer

import warnings
warnings.filterwarnings('ignore')




# Load trained model and scaler once at startup
qsvc_model = joblib.load("qsvm_model.pkl")
scaler = joblib.load("scaler.pkl")

# Initialize FastAPI app
app = FastAPI()

# Define request body model
class LoanRequest(BaseModel):
    age: float
    income: float
    home: int  # 1=Owned, 0=Rented
    emp_length: float
    intent: int  # 1=Business, 0=Personal
    grade: int  # Credit Grade (A=1, B=2, C=3, etc.)0242167862 NAM1
    loan_amnt: float
    loan_int_rate: float
    loan_percent_income: float
    default: int  # 1=Yes, 0=No
    cb_person_cred_hist_length: float

# Define request body model
@app.post("/predict")
def predict_loan_outcome(request: LoanRequest):
    # Convert input into a NumPy array
    loan_data = np.array([[request.age, request.income, request.home, request.emp_length, request.intent, request.grade, 
    request.loan_amnt, request.loan_int_rate, request.loan_percent_income, request.default, 
    request.cb_person_cred_hist_length]])
    
    # Scale the input data using the scaler
    customer_data_scaled = scaler.transform(loan_data)
    
    # Predict the loan outcome using the trained model
    prediction = qsvc_model.predict(customer_data_scaled)
    
    # Return the prediction (0 for default, 1 for non-default)
    return {"prediction": int(prediction[0])}