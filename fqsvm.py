from flask import Flask, request, jsonify
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
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import learning_curve

from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
#from sklearn.pipeline import make_pipeline
#from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_val_score
#from sklearn.impute import SimpleImputer
from scipy.stats import norm
from scipy import stats

#qiskit
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
from qiskit.providers import basic_provider
from qiskit.providers import basic_provider
import warnings
warnings.filterwarnings('ignore')


# Load trained model and scaler once at startup
qsvc_model = joblib.load("qsvm_model.pkl")
scaler = joblib.load("scaler.pkl")

# Initialize Flask app
app = Flask(__name__)

# Define a route to predict loan outcome
@app.route("/predict", methods=["POST"])
def predict_loan_outcome():
    try:
        # Get JSON data from the request
        data = request.get_json()

        # Parse the input data (similar to the FastAPI LoanRequest model)
        loan_data = np.array([[data["age"], data["income"], data["home"], data["emp_length"], 
        data["intent"], data["grade"], data["loan_amnt"], data["loan_int_rate"], 
        data["loan_percent_income"], data["default"], data["cb_person_cred_hist_length"]]])

        # Scale the input data using the scaler
        customer_data_scaled = scaler.transform(loan_data)
        
        # Predict the loan outcome using the trained model
        prediction = qsvc_model.predict(customer_data_scaled)
        
        # Return the prediction (0 for default, 1 for non-default) as a JSON response
        return jsonify({"prediction": int(prediction[0])})

    except Exception as e:
        # Handle any errors that occur during prediction
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    # Run the Flask app
    app.run(debug=True)
