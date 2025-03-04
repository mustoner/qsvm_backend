import streamlit as st
import joblib
import numpy as np
from sklearn.impute import SimpleImputer

# Load the trained QSVC model
model = joblib.load("Myqsvm_model.pkl")

# Streamlit App Interface
st.title("Loan Risk Prediction Using Quantum SVM")

# Create input fields for loan applicant data
age = st.number_input("Age", min_value=18, max_value=100, value=30)
person_income = st.number_input("Income", min_value=0.0, value=50000.0)
person_home_ownership = st.selectbox("Home Ownership", options=[0, 1, 2])  # 0: Rent, 1: Own, 2: Mortgage
person_emp_length = st.number_input("Employment Length (in years)", min_value=0, max_value=50, value=5)
loan_intent = st.selectbox("Loan Intent", options=[0, 1, 2, 3])  # Example intents
loan_grade = st.selectbox("Loan Grade", options=[1, 2, 3, 4, 5])  # Example grades
loan_amnt = st.number_input("Loan Amount", min_value=1000.0, value=10000.0)
loan_int_rate = st.number_input("Loan Interest Rate", min_value=0.0, max_value=100.0, value=5.0)
loan_percent_income = st.number_input("Loan Percent of Income", min_value=0.0, max_value=100.0, value=20.0)
cb_person_default_on_file = st.selectbox("Default on file", options=[0, 1])  # 0: No, 1: Yes
cb_person_cred_hist_length = st.number_input("Credit History Length (years)", min_value=0, max_value=50, value=10)

# Prepare data for prediction
input_data = np.array([
    age,
    person_income,
    person_home_ownership,
    person_emp_length,
    loan_intent,
    loan_grade,
    loan_amnt,
    loan_int_rate,
    loan_percent_income,
    cb_person_default_on_file,
    cb_person_cred_hist_length
]).reshape(1, -1)

# Impute missing values in the input data
imputer = SimpleImputer(strategy="mean")
input_data_imputed = imputer.fit_transform(input_data)

# Predict the loan risk
if st.button("Predict Loan Risk"):
    try:
        # Make the prediction
        prediction = model.predict(input_data_imputed)

        # Show the result
        result = "High Risk" if prediction[0] == 1 else "Low Risk"
        st.write(f"The loan is predicted to be: **{result}**")
    except Exception as e:
        st.write(f"Error: {e}")

