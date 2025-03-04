import pandas as pd
import numpy as np
from fastapi import FastAPI, File, UploadFile
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import io
import uvicorn
from sklearn.metrics import classification_report
import joblib
from tensorflow.keras.models import load_model

# Load the Keras model
model = load_model(r'C:\Users\HP\Desktop\Gaint\credit\path_to_your_model')  # Update with the path to your trained model





# Initialize FastAPI app
app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the uploaded CSV file
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode("utf-8")))

        # Step 2: Separate Features and Targets
        X = df[['Feature_1', 'Feature_2', 'Feature_3']].values
        y = df[['Age_mismatch', 'Medicine_substitution', 'Chronic_visit', 'Over_prescription', 'Inactive_member']].values

        # Normalize Features
        X_scaled = scaler.fit_transform(X)  # Fit and transform the features

        # Step 3: Split the Data (This is only for training, but since we're predicting, we don't need to split)
        # In the case of prediction, we do not split the data.
        # X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

        # Step 4: Define the Model (For prediction, use the pre-trained model, no need to define again)
        # model = Sequential([...])  # You would have this already saved in your model file.

        # Step 5: Predict
        predictions = model.predict(X_scaled)
        predictions_binary = (predictions > 0.5).astype(int)  # Convert probabilities to binary predictions

        # Add predictions to the DataFrame
        prediction_columns = [
            "Age_mismatch", "Medicine_substitution", "Chronic_visit", "Over_prescription", "Inactive_member"
        ]
        for i, col in enumerate(prediction_columns):
            df[col] = predictions_binary[:, i]

        # Return the predictions in JSON format
        return df.to_dict(orient="records")
    
    except Exception as e:
        return {"error": str(e)}

# Run the app (use this line if running the script directly)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
