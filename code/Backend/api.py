from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
import joblib
from io import StringIO
import os

app = FastAPI()

# Load trained model and scaler
model_path = os.path.join(os.path.dirname(__file__), "student_performance_model.pkl")

try:
    model, scaler = joblib.load(model_path)
except FileNotFoundError:
    raise RuntimeError("Model file not found! Ensure 'student_performance_model.pkl' is in the Backend folder.")

# Define required columns
required_columns = ['Attendance_Percentage', 'Assignment_Scores', 'Quiz_Scores', 'Final_Exam_Score', 'Study_Hours']

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode("utf-8")))

        # Normalize column names to avoid case sensitivity issues
        df.columns = df.columns.str.strip().str.replace(" ", "_")

        # Check if all required columns exist
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise HTTPException(status_code=400, detail=f"Missing required columns: {missing_cols}")

        # Ensure only numeric values & fill NaNs
        X = df[required_columns].apply(pd.to_numeric, errors='coerce').fillna(df[required_columns].mean())

        # Scale the input data
        X_scaled = scaler.transform(X)

        # Make predictions
        predictions = model.predict(X_scaled)

        # Attach predictions to the dataframe
        df["Predicted_Performance"] = predictions

        return {"predictions": df[["Student_ID", "Predicted_Performance"]].to_dict(orient="records")}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


