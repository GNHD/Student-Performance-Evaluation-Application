import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    required_columns = ['Attendance_Percentage', 'Assignment_Scores', 'Quiz_Scores', 'Final_Exam_Score', 'Study_Hours']
    
    # Ensure we are working on a copy of the DataFrame
    df = df.copy()

    # Convert required columns to numeric (force non-numeric values to NaN)
    for col in required_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Debugging: Print column types to confirm
    print("\n📊 Column Data Types Before Processing:")
    print(df.dtypes)

    # Fill missing values with column means (ignore non-numeric columns)
    df[required_columns] = df[required_columns].fillna(df[required_columns].mean())

    # Normalize numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[required_columns])

    return X_scaled, df['Final_Performance'], scaler




def train_model():
    data = pd.read_csv("Data/student_performance_data.csv")

    # Debugging: Print data types to check for issues
    print("📊 Checking column types before processing:")
    print(data.dtypes)

    X, y, scaler = preprocess_data(data)

    # Ensure 'Final_Performance' is numeric
    y = pd.to_numeric(y, errors='coerce').fillna(0)  # Replace NaN with 0

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump((model, scaler), "student_performance_model.pkl")

    print("✅ Model Training Complete!")

def predict_performance(input_df):
    required_columns = ['Attendance_Percentage', 'Assignment_Scores', 'Quiz_Scores', 'Final_Exam_Score', 'Study_Hours']

    # Ensure a copy of the DataFrame
    input_df = input_df.copy()

    # Drop non-numeric columns (like Student_ID) before processing
    if 'Student_ID' in input_df.columns:
        input_df = input_df.drop(columns=['Student_ID'])

    # Convert to numeric, forcing non-numeric values to NaN
    for col in required_columns:
        input_df[col] = pd.to_numeric(input_df[col], errors='coerce')

    # Fill missing values with column means
    input_df.fillna(input_df.mean(), inplace=True)

    # Load trained model & scaler
    model, scaler = joblib.load("Backend/student_performance_model.pkl")

    # Transform the input data using the same scaler
    X_scaled = scaler.transform(input_df[required_columns])

    # Predict performance (returns NumPy array)
    predictions = model.predict(X_scaled)

    return predictions  # Ensure this is a NumPy array

if __name__ == "__main__":
    train_model()
