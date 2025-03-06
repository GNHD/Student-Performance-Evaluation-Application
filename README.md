# 📊 Student Performance Evaluation Application

## 🚀 Overview
This project predicts student performance based on key academic metrics such as attendance, assignment scores, quiz scores, and study hours. The system leverages **machine learning** to provide insights for educators to support students effectively.

## 🛠️ Features
- **Machine Learning Model:** Trained using a **Random Forest Regressor**.
- **FastAPI Backend:** Handles CSV file uploads and returns performance predictions.
- **Streamlit UI:** Allows users to upload student data and visualize predictions.
- **Data Processing:** Handles missing values, scales input features, and standardizes data.
- **Graphical Representation:** Generates bar graphs to visualize predictions.

## 📂 Project Structure
```
📁 Student-Performance-Prediction
│── 📂 Backend
│   ├── api.py  # FastAPI server for predictions
│   ├── model.py  # ML model training & prediction logic
│   ├── student_performance_model.pkl  # Saved ML model
│── 📂 Frontend
│   ├── app.py  # Streamlit UI for user interaction
│── 📂 Data
│   ├── student_performance_data.csv  # Sample dataset
│── requirements.txt  # Required dependencies
│── README.md  # Project documentation
```

## 📥 Installation & Setup
### 1️⃣ Clone the Repository
```sh
git clone https://github.com/yourusername/Student-Performance-Prediction.git
cd Student-Performance-Prediction
```

### 2️⃣ Install Dependencies
Ensure you have Python installed, then run:
```sh
pip install -r requirements.txt
```

### 3️⃣ Run the Backend Server
```sh
cd Backend
uvicorn api:app --reload
```
The API will be available at `http://127.0.0.1:8000`.

### 4️⃣ Run the Streamlit UI
```sh
cd Frontend
streamlit run app.py
```

## 📌 API Usage
**Endpoint:** `POST /predict/`
- Accepts a CSV file with required columns.
- Returns predicted student performance scores.

## 🔮 Future Enhancements
- Improve the model with **deep learning** for higher accuracy.
- Implement **real-time prediction** based on student activity.
- Add **personalized recommendations** for students.

---

This project demonstrates the power of **data-driven decision-making in education**! 🎓✨
