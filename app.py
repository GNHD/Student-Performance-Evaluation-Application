import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import numpy as np  # Import numpy for spacing

API_URL = "http://127.0.0.1:8000/predict/"

st.title("📊 Student Performance Dashboard")

st.write("### Upload Student Data (CSV)")

with st.form("upload_form"):
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"], key="fileUploader")
    submit_button = st.form_submit_button("📈 Predict Performance")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        required_columns = {"Student_ID", "Attendance_Percentage", "Assignment_Scores", "Quiz_Scores", "Study_Hours", "Final_Exam_Score"}

        if not required_columns.issubset(df.columns):
            st.error(f"❌ The uploaded CSV is missing required columns. Expected: {required_columns}")
        else:
            st.write("### Preview of Uploaded Data:")
            st.dataframe(df)

            if submit_button:
                with st.spinner("🔍 Analyzing Data..."):
                    try:
                        response = requests.post(API_URL, files={"file": uploaded_file.getvalue()})

                        if response.status_code == 200:
                            response_data = response.json()

                            if "predictions" in response_data:
                                predictions_df = pd.DataFrame(response_data["predictions"])
                                st.success("✅ Predictions Generated!")
                                st.write("### Predicted Results:")
                                st.dataframe(predictions_df)

                                # Visualization Fix: Adjust X-axis Labels and Add Spacing
                                st.write("### Performance Trend")
                                fig, ax = plt.subplots(figsize=(24, 12))

                                student_ids = predictions_df["Student_ID"]
                                performance_scores = predictions_df["Predicted_Performance"]

                                x_positions = np.arange(len(student_ids))*2  # Use numbers instead of text for better spacing

                                ax.bar(x_positions, performance_scores, color="blue")
                                ax.set_xlabel("Student ID")
                                ax.set_ylabel("Predicted Performance Score")
                                ax.set_title("Predicted Student Performance")

                                # Rotate labels and adjust spacing
                                ax.set_xticks(x_positions)
                                ax.set_xticklabels(student_ids, rotation=90, ha="right", fontsize="10")

                                st.pyplot(fig)

                            else:
                                st.error("⚠️ No valid predictions received from API.")

                        else:
                            st.error(f"❌ API Error: {response.status_code}")

                    except requests.exceptions.RequestException as e:
                        st.error(f"❌ Could not connect to the backend: {e}")

    except Exception as e:
        st.error(f"⚠️ Error reading the file: {e}")



