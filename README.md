Customer Churn Prediction Web App (Streamlit)

A machine learning web application that predicts whether a customer is likely to churn (leave the service).
The app provides real-time predictions, model training, data visualization, and interactive dashboards.

🚀 Tech Stack

Python

Pandas, NumPy

Scikit-Learn

Matplotlib, Seaborn

Streamlit

Joblib

📌 Project Overview

This Customer Churn Prediction System analyzes customer behavior and predicts the probability of churn using various ML models such as Random Forest and Logistic Regression.

The app allows:

Real-time prediction

Data visualization

Custom input-based prediction

Model training

Saving model for future use

🧠 Features

✔ Interactive Streamlit dashboard
✔ Clean UI with side panels
✔ EDA (Distribution plots, churn rate, contract analysis)
✔ Model comparison (Random Forest vs Logistic Regression)
✔ Real-time prediction for custom user input
✔ Confusion matrix + classification report
✔ Auto-save trained model (pkl format)
✔ Uses Telco Customer Churn dataset

📂 Project Folder Structure
customer-churn/
│── app.py
│── requirements.txt
│── data/
│     └── dataset.csv
│── model/
│     ├── churn_model.pkl
│     └── scaler.pkl
│── images/
│     ├── dashboard.png
│     ├── prediction.png
│── README.md



2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Ensure dataset exists

Place dataset.csv here:

customer-churn/data/telco.csv

4️⃣ Run the Streamlit app
streamlit run app.py

📈 Model Details
🔹 Models used:

Random Forest Classifier

Logistic Regression

🔹 Evaluation Metrics

Accuracy

Classification Report

Confusion Matrix

🔮 How Prediction Works

The model predicts churn based on:

Monthly charges

Contract type

Payment method

Internet service

Tenure

Online security/services

Support usage

And many more customer features

📌 Dataset Information

Dataset used: Telco Customer Churn (Kaggle)
Link: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

✨ Future Improvements

Add XGBoost model

Add feature importance plots



Add login authentication


✍️ Author

Your Name
AI/ML Developer | Python Developer
📧 email - hpcrc2005@gmail.com

