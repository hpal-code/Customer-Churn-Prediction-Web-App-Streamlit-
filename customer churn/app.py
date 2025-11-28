# ============================================================
# 💡 CUSTOMER CHURN PREDICTION WEB APP (STREAMLIT)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

# ============================================================
# 📥 LOAD DATA
# ============================================================

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

@st.cache_data
def load_data():
    data = pd.read_csv(r"C:\Users\DELL\Downloads\dataset.zip")
    data.drop('customerID', axis=1, inplace=True)
    # Fix invalid numeric values
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    data['TotalCharges'].fillna(data['TotalCharges'].mean(), inplace=True)

    # Encode churn YES/NO
    data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})
    return data


st.title("💡 Customer Churn Prediction Dashboard")
st.markdown("### Predict and analyze customer churn using Machine Learning")

st.sidebar.header("⚙️ Configuration Panel")


# ============================================================
# 📊 LOAD & DISPLAY DATA
# ============================================================

data = load_data()

if st.sidebar.checkbox("Show Raw Data"):
    st.subheader("📄 Raw Dataset")
    st.dataframe(data.head())


# ============================================================
# 📈 EXPLORATORY DATA ANALYSIS
# ============================================================

st.subheader("📊 Data Insights")

col1, col2 = st.columns(2)

with col1:
    churn_rate = data['Churn'].mean() * 100
    st.metric("Overall Churn Rate", f"{churn_rate:.2f}%")

with col2:
    st.metric("Total Customers", len(data))


# 🔹 Churn Count
st.markdown("#### Churn Distribution")
fig, ax = plt.subplots()
sns.countplot(x='Churn', data=data, palette='Set2', ax=ax)
ax.set_title("Churn Count")
st.pyplot(fig)

# 🔹 Churn vs Contract Type
st.markdown("#### Churn By Contract Type")
fig, ax = plt.subplots()
sns.countplot(x='Contract', hue='Churn', data=data, palette='coolwarm', ax=ax)
plt.xticks(rotation=20)
st.pyplot(fig)


# ============================================================
# 🧠 DATA PREPROCESSING
# ============================================================

st.subheader("⚙️ Model Training")

cat_cols = data.select_dtypes(include=['object']).columns

# Encode categorical columns
for col in cat_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])

# Feature / Label split
X = data.drop('Churn', axis=1)
y = data['Churn']

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train–Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)


# ============================================================
# ⚙️ MODEL SELECTION
# ============================================================

model_option = st.sidebar.radio("Select Model", ("Random Forest", "Logistic Regression"))

if model_option == "Random Forest":
    model = RandomForestClassifier(n_estimators=200, random_state=42)
else:
    model = LogisticRegression(max_iter=500)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)

st.success(f"✅ Model Trained Successfully → {model_option}")
st.write(f"### 📌 Accuracy: **{acc*100:.2f}%**")

# Classification Report
st.text("📄 Classification Report:")
st.text(classification_report(y_test, y_pred))


# Confusion Matrix
st.markdown("#### Confusion Matrix")
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
st.pyplot(fig)


# ============================================================
# 🔮 CUSTOM INPUT PREDICTION
# ============================================================

st.subheader("🔮 Try Custom Prediction (Your Own Inputs)")

sample_input = {}

for col in X.columns:
    if np.issubdtype(data[col].dtype, np.number):
        sample_input[col] = st.number_input(f"{col}", value=float(data[col].mean()))
    else:
        sample_input[col] = st.selectbox(f"{col}", options=sorted(data[col].unique()))

if st.button("🔍 Predict Customer Churn"):
    input_df = pd.DataFrame([sample_input])

    # Encode categorical values
    for col in input_df.columns:
        if col in cat_cols:
            le = LabelEncoder()
            input_df[col] = le.fit_transform(input_df[col])

    input_df = input_df.reindex(columns=X.columns, fill_value=0)
    input_scaled = scaler.transform(input_df)
    pred = model.predict(input_scaled)[0]

    if pred == 1:
        st.error("🚨 Customer is likely to CHURN!")
    else:
        st.success("💚 Customer is likely to STAY!")


# ============================================================
# 💾 SAVE MODEL
# ============================================================

if st.sidebar.button("💾 Save Model"):
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/churn_model.pkl")
    joblib.dump(scaler, "model/scaler.pkl")
    st.sidebar.success("Model Saved Successfully! ✔")


# ============================================================
# 📎 FOOTER
# ============================================================

st.sidebar.markdown("---")
st.sidebar.markdown("📂 **Dataset:** Telco Customer Churn (Kaggle)")
st.sidebar.markdown("Developed with ❤️ using Streamlit & Scikit-Learn")
