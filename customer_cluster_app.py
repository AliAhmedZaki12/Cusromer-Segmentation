import streamlit as st
import pandas as pd
import joblib

# Load the trained model and scaler
model = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")
column = joblib.load("column.pkl")

# Define expected column order
column = ['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']

# Cluster descriptions
descriptions = {
        0: "🟢 Low income, low spending customers.",
        1: "🔴 High income, high spending (VIP customers).",
        2: "🟡 Moderate income and spending.",
        3: "🔵 High income, low spending.",
        4: "🟣 Low income but high spending customers."
}

# Streamlit UI
st.title("Customer Segmentation Prediction")

# Input fields
gender_input = st.radio("Gender", ["Male", "Female"])
age_input = st.number_input("Age", min_value=10, max_value=100, value=30)
income_input = st.number_input("Annual Income (in k$)", min_value=0, max_value=200, value=50)
score_input = st.slider("Spending Score (1-100)", 1, 100, 50)

# Encode gender
gender_encoded = 1 if gender_input == "Male" else 0

# Create customer DataFrame
new_customer = pd.DataFrame([{
    'Gender': gender_encoded,
    'Age': age_input,
    'Annual Income (k$)': income_input,
    'Spending Score (1-100)': score_input
}])

# Reorder columns
new_customer = new_customer[column]

# Scale the data
scaled_customer = scaler.transform(new_customer)

# Predict cluster
if st.button("Predict Cluster"):
    cluster = model.predict(scaled_customer)[0]
    description = descriptions.get(cluster, "No description available for this cluster.")

    st.write(f"The customer belongs to cluster number: {cluster}")
    st.info(f"Cluster Description: {description}")

