import streamlit as st
import numpy as np
import pickle
import pandas as pd

# Load models
clf = pickle.load(open("clf.pkl", "rb"))
reg = pickle.load(open("reg.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("🏥 Medical Cost & Disease Prediction App")

st.write("Enter patient details:")

# Inputs
age = st.number_input("Age", 1, 100)
gender = st.selectbox("Gender", ["Male", "Female"])
bmi = st.number_input("BMI", 10.0, 50.0)
children = st.number_input("Number of Children", 0, 10)

smoker = st.selectbox("Smoker", ["Yes", "No"])
physical_activity = st.selectbox("Physical Activity Level", ["Low", "Medium", "High"])
insurance = st.selectbox("Insurance Type", ["Basic", "Premium"])
city = st.selectbox("City Type", ["Urban", "Rural"])

diabetes = st.selectbox("Diabetes", [0, 1])
hypertension = st.selectbox("Hypertension", [0, 1])
heart_disease = st.selectbox("Heart Disease", [0, 1])
asthma = st.selectbox("Asthma", [0, 1])

# Convert categorical manually (same encoding as training)
gender = 1 if gender == "Male" else 0
smoker = 1 if smoker == "Yes" else 0

physical_map = {"Low": 0, "Medium": 1, "High": 2}
insurance_map = {"Basic": 0, "Premium": 1}
city_map = {"Urban": 1, "Rural": 0}

physical_activity = physical_map[physical_activity]
insurance = insurance_map[insurance]
city = city_map[city]

# Feature array
#features = np.array([[age, gender, bmi, children, smoker,
              #        physical_activity, insurance, city,
              #        diabetes, hypertension, heart_disease, asthma]])
columns = pickle.load(open("columns.pkl", "rb"))

input_dict = {
    "age": age,
    "gender": gender,
    "bmi": bmi,
    "children": children,
    "smoker": smoker,
    "physical_activity_level": physical_activity,
    "insurance_type": insurance,
    "city_type": city,
    "diabetes": diabetes,
    "hypertension": hypertension,
    "heart_disease": heart_disease,
    "asthma": asthma
}

features = pd.DataFrame([input_dict])
features = features.reindex(columns=columns, fill_value=0)

features = scaler.transform(features)
# Scale input
features = scaler.transform(features)

# Predict
if st.button("Predict"):
    disease_pred = clf.predict(features)[0]
    cost_pred = reg.predict(features)[0]

    st.subheader("Results:")

    if disease_pred == 1:
        st.error("⚠️ High risk of disease")
    else:
        st.success("✅ Low risk of disease")

    st.info(f"💰 Estimated Medical Cost: ₹ {round(cost_pred, 2)}")