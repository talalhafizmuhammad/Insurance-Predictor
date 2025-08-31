import streamlit as st  # type: ignore 
import pickle

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("Insurance Charges Predictor")

age = st.number_input("Enter Age", min_value = 0, max_value = 120, value = 30)
bmi = st.number_input("Enter BMI", min_value = 10.0, max_value = 50.0, value = 25.0)
children = st.number_input("Enter the Number of Children", min_value = 0, max_value = 10, value = 0)

smoker_choice = st.selectbox("Is the person Smoker?", options=['yes', 'no'], index = 1)
is_smoker = 1 if smoker_choice == 'yes' else 0

if st.button("Predict Charges"):
    features_to_scale = [[bmi, age, children]]  
    scaled = scaler.transform(features_to_scale)[0]

    features_final = [[scaled[1], scaled[0], scaled[2], is_smoker]]

    prediction = model.predict(features_final)

    st.success(f"Predicted Insurance Charges: ${prediction[0]:.0f}")
