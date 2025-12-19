import streamlit as st
import joblib
import numpy as np

model = joblib.load("model.joblib")

st.title("Anomaly Detection using Local Outlier Factor")
st.write("Enter the value below:")

hours = st.number_input("Hours studied")

if st.button("Check"):
    X = np.array([[hours]]) 

    pred = model.predict(X)[0]
    score = model.decision_function(X)[0]

    flag = "Anomaly" if pred == -1 else "Safe"
    st.write(f"Result: {flag}")
    st.write(f"Score: {score}")
