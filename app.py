import streamlit as st

import pickle

import numpy as np

import pandas as pd

model = pickle.load(open("model.pkl", "rb"))

st.title("Titanic Survival Prediction")

passengerid = st.number_input("Passenger ID")

pclass = st.number_input("Passenger Class")
sex = st.selectbox(
"Select Gender",
["Male", "Female"]
)
if sex == "Male":
    sex=0
else:
    sex=1
age = st.number_input("Age")

sibsp = st.number_input("Siblings/Spouse")

parch = st.number_input("Parents/Children")

fare = st.number_input("Fare")

embarked = st.number_input("Embarked (0=C, 1=Q, 2=S)")

if st.button("Predict"):
    data = np.array([[passengerid, pclass, sex, age, sibsp, parch, fare, embarked]], dtype=float)
    prediction = model.predict(data)
    log_data = pd.DataFrame({
    "PassengerId": [passengerid],
    "Pclass": [pclass],
    "Sex": [sex],
    "Age": [age],
    "SibSp": [sibsp],
    "Parch": [parch],
    "Fare": [fare],
    "Embarked": [embarked],
    "Prediction": [prediction[0]]
})
    log_data.to_csv(
        "user_logs.csv",
        mode="a",
        header=False,
        index=False
    )

    if prediction[0] == 1:
    
        st.success("Passenger Survived")
    
    else:
    
        st.error("Passenger Did Not Survive")
            
    
st.subheader("User Logs")
try:
    logs = pd.read_csv("user_logs.csv")

    st.dataframe(logs)
except:
    st.write("No logs available")
    
    
    
    