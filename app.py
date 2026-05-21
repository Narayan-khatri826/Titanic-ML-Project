# Fir uske andar ye code paste karo 👇

import streamlit as st

import pickle

import numpy as np

model = pickle.load(open("model.pkl", "rb"))

st.title("Titanic Survival Prediction")

passengerid = st.number_input("Passenger ID")

pclass = st.number_input("Passenger Class")

sex = st.number_input("Sex (0=Male, 1=Female)")

age = st.number_input("Age")

sibsp = st.number_input("Siblings/Spouse")

parch = st.number_input("Parents/Children")

fare = st.number_input("Fare")

embarked = st.number_input("Embarked (0=C, 1=Q, 2=S)")



if st.button("Predict"):
    data = np.array([[passengerid, pclass, sex, age, sibsp, parch, fare, embarked]], dtype=float)
    st.write(data.shape)
    prediction = model.predict(data)
    if prediction[0] == 1:
        st.success("Passenger Survived")
    else:
        st.error("Passenger Did Not Survive")

