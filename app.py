# Fir uske andar ye code paste karo 👇

import streamlit as st

import pickle

import numpy as np

model = pickle.load(open("model.pkl", "rb"))

st.title("Titanic Survival Prediction")

pclass = st.number_input("Passenger Class")

sex = st.number_input("Sex (0=Male, 1=Female)")

age = st.number_input("Age")

sibsp = st.number_input("Siblings/Spouse")

parch = st.number_input("Parents/Children")

fare = st.number_input("Fare")

if st.button("Predict"):
    data = np.array([[pclass, sex, age, sibsp, parch, fare]])

prediction = model.predict(data)

if prediction[0] == 1:

    st.success("Passenger Survived")

else:
      st.error("Passenger Did Not Survive")

