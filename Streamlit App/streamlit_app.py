
import streamlit as st
import pandas as pd
import joblib

model = joblib.load("titanic_logreg_pipeline.pkl")

st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")

st.title("🚢 Titanic Survival Prediction")
st.write("Professional Logistic Regression Deployment")

pclass = st.selectbox("Passenger Class", [1,2,3])
sex = st.selectbox("Sex", ["male","female"])
age = st.slider("Age",0,100,25)
sibsp = st.number_input("Siblings/Spouses",0,10,0)
parch = st.number_input("Parents/Children",0,10,0)
fare = st.number_input("Fare",0.0,600.0,32.0)
embarked = st.selectbox("Embarked",["S","C","Q"])

df = pd.DataFrame({
    "Pclass":[pclass],
    "Sex":[sex],
    "Age":[age],
    "SibSp":[sibsp],
    "Parch":[parch],
    "Fare":[fare],
    "Embarked":[embarked]
})

if st.button("Predict Survival"):
    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]

    if pred==1:
        st.success(f"Survives (Probability: {prob:.2f})")
    else:
        st.error(f"Does Not Survive (Probability: {prob:.2f})")

uploaded = st.file_uploader("Batch CSV Prediction (optional)")

if uploaded:
    data = pd.read_csv(uploaded)
    preds = model.predict(data)
    probs = model.predict_proba(data)[:,1]
    data["Prediction"] = preds
    data["Probability"] = probs
    st.write(data)
