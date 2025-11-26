# app.py
# Final single-file Streamlit app that mirrors your Colab ML environment,
# auto-installs missing packages, defines SMOTEWithColumns, loads your pipeline,
# and provides the UI (gauge, chatbot, nav to reduce-risk page).


# -----------------------------------------------------------------
import pandas as pd
from imblearn.over_sampling import SMOTE

class SMOTEWithColumns(SMOTE):
    def fit_resample(self, X, y):
        X_res, y_res = super().fit_resample(X, y)
        if isinstance(X, pd.DataFrame):
            X_res = pd.DataFrame(X_res, columns=X.columns)
        return X_res, y_res
# ---------------- ACTUAL IMPORTS (mirror your Colab notebook) ----------------
from sklearn.experimental import enable_iterative_imputer  # using iterative imputer for regression imputing

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, f1_score, classification_report, confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

import xgboost as xgb
import lightgbm as lgb
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, ADASYN
from imblearn.pipeline import Pipeline as ImbPipeline

import shap
import joblib
import os

# catboost import (after pip install)
from catboost import CatBoostClassifier

# -----------------------------------------------------------------------------

# ---------------- SMOTEWithColumns CLASS (necessary) ----------------


# ----------------------------------------------------------------------

# ---------------- Streamlit UI + Model Loading ----------------
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Stroke Prediction App", page_icon="🧠", layout="centered")

# Chatbot in sidebar
with st.sidebar:
    st.title("💬 Chatbot Assistant")
    st.write("Ask me anything about stroke risk")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    user_msg = st.text_input("Your message")

    if st.button("Send"):
        if user_msg:
            st.session_state["messages"].append(("user", user_msg))
            # very simple rule-based replies (local)
            if "reduce" in user_msg.lower() or "how" in user_msg.lower():
                reply = "You can reduce stroke risk via healthy diet, regular exercise, controlling blood pressure, and quitting smoking."
            elif "hi" in user_msg.lower() or "hello" in user_msg.lower():
                reply = "Hi — tell me what you'd like to know about stroke risk."
            else:
                reply = "I can explain risk factors, prevention tips, or the meaning of your result."
            st.session_state["messages"].append(("bot", reply))

    for sender, msg in st.session_state["messages"]:
        if sender == "user":
            st.write(f"🧑🏻‍💬 **You:** {msg}")
        else:
            st.write(f"🤖 **Bot:** {msg}")

# Load your trained model (pipeline saved as stroke_prediction_calibrated.pkl)
MODEL_PATH = "stroke_prediction_calibrated.pkl"
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at {MODEL_PATH}. Put stroke_prediction_calibrated.pkl in the app folder.")
    st.stop()

model = joblib.load(MODEL_PATH)

st.title("🧠 Stroke Risk Prediction")
st.write("Fill out the details below to estimate your stroke risk. (This app expects the saved preprocessing + model pipeline)")

with st.form("stroke_form"):
    age = st.number_input("Age", 1, 120, 50)
    avg_glucose_level = st.number_input("Average Glucose Level", 50.0, 300.0, 100.0)
    bmi = st.number_input("BMI", 10.0, 60.0, 25.0)

    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    ever_married = st.selectbox("Ever Married", ["Yes", "No"])
    hypertension = st.selectbox("Hypertension", [0, 1])
    heart_disease = st.selectbox("Heart Disease", [0, 1])
    work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children"])
    residence = st.selectbox("Residence Type", ["Urban", "Rural"])
    smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])

    submit = st.form_submit_button("Predict Stroke Risk")

if submit:
    # prepare DataFrame exactly like your training pipeline expects
    example = pd.DataFrame([{
        'age': age,
        'avg_glucose_level': avg_glucose_level,
        'bmi': bmi,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'gender': gender,
        'ever_married': ever_married,
        'work_type': work_type,
        'Residence_type': residence,
        'smoking_status': smoking_status
    }])

    # Prediction (pipeline should include preprocessing)
    try:
        pred = model.predict(example)
        pred_proba = model.predict_proba(example)[0][1]
    except Exception as e:
        st.error("Error while running the model. Likely mismatch between saved pipeline and input schema.")
        st.exception(e)
        st.stop()

    risk_pct = round(pred_proba * 100, 2)
    st.subheader("📊 Your Stroke Risk")
    st.write(f"**Estimated Probability: {risk_pct}%**")

    # Animated gauge (plotly)
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_pct,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Stroke Risk (%)"},
        delta={'reference': 10, 'increasing': {'color': "red"}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 10], 'color': "lightgreen"},
                {'range': [10, 25], 'color': "yellow"},
                {'range': [25, 100], 'color': "red"},
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.8,
                'value': risk_pct
            }
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

    if risk_pct < 10:
        st.success("Low Risk — keep healthy habits.")
    elif risk_pct < 25:
        st.warning("Moderate Risk — monitor and consult if needed.")
    else:
        st.error("High Risk — please consult a healthcare professional.")
# Add button
notion_url = "https://dandy-onyx-fa3.notion.site/How-to-Reduce-Stroke-Risk-2b78283a207c8037a169f599befba49e"

st.markdown(
    f"""
    <div style="text-align:center; margin-top:25px;">
        <a href="{notion_url}" target="_blank">
            <button style="
                background-color:#4A90E2;
                color:white;
                padding:14px 28px;
                border:none;
                border-radius:8px;
                font-size:18px;
                cursor:pointer;
            ">
                Learn How to Reduce Stroke Risk ➜
            </button>
        </a>
    </div>
    """,
    unsafe_allow_html=True
)


