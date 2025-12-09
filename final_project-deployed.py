import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import streamlit as st

def clean_sm(x):
    return np.where(x == 1, 1, 0)

s = pd.read_csv('social_media_usage.csv')

ss = pd.DataFrame({
    'sm_li': clean_sm(s['web1h']),
    'income' : np.where(s['income'] > 9, np.nan, s['income']),
    'education' : np.where(s['educ2'] > 9, np.nan, s['educ2']),
    'parent' : clean_sm(s['par']),
    'married' : clean_sm(s['marital']),
    'female' : np.where(s['gender'] == 2, 1, 0),
    'age' : np.where(s['age'] > 98, np.nan, s['age'])
})

ss= ss.dropna()


df_y = ss['sm_li']
df_x = ss.drop(columns=['sm_li'])

y_train, y_test = train_test_split(
    df_y,
    test_size=0.2,
    random_state=94587
)

x_train, x_test = train_test_split(
    df_x,
    test_size=0.2,
    random_state=94587
)

li_logit_2 = LogisticRegression(class_weight='balanced')

li_logit_2.fit(x_train, y_train)

##########################################################
# Input Setup
##########################################################

income_dict = {
    "Less than $10,000": 1,
    "10 to under $20,000": 2,
    "20 to under $30,000": 3,
    "30 to under $40,000": 4,
    "40 to under $50,000": 5,
    "50 to under $75,000": 6,
    "75 to under $100,000": 7,
    "100 to under $150,000": 8,
    "$150,000 or more": 9
}

education_dict = {
    "Less than high school": 1,
    "High school incomplete": 2,
    "High school graduate": 3,
    "Some college, no degree": 4,
    "Two-year associate degree": 5,
    "Four-year college or university degree": 6,
    "Some postgraduate or professional schooling, no degree": 7,
    "Postgraduate or professional degree": 8,
}

##########################################################
# Streamlit App
##########################################################

st.title("🔮 LinkedIn Usage Probability Predictor")
st.write(
    "Enter the demographic information below to estimate the probability "
    "that a person uses **LinkedIn** based on survey data and logistic regression."
)

st.divider()

# Use columns to organize inputs
col1, col2 = st.columns(2)

with col1:
    income_sel = st.selectbox("**Income Level**", list(income_dict.keys()))
    parent = st.selectbox("**Parent of 18+ child?**", ("No", "Yes"))
    gender = st.selectbox("**Gender**", ("Male", "Female"))

with col2:
    education_sel = st.selectbox("**Education Level**", list(education_dict.keys()))
    marriage = st.selectbox("**Married?**", ("No", "Yes"))
    age = st.number_input("**Age**", min_value=18, max_value=98, value=25)

# Build input row
input_pred = pd.DataFrame({
    'income': [income_dict[income_sel]],
    'education': [education_dict[education_sel]],
    'parent': [1 if parent == "Yes" else 0],
    'married': [1 if marriage == "Yes" else 0],
    'female': [1 if gender == "Female" else 0],
    'age': [age]
})

# Predict
y_prob = li_logit_2.predict_proba(input_pred)[0][1]

st.divider()

# -------------------------------------------------------
# Output
# -------------------------------------------------------
st.subheader("📊 Prediction Results")

colA, colB = st.columns(2)

with colA:
    st.metric("Probability of LinkedIn Usage", f"{y_prob:.1%}")

with colB:
    if y_prob >= 0.5:
        st.success("Likely **LinkedIn User** 👍")
    else:
        st.error("Unlikely to Use LinkedIn ❌")

st.write(
    "This prediction is based on demographic survey data used to train a "
    "logistic regression classifier."
)