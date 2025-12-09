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

st.title("Linkedin Probability Prediction")
income_sel = st.selectbox("Select Income Level (1-9):", (
    "Less than $10,000",   
    "10 to under $20,000",
    "20 to under $30,000",
    "30 to under $40,000",
    "40 to under $50,000",
    "50 to under $75,000",
    "75 to under $100,000",
    "100 to under $150,000",
    "$150,000 or more")
    )

income = income_dict[income_sel]

education_sel = st.selectbox("Select Education Level:",
    ("Less than high school",
    "High school incomplete",
    "High school graduate",
    "Some college, no degree",
    "Two-year associate degree",
    "Four-year college or university degree"
))

education = education_dict[education_sel]

parent = st.selectbox("Are you a parent of an 18+ child?", ("No", "Yes"))
marriage = st.selectbox("Are you married?", ("No", "Yes"))
gender = st.selectbox("Select Gender:", ("Male", "Female"))
age = st.number_input("Age of user:", min_value=18, max_value=98, value=25)

input_pred = pd.DataFrame({
    'income': [income],
    'education': [education],
    'parent': [1 if parent == "Yes" else 0],
    'married': [1 if marriage == "Yes" else 0],
    'female': [1 if gender == "Female" else 0],
    'age': [age]
})

y_prob = li_logit_2.predict_proba(input_pred)

st.subheader("Predicted Probability of Linkedin Usage")
st.write(f"Probability of Linkedin Usage: {y_prob[0][1]:.1%}")

if y_prob[0][1] >= 0.5:
    st.write("This user is likely to use Linkedin.")
else:  
    st.write("This user is unlikely to use Linkedin.")

