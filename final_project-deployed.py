import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import streamlit as st
import plotly.graph_objects as go

def clean_sm(x):
    return np.where(x == 1, 1, 0)

def reverse_lookup(d, value):
    for k, v in d.items():
        if v == value:
            return k
    return None


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


# -------------------------------------------------------
# Compute demographic averages for LinkedIn users
# -------------------------------------------------------
li_users = ss[ss["sm_li"] == 1]

radar_values = {
    "Income": li_users["income"].mean() / 9, # scale income 0–1
    "Education": li_users["education"].mean() / 9, # scale education 0–1
    "Parent": li_users["parent"].mean().astype(int),
    "Married": li_users["married"].mean().astype(int),
    "Female": li_users["female"].mean().astype(int),
    "Age": li_users["age"].mean() / 100   # scale age 0–1
}

income_int = int(round(radar_values["Income"] * 9))
education_int = int(round(radar_values["Education"] * 9))
parent_bool = radar_values["Parent"] == 1
married_bool = radar_values["Married"] == 1
female_bool = radar_values["Female"] == 1

radar_labels = [
    f"Income: {reverse_lookup(income_dict, income_int)}",
    f"Education: {reverse_lookup(education_dict, education_int)}",
    f"Parent: {"Yes" if parent_bool else "No"}",
    f"Married: {"Yes" if married_bool else "No"}",
    f"Gender: {"Female" if female_bool else "Male"}"
]

labels = radar_labels
values = list(radar_values.values())

# Close the loop (required for radar charts)
values_closed = values + values[:1]
labels_closed = labels + labels[:1]

# -------------------------------------------------------
# Build interactive Plotly radar chart
# -------------------------------------------------------
fig = go.Figure()

fig.add_trace(go.Scatterpolar(
    r=values_closed,
    theta=labels_closed,
    fill='toself',
    name="LinkedIn User Profile",
    line=dict(color='royalblue'),
    fillcolor='rgba(65, 105, 225, 0.3)'
))

fig.update_layout(
    polar=dict(
        radialaxis=dict(visible=True, range=[0, 1])
    ),
    showlegend=False,
    height=450,
    margin=dict(l=40, r=40, t=40, b=40)
)



##########################################################
# Streamlit App
##########################################################

st.title("🔮 LinkedIn Usage Probability Predictor")
st.write(
    "Choose user demographics to predict the probability "
    "that a person uses **LinkedIn**."
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

# -------------------------------------------------------
# Display chart
# -------------------------------------------------------

st.subheader("📈 Typical LinkedIn User Radar Profile")
st.plotly_chart(fig, use_container_width=True)