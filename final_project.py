import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

s = pd.read_csv("social_media_usage.csv")

def clean_sm(x):
    return np.where(x == 1, 1, 0)

ss = s[['income', 'educ2', 'par', 'marital', 'gender', 'age']].copy()

ss['sm_li'] = clean_sm(s['web1h'])

ss.dropna(inplace=True)

y = ss['sm_li']
X = ss.drop('sm_li', axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

logreg_model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
logreg_model.fit(X_train, y_train)

st.markdown("### Machine Learning model for predicting likelihood of person using LinkedIn")

st.sidebar.header('Input Variables')
income = st.sidebar.slider('Income', min_value=1, max_value=9, value=8)
educ2 = st.sidebar.slider('Education', min_value=1, max_value=8, value=7)
par = st.sidebar.slider('Parents', min_value=1, max_value=2, value=2)
marital = st.sidebar.slider('Marital Status', min_value=1, max_value=6, value=1)
gender = st.sidebar.slider('Gender', min_value=1, max_value=3, value=2)
age = st.sidebar.slider('Age', min_value=18, max_value=100, value=42)

if st.sidebar.button('Predict'):
    person = np.array([income, educ2, par, marital, gender, age]).reshape(1, -1)
    probability = logreg_model.predict_proba(person)[:, 1]
    st.write("Probability for the input person:", probability[0])