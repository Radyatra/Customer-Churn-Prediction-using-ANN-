import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np
import pickle

preprocess = pickle.load(open('proc_pipe_ml1_p2.pkl', 'rb'))
model=tf.keras.models.load_model('ml1_p2_model.h5')

st.title('Telco Customer Churn Classifier')

st.header('If you love them, Keep Them!')
st.header ('By Radyatara Group (PERSERO)') 
st.image ('https://d35fo82fjcw0y8.cloudfront.net/2017/09/26225705/header%402x.png')
st.text ('Find out whether your precious customers will churn or not.')

st.header('Input Customer Parameters Below')

seniors = st.selectbox('Is customer a senior?', ['No', 'Yes'])
if seniors == 'Yes':
   seniors=1
else:
    seniors=0

partner = st.selectbox('Does customer have a partner?', ['Yes', 'No'])
dependents = st.selectbox('Is customer financially dependent?', ['Yes', 'No'])
internet = st.selectbox('Does customer subscribe to internet service?', ['DSL', 'Fiber optic', 'No internet service'])
sec_ol = st.selectbox('Does customer subscribe to online security services?', ['Yes', 'No', 'No internet service'])
backup_ol = st.selectbox('Does customer subscribe to online backup services?', ['Yes', 'No', 'No internet service'])
dev_protect = st.selectbox('Does customer subscribe to device protection services?', ['Yes', 'No', 'No internet service'])
tech_support = st.selectbox('Does customer subscribe to tech support?', ['Yes', 'No', 'No internet service'])
tv_stream = st.selectbox('Does customer subscribe to TV Streaming?', ['Yes', 'No', 'No internet service'])
movie_stream = st.selectbox('Does customer subscribe to Movie Streaming?', ['Yes', 'No', 'No internet service'])
contract = st.selectbox('What type of contract is the customer bind to?', ['Month-to-month', 'One year', 'Two year'])
billing = st.selectbox('Does the customer use paperless billing?', ['Yes', 'No'])
method_pay = st.selectbox('What is the payment method of the customer?', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
total_charge = st.number_input('What is the total charge of the customer (in USD)?')

if st.button('Submit'):

    num_cols = ['TotalCharges']
    cat_cols = ['SeniorCitizen', 'Partner', 'Dependents', 'InternetService',
       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
       'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
       'PaymentMethod']

    df_num = pd.DataFrame([[total_charge]], columns=num_cols)

    df_cat = pd.DataFrame([[seniors, partner, dependents, internet,
       sec_ol, backup_ol, dev_protect, tech_support,
       tv_stream, movie_stream, contract, billing,
       method_pay]], columns=cat_cols)

    df_deploy = pd.concat([df_num, df_cat], axis=1)

    X = pd.DataFrame(preprocess.transform(df_deploy))

    pred = model.predict(X)

    if pred [0][0] < 0.5:
        st.text('Customer will Churn! Please, react accordingly.')
    else:
        st.text('Congratulations! Your customer will not churn.')