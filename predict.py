import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score,precision_score
import streamlit as st

data = pd.read_csv("/Users/ajay/Documents/programs/mini-project/creditcard.csv")

# Determine number of fraud cases in dataset

Fraud = data[data['Class'] == 1]
Valid = data[data['Class'] == 0]

outlier_fraction = len(Fraud)/float(len(Valid))
print(outlier_fraction)

print('Fraud Cases: {}'.format(len(data[data['Class'] == 1])))
print('Valid Transactions: {}'.format(len(data[data['Class'] == 0])))

#seperating the X and the Y from the dataset
X=data.drop(['Class'], axis=1)
Y=data["Class"]

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

# Building the Random Forest Classifier (RANDOM FOREST)
# random forest model creation
rfc = RandomForestClassifier()
rfc.fit(X_train,Y_train)

# predictions
y_pred = rfc.predict(X_test)


def show_predict():

    st.title(" Fraud Detection ")
    input_df = st.text_input("enter all required values")
    input_df_splited = input_df.split(',')


    submit = st.button("Sumbit")

    if submit:
        required_values = np.asarray(input_df_splited,dtype=np.float64)
        prediction = rfc.predict(required_values.reshape(1,-1))


        if prediction[0] == 0:
            st.write("Legitimate Transaction")
        else:
            st.write("Fraudlent Transaction")
