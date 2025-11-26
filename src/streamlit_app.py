import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import EasyEnsembleClassifier
from imblearn.metrics import classification_report_imbalanced
from imblearn.under_sampling import ClusterCentroids


mitbih_test = pd.read_csv('input/mitbih_test.csv')
mitbih_train = pd.read_csv('input/mitbih_train.csv')
ptbdb_abnormal = pd.read_csv('input/ptbdb_abnormal.csv')
ptbdb_normal = pd.read_csv('input/ptbdb_normal.csv')

#rename columns

for df in [ptbdb_abnormal, ptbdb_normal, mitbih_test, mitbih_train]:
    df.columns = [i for i in range(len(df.columns))]

#combine datasets, remove class 4, combine classes 1,2,3

ptbdb = pd.concat([ptbdb_abnormal, ptbdb_normal])
mitbih = pd.concat([mitbih_train, mitbih_test])

mitbih_recoded = mitbih.loc[mitbih[187] != 4]
mitbih_recoded.loc[:, 187] = mitbih_recoded[187].replace([1,2,3], 1)

df_total = pd.concat([mitbih_recoded, ptbdb])

#split into train and test

X = df_total.drop(187, axis=1)
y = df_total[187]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8)

#load models

knn = joblib.load('best_knn_cc')

#streamlit code

st.title("ECG Heartbeat Categorization")
st.sidebar.title("Table of contents")
pages=["The ECG Heartbeat Categorization Problem", "Data Analysis and Visualisation", "Modelling", "Model Analysis", "Conclusion"]
page=st.sidebar.radio("Go to", pages)

if page == pages[0]:
    st.write('### Context')

if page == pages[1]:
    st.write('### Data Analysis and Visualization')

if page == pages[2]:
    st.write('### Modelling')

    def scores(clf, choice):
        if choice == 'Accuracy':
            return clf.score(X_test, y_test)
        elif choice == 'Confusion matrix':
            return confusion_matrix(y_test, clf.predict(X_test))
    
    choice = ['KNN with Cluster Centroids']
    option = st.selectbox('Choice of the model', choice)
    st.write('The chosen model is :', option)

    display = st.radio('What do you want to show ?', ('Accuracy', 'Confusion matrix'))
    if display == 'Accuracy':
        st.write(scores(knn, display))
    elif display == 'Confusion matrix':
        st.dataframe(scores(knn, display))

if page == pages[3]:
    st.write('### Model Analysis')

if page == pages[4]:
    st.write('### Conclusion')