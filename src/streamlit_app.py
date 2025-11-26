import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import random
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

    st.markdown('''
                The following datasets have been provided with the Project-data:

                The MIT-BIH Arrhythmia Dataset: mitbih_test, mitbih_train  
                The PTB Diagnostic ECG Dataset: ptbdb_abnormal, ptbdb_normal 

                Both Datasets contain ecg information of heartbeats which were further divided into train and test data for mitbih and into normal and abnormal beats for ptbdb.

                The columns in all datasets have been renamed 0 through 187 for ease of analysis.
                ''')
    
    st.write('#### MIT-BIH')

    with st.expander("mitbih_train"):
        st.dataframe(mitbih_train.head())

    st.write('The columns in each set are points in time (HZ) and represent the timeline of a heartbeat.')

    st.markdown('''
                The last column in the MIT-BIH dataset represents a categorical variable, which corresponds to one of the following heartbeat-types:  
                0.0 = Normal beat  
                1.0 = Supraventricular ectopic beat  
                2.0 = Ventricular ectopic beat  
                3.0 = Fusion beat  
                4.0 = Unknown beat  
''')
    
    st.write('Category distribution:')
    st.write(mitbih_train.iloc[:, -1:].value_counts())

    st.markdown('''
We can see that the most prevalent type of heartbeat is the “normal heartbeat”, followed by “unknown heartbeat” and “ventricular ectopic beat”.  
Since the unknown heartbeat could also be a normal heartbeat we will omit this variable from our analysis and calculated the percentage of known abnormal heartbeats to be 17.23 %
''')
    
    st.write('In order to get a broad overview of the different heartbeats we randomly chose 5 of each type and visualized them.')

    def five_graphs_mitbih():
        mitbih_train_class0 = mitbih_train[mitbih_train[187]==0].drop(187, axis=1)
        mitbih_train_class1 = mitbih_train[mitbih_train[187]==1].drop(187, axis=1)
        mitbih_train_class2 = mitbih_train[mitbih_train[187]==2].drop(187, axis=1)
        mitbih_train_class3 = mitbih_train[mitbih_train[187]==3].drop(187, axis=1)

        plt.figure(figsize=(20,20))
        plt.suptitle('5 randomly selected rows for each class in mitbih')

        graph_index = 1

        for i, df in enumerate([mitbih_train_class0, mitbih_train_class1, mitbih_train_class2, mitbih_train_class3]):
            for _ in range(5):
                n = random.choice(df.index.values)
                plt.subplot(4 ,5, graph_index)
                plt.plot(df.columns, df.loc[n])
                plt.title(f'Class {i}')
                graph_index += 1

        st.pyplot(plt)
    
    five_graphs_mitbih()
    

    st.write('#### PTB')
    with st.expander("ptbdb_abnormal"):
        st.dataframe(ptbdb_abnormal.head())
    with st.expander("ptbdb_normal"):
        st.dataframe(ptbdb_normal.head())

    st.markdown('''
                As with mitbih, the ptb datasets are made up of 188 coolumns with the final column representing the heartbeat classification category, and the preceding columns representing
                a sample of the heartbeat taken at a sampling frequency of 125Hz, i.e. a sampling rate of every 8 milliseconds.  

                We see that ptbdb_abnormal takes value 1 in the final column, whereas ptbdb_normal takes value 0. This aligns with the mitbih dataset, insofar as 0 represents the normal 
                category for a heartbeat.  

                It therefore makes sense to combine ptbdb_normal and ptbdb_abnormal for analysis, so that both normal and abnormal heartbeats are contained within one dataset, as with mitbih.

                We can then analyse the distribution of the target variable for the combined dataset.
                ''')
    
    def ptb_target_dist():
        plt.figure(figsize=(6,4))
        sns.countplot(x=ptbdb[187])
        plt.title('Distribution of target variable for combined ptbdb dataset')
        plt.xlabel('Heartbeat Classification')
        plt.grid(True)
        st.pyplot(plt)

        prop0 = ptbdb[187].value_counts(normalize=True)[0] 
        prop1 = ptbdb[187].value_counts(normalize=True)[1]
        st.write('Proportion of class 0: ', prop0)
        st.write('Proportion of class 1: ', prop1)

    ptb_target_dist()

    st.write('We will now consider what a random sample of heartbeats from each class looks like.')

    def five_graphs_ptb():
        plt.figure(figsize=(20,7))
        plt.suptitle('5 randomly selected rows for ptbdb normal and abnormal')

        graph_index = 1

        for label, df in [('Abnormal', ptbdb_abnormal.drop(187, axis=1)), ('Normal', ptbdb_normal.drop(187, axis=1))]:
            for _ in range(5):
                n = random.choice(df.index.values)
                plt.subplot(2 ,5, graph_index)
                plt.plot(df.columns, df.loc[n])
                plt.title(f'Class {label}')
                graph_index += 1

        st.pyplot(plt)

    five_graphs_ptb()

    st.write('#### Combined Datasets')

    st.markdown('''
                We now consider the datasets as a whole.   
                Given that mitbih has classes 0 for normal and 1,2,3,4 for abnormalities, we can recode the classes 1,2,3,4 into simply class 1 so that it matches with ptb. 
                We can then perform an analysis on the entire dataset. In fact, class 4 represents unclassifiable data so we exclude this category entirely.   

                We first combine mitbih_train and mitbih_test, then perform the encoding, and then combine all data and analyse the distribution of the target variable:
                ''')
    
    def combined_target_dist():
        plt.figure(figsize=(8,12))

        ax1 = plt.subplot2grid((3, 2), (0, 0))  
        ax2 = plt.subplot2grid((3, 2), (0, 1))
        ax3 = plt.subplot2grid((3, 4), (1, 1), colspan=2)  
        ax4 = plt.subplot2grid((3, 4), (2, 1), colspan=2)  

        sns.countplot(x=mitbih[187], ax=ax1)
        ax1.set_title('Distribution of target variable for mitbih')
        ax1.set_xlabel('Target variable')
        ax1.grid(True)

        sns.countplot(x=mitbih_recoded[187], ax=ax2)
        ax2.set_title('Distribution of target variable for mitbih encoded')
        ax2.set_xlabel('Target variable')
        ax2.grid(True)

        sns.countplot(x=ptbdb[187], ax=ax3)
        ax3.set_title('Distribution of target variable for ptbdb')
        ax3.set_xlabel('Target variable')
        ax3.grid(True)

        sns.countplot(x=df_total[187], ax=ax4)
        ax4.set_title('Distribution of target variable for combined ptbdb and mitbih datasets')
        ax4.set_xlabel('Target variable')
        ax4.grid(True)

        plt.tight_layout()
        st.pyplot(plt)

    combined_target_dist()

    st.write('From this we can conclude that a combined dataset provides a slightly more even distribution of the target variable, which is desirable when we come to apply a machine learning model.')

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