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
from tensorflow.keras.models import load_model


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

knn = joblib.load('src/saved_models/best_knn_cc')
mlp = load_model('src/saved_models/best_mlp.keras')
history = joblib.load('src/saved_models/best_mlp_history.joblib')

#streamlit code

st.title("ECG Heartbeat Categorization")
st.sidebar.title("Table of contents")
pages=["The ECG Heartbeat Categorization Problem", "Data Analysis and Visualisation", "Modelling and Analysis", "Conclusion"]
page=st.sidebar.radio("Go to", pages)

if page == pages[0]:
    st.write('### Context')

if page == pages[1]:
    st.title('Data Analysis and Visualization')

    st.markdown('''
                The following datasets have been provided with the Project-data:

                The MIT-BIH Arrhythmia Dataset: mitbih_test, mitbih_train  
                The PTB Diagnostic ECG Dataset: ptbdb_abnormal, ptbdb_normal 

                Both Datasets contain ecg information of heartbeats which were further divided into train and test data for mitbih and into normal and abnormal beats for ptbdb.

                The columns in all datasets have been renamed 0 through 187 for ease of analysis.
                ''')
    
    st.subheader('MIT-BIH')

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
    

    st.subheader('PTB')
    with st.expander("ptbdb_abnormal"):
        st.dataframe(ptbdb_abnormal.head())
    with st.expander("ptbdb_normal"):
        st.dataframe(ptbdb_normal.head())

    st.markdown('''
                As with mitbih, the ptb datasets are made up of 188 columns with the final column representing the heartbeat classification category, and the preceding columns representing
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

    st.subheader('Combined Datasets')

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

    st.title('Modelling and Analysis')

    st.subheader('Machine Learning Model Comparison')

    st.markdown('''
                Using the preprocessed data defined before, four initial models were tested:  
                Linear Regression, K-Nearest-Neighbours, Random Forest Classifier, and Decision Tree.  
                  
                These models were evaluated with default parameters.  

                Given the imbalance in the data, the models were then evaluated on a resampled dataset using the following resampling methods:  
                Random Oversampling, Random Undersampling, SMOTE, Cluster Centroids.  
                  
                The results are presented in the table below, with each model evaluated on training set accuracy, test set accuracy, normal category F1 score, and abnormal category F1 score.
                ''')
    
    ml_model_comparison = pd.read_csv('src/streamlit_tables/ML_model_comparison.csv').drop(columns='Comments')

    with st.expander('Machine Learning Model Comparison'):
        st.dataframe(ml_model_comparison)
    
    st.write('We conclude that the best performing machine learning model is the KNN with cluster centroids resampling, with a test set accuract of 0.95 and no apparent overfitting')

    st.markdown('''
                A note on cross validation:  
                A five fold cross validation was performed using GridSearchCV on Logistic Regression, KNN, and Random Forest Classifier. We concluded that optimisation of the hyperparameters
                does not improve test set accuracy whilst avoiding overfitting, thus the choice of default parameters is justified.
                ''')

    st.subheader('MLP Model Comparison')

    st.markdown('''
                We next built and trained a series of Multi-Layer Perceptron models, varying the number of dense layers, the number of neurons in these layers, the introduction 
                of dropout layers, and the activation function of the dense layers.  

                The sigmoid output activation function was used as it is optimal for binary classification problems. The models were trained over 50 epochs, with an early stopping callback
                based on validation loss.

                As before the results are presented in the table below, with the same evaluation metrics. 
                ''')
    
    mlp_model_comparison = pd.read_csv('src/streamlit_tables/mlp_model_comparison.csv').drop(columns='Comments')

    with st.expander('MLP Model Comparison'):
        st.dataframe(mlp_model_comparison)

    st.markdown('''
                We conclude that the best model had dense layers [512, 256, 128], relu activation, Dropout(0.4) after each layer, and sigmoid output activation.  
                With a test set accuracy of 0.97, it is our best performer and with a training set accuracy of 0.97 appears to avoid overfitting.  

                Whilst all the models achieved good test set accuracy, many had a higher training set accuracy indicating overfitting. This is also seen in the diverging accuracy graphs.
                ''')


    st.subheader('Summary of Best Models')

    st.write('We conclude this section with a summary of our best machine learning model and our best deep learning model.')
    
    models = {'KNN with Cluster Centroids': knn, 'MLP': mlp}
    choice = ['KNN with Cluster Centroids', 'MLP']
    option = st.selectbox('Choice of the model', choice)
    st.write('The chosen model is :', option)
    clf = models[option]

    if option == 'KNN with Cluster Centroids':

        display = st.radio('What do you want to show ?', ('Test Set Accuracy', 'Training Set Accuracy', 'Confusion matrix'))

        if display == 'Test Set Accuracy':
            st.write(clf.score(X_test, y_test))
        elif display == 'Training Set Accuracy':
            st.write(clf.score(X_train, y_train))
        elif display == 'Confusion matrix':
            y_pred = clf.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            ax.set_title('Confusion Matrix')
            ax.invert_yaxis()
            st.pyplot(fig)

    if option == 'MLP':

        display = st.radio('What do you want to show ?', ('Test Set Accuracy', 'Training Set Accuracy', 'Confusion matrix', 'Accuracy and Loss Graphs'))

        if display == 'Test Set Accuracy':
            st.write(clf.evaluate(X_test, y_test)[1])
        elif display == 'Training Set Accuracy':
            st.write(clf.evaluate(X_train, y_train)[1])
        elif display == 'Confusion matrix':
            y_pred_probs = mlp.predict(X_test)
            y_pred_classes = (y_pred_probs > 0.5).astype(int).flatten()
            cm = confusion_matrix(y_test, y_pred_classes)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title("Confusion Matrix")
            ax.invert_yaxis()
            st.pyplot(fig)
        elif display == 'Accuracy and Loss Graphs':
            fig_acc, ax_acc = plt.subplots()
            ax_acc.plot(history['accuracy'], label='Train Acc')
            ax_acc.plot(history['val_accuracy'], label='Val Acc')
            ax_acc.set_xlabel('Epoch')
            ax_acc.set_ylabel('Accuracy')
            ax_acc.legend()
            st.pyplot(fig_acc)  

            fig_loss, ax_loss = plt.subplots()
            ax_loss.plot(history['loss'], label='Train Loss')
            ax_loss.plot(history['val_loss'], label='Val Loss')
            ax_loss.set_xlabel('Epoch')
            ax_loss.set_ylabel('Loss')
            ax_loss.legend()
            st.pyplot(fig_loss)  

if page == pages[3]:
    st.title('Conclusion')

    st.markdown('''
        In this DS Heartbeat Project, we developed, evaluated, and compared multiple models for the binary classification of heartbeats from ECG data, aiming to distinguish between normal and abnormal classes using the MIT-BIH and PTBDB datasets.
        Key takeaways include:
            ''')
    
    st.markdown("- The KNeighborsClassifier combined with Cluster Centroids resampling consistently demonstrated strong performance, achieving excellent accuracy and balanced precision-recall metrics without clear signs of overfitting. This combination is a robust choice for this task.")
    st.markdown("- However, the KNN model paired with SMOTE resampling also showed very competitive, and in some metrics even slightly better results—particularly in detecting the minority abnormal class—highlighting that interpretation of “best” performance can vary depending on which metric or trade-off is prioritized.")
    st.markdown("- The Multi-Layer Perceptron (MLP) models delivered impressive accuracy and F1 scores, in some cases surpassing traditional machine learning models. While potential overfitting was indicated by discrepancies between training and validation accuracy, their performance suggests that deep learning approaches are promising for ECG classification.")
    st.markdown("- Simpler models like Logistic Regression, despite hyperparameter tuning, were not sufficient to capture the complex patterns in the ECG data.")
    st.markdown("- Random Forest models performed well but tended to overfit, requiring further regularization or tuning.")
    st.markdown("- Overall, resampling methods such as Cluster Centroids and SMOTE proved essential to address class imbalance and improve minority class detection.")
    st.markdown('''
    In conclusion, while the KNN + Cluster Centroids combination provides a stable and interpretable model with strong generalization, the results for KNN + SMOTE and MLPs highlight alternative promising approaches, depending on specifi	c performance priorities. Future work should explore advanced regularization, incorporate more diverse datasets, and expand on deep learning architectures and transfer learning techniques to further improve classification performance and clinical applicability.
This project demonstrates the potential of data-driven methods to enhance automated ECG analysis and contribute meaningfully to cardiovascular diagnostics.
            ''')