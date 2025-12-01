#import libraries and data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import data_load_data
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


mitbih_test = data_load_data.load_dataframe("test")
mitbih_train = data_load_data.load_dataframe("train")
ptbdb_abnormal = data_load_data.load_dataframe("abnormal")
ptbdb_normal = data_load_data.load_dataframe("normal")

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

#create cluster centroids resample, fit to data 

cc = ClusterCentroids()
X_cc, y_cc = cc.fit_resample(X_train, y_train)

#fit knn to resampled data

knn = KNeighborsClassifier()
knn.fit(X_cc, y_cc)

#save model

joblib.dump(knn, "best_knn_cc")