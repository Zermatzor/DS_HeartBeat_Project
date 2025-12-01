#import packages and data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import data_load_data 
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.under_sampling import ClusterCentroids
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

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

# build MLP model sequentially; Dense layers: [512, 256, 128], activation: relu, dropout layers: Dropout(0.4) after each layer
# output activation: sigmoid
# epochs 50

model = Sequential()

model.add(Dense(units=512, activation='relu', input_shape=(187,)))
model.add(Dropout(0.4))
model.add(Dense(units=256, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(units=1, activation='sigmoid'))

#compile model

model.compile(loss="binary_crossentropy", #loss function for binary classification
              optimizer="adam",
              metrics=["accuracy"])

early_stop = EarlyStopping(monitor='val_loss',  #if no improvement in val_loss after 5 epochs, stop and keep best model
                           patience=5,         
                           restore_best_weights=True)  

#train model

history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, callbacks=[early_stop])

#save model

model.save("src/saved_models/best_mlp.keras")
joblib.dump(history.history, 'src/saved_models/best_mlp_history.joblib')