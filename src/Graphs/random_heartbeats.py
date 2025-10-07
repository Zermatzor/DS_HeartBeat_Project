import pandas as pd
import matplotlib.pyplot as plt
import random
from Data.load_data import load_dataframe



def show_plot():
    #Import data
    
    mitbih_test = load_dataframe("test")
    mitbih_train = load_dataframe("train")
    ptbdb_abnormal = load_dataframe("abnormal")
    ptbdb_normal = load_dataframe("normal")

    #replace column name with the index 0 through 187

    for df in [mitbih_test, mitbih_train, ptbdb_abnormal, ptbdb_normal]:
        df.columns = [i for i in range(len(df.columns))]

    #make new dataframes for each target class in mitbih_train

    mitbih_train_class0 = mitbih_train[mitbih_train[187]==0].drop(187, axis=1)
    mitbih_train_class1 = mitbih_train[mitbih_train[187]==1].drop(187, axis=1)
    mitbih_train_class2 = mitbih_train[mitbih_train[187]==2].drop(187, axis=1)
    mitbih_train_class3 = mitbih_train[mitbih_train[187]==3].drop(187, axis=1)
    mitbih_train_class4 = mitbih_train[mitbih_train[187]==4].drop(187, axis=1)

    #plot graphs

    plt.figure(figsize=(20,20))
    plt.suptitle('5 randomly selected rows for each class in mitbih')

    graph_index = 1

    for i, df in enumerate([mitbih_train_class0, mitbih_train_class1, mitbih_train_class2, mitbih_train_class3, mitbih_train_class4]):
        for _ in range(5):
            n = random.choice(df.index.values)
            plt.subplot(5 ,5, graph_index)
            plt.plot(df.columns, df.loc[n])
            plt.title(f'Class {i}')
            graph_index += 1

    plt.show()