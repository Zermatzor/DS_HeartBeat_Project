from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


input_folder = os.path.join(os.path.dirname(__file__), "..", "input")
files = os.listdir(input_folder)
files = [f for f in files if os.path.isfile(os.path.join(input_folder, f))]
print("Dateien im Input-Ordner:")
for f in files:
    print(f)


nRowsRead = 1000  
csv_path = os.path.join(os.path.dirname(__file__), "..", "input", "mitbih_test.csv")
df1 = pd.read_csv(csv_path, delimiter=",", nrows=nRowsRead)
df1.dataframeName = "mitbih_test.csv"

nRow, nCol = df1.shape
print(f"There are {nRow} rows and {nCol} columns")
print(df1.head())



