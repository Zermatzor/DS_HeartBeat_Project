from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# Path to the input folder (relative to main.py)
input_folder = os.path.join(os.path.dirname(__file__), "..", "input")

# List all files in the folder
files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
print("Files in input folder:")
for f in files:
    print(f)

# Number of rows to read (None = read all rows)
nRowsRead = None #1000

# Loop through each CSV file: read and show its shape
for f in files:
    if f.lower().endswith(".csv"):  # only CSV files
        csv_path = os.path.join(input_folder, f)
        df = pd.read_csv(csv_path, delimiter=",", nrows=nRowsRead)
        df.dataframeName = f  # store filename for reference
        nRow, nCol = df.shape
        print(f"\nFile: {f}")
        print(f"Rows: {nRow}, Columns: {nCol}")
        #print(df.head())



