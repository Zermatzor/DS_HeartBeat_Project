import os
from load_data import load_csv_data

# Path to input folder is already defined in datasets.py
# We'll just load one file at a time, e.g., train

# Load the training data (first 1000 rows as preview)
data = load_csv_data(files_to_load=["train"], nRows=1000)

# Access the loaded data
X_train, y_train = data["train"]

print("Loaded training data")
print("Shape X_train:", X_train.shape)
print("Shape y_train:", y_train.shape)

# Similarly, load test data
data_test = load_csv_data(files_to_load=["test"], nRows=1000)
X_test, y_test = data_test["test"]

print("Loaded test data")
print("Shape X_test:", X_test.shape)
print("Shape y_test:", y_test.shape)


