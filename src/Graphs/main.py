# code/main.py
from Data.load_data import load_csv_data
import Graphs.plot as plot
import pandas as pd

def load_preview_data(nRows=1000):
    """
    Load preview of training and test data.
    
    Returns:
    - X_train, y_train, X_test, y_test
    """
    # Load training data
    train_data = load_csv_data(files_to_load=["train"], nRows=nRows)
    X_train, y_train = train_data["train"]
    print("Loaded training data")
    print("Shape X_train:", X_train.shape)
    print("Shape y_train:", y_train.shape)

    # Load test data
    test_data = load_csv_data(files_to_load=["test"], nRows=nRows)
    X_test, y_test = test_data["test"]
    print("Loaded test data")
    print("Shape X_test:", X_test.shape)
    print("Shape y_test:", y_test.shape)

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    # Load preview data
    #X_train, y_train, X_test, y_test = load_preview_data(nRows=1000)
  
    ab_normal_data = load_csv_data(files_to_load=["normal", "abnormal"], nRows=500)

    # Plot heatmap for each dataset
    for key, (X, y) in ab_normal_data.items():
        df = pd.DataFrame(X)
        plot.heatmap(df, title=f"Correlation heatmap for {key} ({X.shape[0]} samples)")

    # Single Plot of Test csv
    X_test, y_test = load_csv_data(["test"], nRows=500)["test"]
    plot.heatmap(pd.DataFrame(X_test), title=f"Correlation heatmap for test ({X_test.shape[0]} samples)")

    # Single Plot of Train csv
    plot.heatmap(
        pd.DataFrame(load_csv_data(["train"], nRows=500)["train"][0]),
        title="Correlation heatmap for train"
    )

    # load training data
    X_train, y_train = load_csv_data(["train"], nRows=None)["train"]
    df_train = pd.DataFrame(X_train)
    # plot heatmaps per class
    plot.class_heatmaps(df_train, y_train, class_ids=[0, 1, 2, 3, 4], dataset_name="train")



