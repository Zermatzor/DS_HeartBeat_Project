# code/main.py
from load_data import load_csv_data
from plot import heatmap

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

    
    # Plot
    # Single class
    heatmap("test", nRows=500)

    # Multiple classes
    heatmap(["normal","abnormal"], nRows=500)



