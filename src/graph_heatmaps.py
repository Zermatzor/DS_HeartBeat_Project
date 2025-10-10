# code/main.py
from data_load_data import load_csv_data
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def heatmap_for_classes():
    X_train, y_train = load_csv_data(["train"], nRows=None)["train"]

    df_train = pd.DataFrame(X_train)
    class_heatmaps(df_train, y_train, class_ids=[0, 1, 2, 3, 4], dataset_name="train")


def class_heatmaps(df, labels, class_ids, dataset_name="dataset"):
    """Plot heatmaps for selected classes in the dataset."""
    labels = pd.Series(labels)  # ensure it's a Series
    fig, axes = plt.subplots(len(class_ids),1, figsize=(7, 35))
    for idx, class_id in enumerate(class_ids):
        # Select only rows belonging to this class
        df_class = df[labels == class_id]

        if df_class.empty:
            print(f"No samples found for class {class_id} in {dataset_name}")
            continue

        corr = df_class.corr()

        sns.heatmap(corr, cmap="coolwarm", center=0, ax=axes[idx])

    plt.title(f"Correlation heatmap for class {class_id} in {dataset_name} ({len(df_class)} samples)")
    plt.xlabel("Signal index")
    plt.ylabel("Signal index")
    plt.show()

def heatmap_train():
    # Single Plot of Train csv
    heatmap(
        pd.DataFrame(load_csv_data(["train"], nRows=500)["train"][0]),
        title="Correlation heatmap for train"
    )

def heatmap_test():
    # Single Plot of Test csv
    X_test, y_test = load_csv_data(["test"], nRows=500)["test"]
    heatmap(pd.DataFrame(X_test), title=f"Correlation heatmap for test ({X_test.shape[0]} samples)")

def heatmap(df: pd.DataFrame, title="Correlation Heatmap"):
    """
    Plot correlation heatmap from a pandas DataFrame.

    Parameters:
    - df: pandas DataFrame (features only, no labels)
    - title: str, plot title
    """
    corr = df.corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title(title)
    plt.xlabel("Signal index")
    plt.ylabel("Signal index")
    plt.show()