import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


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

def class_heatmaps(df, labels, class_ids, dataset_name="dataset"):
    """Plot heatmaps for selected classes in the dataset."""
    labels = pd.Series(labels)  # ensure it's a Series
    for class_id in class_ids:
        # Select only rows belonging to this class
        df_class = df[labels == class_id]

        if df_class.empty:
            print(f"No samples found for class {class_id} in {dataset_name}")
            continue

        heatmap(
            df_class,
            title=f"Correlation heatmap for class {class_id} in {dataset_name} ({len(df_class)} samples)"
        )
