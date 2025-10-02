import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_heatmap(df: pd.DataFrame, title="Correlation Heatmap"):
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

