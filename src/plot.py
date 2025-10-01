# code/plot.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from load_data import load_csv_data

# plot.py
from load_data import load_csv_data
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def heatmap(classes, nRows=1000):
    """
    Plot heatmaps for one or multiple classes.
    
    Parameters:
    - classes: str or list of str, e.g., "normal" or ["normal","abnormal"]
    - nRows: int, number of rows to load
    """
    # Convert single string to list
    if isinstance(classes, str):
        classes = [classes]
    
    for class_key in classes:
        # Load data
        data = load_csv_data(files_to_load=[class_key], nRows=nRows)
        X, _ = data[class_key]
        
        # Convert to DataFrame
        df = pd.DataFrame(X)
        
        # Compute correlation
        corr = df.corr()
        
        # Plot heatmap
        plt.figure(figsize=(12,10))
        sns.heatmap(corr, cmap="coolwarm", center=0)
        plt.title(f"Correlation heatmap for {class_key} ({X.shape[0]} samples)")
        plt.xlabel("Signal index")
        plt.ylabel("Signal index")
        plt.show()
