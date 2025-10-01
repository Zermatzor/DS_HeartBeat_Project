import pandas as pd
from utility.datasets import FULL_PATHS

def load_csv_data(
    files_to_load=None,
    nRows=1000,
    chunksize=None
):
    """
    Load selected CSV files from input folder.
    
    Parameters:
    - files_to_load: list of str, keys of FULL_PATHS to load ('train', 'test', 'abnormal', 'normal')
                     If None, load all files.
    - nRows: int or None, number of rows to read from each CSV (default 1000)
    - chunksize: int or None, optional chunk size for reading large files
    
    Returns:
    - data_dict: dict, keys = filenames, values = tuples (X, y) or generator of tuples if chunksize used
    """
    if files_to_load is None:
        files_to_load = FULL_PATHS.keys()
    
    data_dict = {}
    
    for key in files_to_load:
        if key not in FULL_PATHS:
            print(f"Warning: {key} not in dataset keys, skipping.")
            continue
        
        csv_path = FULL_PATHS[key]
        
        if chunksize:
            # Return a generator of chunks
            chunk_gen = pd.read_csv(csv_path, delimiter=",", nrows=nRows, chunksize=chunksize)
            data_dict[key] = ((chunk.iloc[:, :-1].values, chunk.iloc[:, -1].values) for chunk in chunk_gen)
            print(f"Loaded {key} in chunks of size {chunksize}, preview {nRows} rows")
        else:
            df = pd.read_csv(csv_path, delimiter=",", nrows=nRows)
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
            data_dict[key] = (X, y)
            print(f"Loaded {key}: rows={X.shape[0]}, columns={X.shape[1]}")
    
    return data_dict
