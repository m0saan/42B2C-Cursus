import pandas as pd
import numpy as np

def data_split(file_path):
    df = pd.read_csv(file_path)
    X = df.iloc[:, 2:]
    y = df.iloc[:, 1].values
    y = np.where(y == 'M', 1, 0)
    
    split_ratio = 0.8
    split_index = int(split_ratio * len(X))
    X_tr, y_tr = X[:split_index], y[:split_index]
    X_val, y_val = X[split_index:], y[split_index:]
    return X_tr, y_tr, X_val, y_val