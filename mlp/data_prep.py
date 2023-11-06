import pandas as pd
import numpy as np

def normalize(data):
    mean = np.mean(data)
    std = np.std(data)
    normalized_data = (data - mean) / std
    return normalized_data

def data_split_normalize(file_path, seed=42):
    df = pd.read_csv(file_path).sample(frac=1, random_state=seed).reset_index(drop=True)
    X = df.iloc[:, 2:].values
    y = df.iloc[:, 1].values
    y = np.where(y == 'M', 1, 0)
    
    normalized_X = normalize(X)
    
    split_ratio = 0.8
    split_index = int(split_ratio * len(X))
    X_tr, y_tr = normalized_X[:split_index], y[:split_index]
    X_val, y_val = normalized_X[split_index:], y[split_index:]
    return X_tr, y_tr, X_val, y_val