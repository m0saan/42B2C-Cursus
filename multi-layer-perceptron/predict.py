import numpy as np
from common_nn import NeuralNetwork
from data_prep import data_split_normalize
from mlp.tensor import Tensor
import mlp.nn as nn
import pandas as pd

def predict(X, model):
    preds = model(X)
    pred_labels = preds.argmax(axis=1)
    
    for i, pred in enumerate(pred_labels):
        if pred == 1:
            print(f'>>> Sample {i} is a malignant tumor.')
        else:
            print(f'>>> Sample {i} is a benign tumor.')
    


if __name__ == "__main__":
    _, _, X_valid, y_valid = data_split_normalize('data/data.csv', seed=42)
    X_val, y_val = map(Tensor, (X_valid, y_valid))
    print('>>> Loading Dataset...')
    
    input_shape = X_val.shape[1]
    output_shape = 2
    model = NeuralNetwork(input_shape, output_shape)
    model.load_weights(path='model_params.pkl')
    
    predict(X_val, model)    
    
