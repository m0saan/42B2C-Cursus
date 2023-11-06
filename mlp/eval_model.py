import numpy as np
from common_nn import NeuralNetwork
from data_prep import data_split_normalize
from mlp.tensor import Tensor
import mlp.nn as nn

def binary_cross_entropy(predictions, y_true):
    y_true = y_true.numpy()
    predictions = nn.Softmax()(predictions).numpy().max(axis=1)
    epsilon = 1e-15  # to avoid log(0) error
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(predictions) + (1 - y_true) * np.log(1 - predictions))


if __name__ == "__main__":
    _, _, X_valid, y_valid = data_split_normalize('data/data.csv', seed=42)
    X_val, y_val = map(Tensor, (X_valid, y_valid))
    print('>>> Loading Dataset...')
    
    input_shape = X_val.shape[1]
    output_shape = 2
    model = NeuralNetwork(input_shape, output_shape)
    print('>>> Creating Model...')
    
    # Load the model weights
    model.load_weights(path='model_params.pkl')
    print('>>> Loading Model Weights...')
    
    # Evaluate the model on the validation set
    preds = model(X_val)
    
    pred_labels = preds.argmax(axis=1)
    accuracy = Tensor.accuracy(pred_labels, y_val).item()
    print(f'>>> Accuracy: {accuracy:.4f}')
    
    bce = binary_cross_entropy(preds, y_val)
    print(f'>>> Binary Cross-Entropy Error: {bce:.4f}')
    
    
    
