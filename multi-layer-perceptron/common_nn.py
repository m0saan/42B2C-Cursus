from mlp.tensor import Tensor
import mlp.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(NeuralNetwork, self).__init__()
        self.dense1 = nn.Linear(in_features=input_shape, out_features=100)
        self.dense2 = nn.Linear(100, 50)
        self.dense3 = nn.Linear(50, 25)
        self.dense4 = nn.Linear(25, output_shape)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.relu(self.dense1(x))
        x = self.relu(self.dense2(x))
        x = self.relu(self.dense3(x))
        x = self.dense4(x)
        return x

# Custom Dataset class
class MyDataset(nn.Dataset):
    def __init__(self, X, y):
        self.X = Tensor(X)
        self.y = Tensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]