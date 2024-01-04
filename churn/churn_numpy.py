import nn as nnx
from tensor import Tensor
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle

class NeuralNetwork(nnx.Module):
    def __init__(self, input_shape, output_shape):
        super(NeuralNetwork, self).__init__()
        self.dense1 = nnx.Linear(in_features=input_shape, out_features=100)
        self.dense2 = nnx.Linear(100, 50)
        self.dense3 = nnx.Linear(50, 25)
        self.dense4 = nnx.Linear(25, output_shape)
        self.relu = nnx.ReLU()

    def forward(self, x):
        x = self.relu(self.dense1(x))
        x = self.relu(self.dense2(x))
        x = self.relu(self.dense3(x))
        x = self.dense4(x)
        return x

# Custom Dataset class
class MyDataset(nnx.Dataset):
    def __init__(self, X, y):
        self.X = Tensor(X)
        self.y = Tensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
    
def fit(epochs, lr, model, loss_func, opt_fn, train_dl, valid_dl):
    recorder = {'tr_loss': [], 'val_loss': [], 'tr_acc': [], 'val_acc': []}
    losses = [[], []]
    opt = opt_fn(model.parameters(), lr=lr)
    
    for epoch in tqdm(range(epochs)):
        model.train()
        train_tot_loss, train_tot_acc, t_count = 0.,0.,0
        for xb,yb in train_dl:
            preds = model(xb)
            loss = loss_func(preds, yb)
            loss.backward()
            opt.step()
            opt.zero_grad()

            # Calculate accuracy & loss
            predicted_labels = preds.argmax(axis=1)
            n = len(xb)
            t_count += n
            train_tot_loss += loss.item()*n
            train_tot_acc  += Tensor.accuracy(predicted_labels, yb).item()*n
            recorder['tr_loss'].append(loss.item())
            recorder['tr_acc'].append(Tensor.accuracy(predicted_labels, yb).item())
            losses[0].append(loss.item())
            
        print(f"epoch {epoch + 1:02d}/{epochs:02d} - loss: {train_tot_loss/t_count:.4f} - acc: {train_tot_acc/t_count:.4f}")
            
    return recorder


ds_path = 'ds.pkl'
import torch

with open(ds_path, 'rb') as f:
    X_tr, y_tr, X_val, y_val = pickle.load(f)
y_tr = y_tr[:, 0]
y_val = y_val[:, 0]

# Create the neural network
input_shape = X_tr.shape[1]  # Replace with the actual input shape
output_shape = 2  # Replace with the actual output shape

model = NeuralNetwork(input_shape, output_shape)

tr_ds = MyDataset(X_tr, y_tr)
val_ds = MyDataset(X_val, y_val)

# Creating the data loader
bs = 512
tr_dl = nnx.DataLoader(tr_ds, batch_size=bs)
val_dl = nnx.DataLoader(val_ds, batch_size=bs)

lr = 0.001
n_epochs = 20
recorder = fit(n_epochs, lr, model, nnx.CrossEntropyLoss(), nnx.SGD, tr_dl, val_dl)
