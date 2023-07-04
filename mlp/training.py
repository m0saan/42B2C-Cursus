import IPython.display as ipd
import matplotlib.pyplot as plt
from data_split import data_split
import numpy as np
from utils import normalize
from tensor import *
from tqdm.notebook import tqdm
from IPython.display import display
from nn import (
    Linear,
    ReLU,
    Softmax,
    CrossEntropyLoss,
    Module,
    Dataset,
    DataLoader,
    SGD,
)

class NeuralNetwork(Module):
    def __init__(self, input_shape, output_shape):
        super(NeuralNetwork, self).__init__()
        self.dense1 = Linear(in_features=input_shape, out_features=100)
        self.dense2 = Linear(100, 50)
        self.dense3 = Linear(50, 25)
        self.dense4 = Linear(25, output_shape)
        self.relu = ReLU()
        self.softmax = Softmax()

    def forward(self, x):
        x = self.relu(self.dense1(x))
        x = self.relu(self.dense2(x))
        x = self.relu(self.dense3(x))
        x = self.dense4(x)
        return x

# Custom Dataset class
class MyDataset(Dataset):
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
    fig, axs = plt.subplots(1, 1, figsize=(14, 7))
    p = display(fig,display_id=True)

    opt = opt_fn(model.parameters(), lr=lr, wd=0.05, momentum=0.9)
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
            

            

        model.eval()
        val_tot_loss, val_tot_acc,v_count = 0.,0.,0
        for xb,yb in valid_dl:
            preds = model(xb)

            pred_labels = preds.argmax(axis=1)
            n = len(xb)
            v_count += n
            val_tot_acc  += Tensor.accuracy(pred_labels, yb).item()*n
            val_tot_loss += loss_func(preds,yb).item()*n
            recorder['val_loss'].append(loss_func(preds,yb).item())
            recorder['val_acc'].append(Tensor.accuracy(pred_labels, yb).item())
            losses[1].append(loss_func(preds,yb).item())

            
        print(f"epoch {epoch + 1:02d}/{epochs:02d} - loss: {train_tot_loss/t_count:.4f} - acc: {train_tot_acc/t_count:.4f} - val_loss: {val_tot_loss/v_count:.4f} - val_acc: {val_tot_acc/v_count:.4f}")
        
        axs.plot(losses[0], label='Train loss:')
        axs.plot(losses[1], label='Validation loss:')
        # p.update(fig)

    # ipd.clear_output()
    # p.update(fig)    
    return recorder

if __name__ == "__main__":
    X_train, y_train, X_valid, y_valid = data_split('data/data.csv')
    X_train = normalize(X_train)
    X_valid = normalize(X_valid)
    
    print(f'--------> X_train shape : {X_train.shape}')
    print(f'--------> X_valid shape : {X_valid.shape}')

    X_tr, y_tr, X_val, y_val = map(Tensor, (X_train, y_train, X_valid, y_valid))
    
    # Create the neural network
    input_shape = X_tr.shape[1]  # Replace with the actual input shape
    output_shape = 2  # Replace with the actual output shape

    model = NeuralNetwork(input_shape, output_shape)
    print(model)
    
    tr_ds = MyDataset(X_tr, y_tr)
    val_ds = MyDataset(X_val, y_val)

    # # Creating the data loader
    bs = 128
    tr_dl = DataLoader(tr_ds, batch_size=bs, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False)
    # print(next(iter(val_dl)))
    
    recorder = fit(100, 0.005, model, CrossEntropyLoss(), SGD, tr_dl, val_dl)


    # plt.figure(figsize=(10, 7))
    # plt.plot(recorder['tr_loss'], label='Training Loss')
    # plt.plot(recorder['val_loss'], label='Validation Loss', linestyle='--')
    # plt.xlabel('Iterations')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.title('Loss Curve')
    # plt.show()

    # plt.figure(figsize=(10, 7))
    # plt.plot(recorder['tr_acc'], label='Training Accuracy')
    # plt.plot(recorder['val_acc'], label='Validation Accuracy')
    # plt.xlabel('Iterations')
    # plt.ylabel('Accuracy')
    # plt.legend()
    # plt.title('Accuracy Curve')
    # plt.show()

