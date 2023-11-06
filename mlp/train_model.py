from data_prep import data_split_normalize
from mlp.tensor import Tensor
from tqdm import tqdm
from IPython.display import display
import IPython.display as ipd
import matplotlib.pyplot as plt
import mlp.nn as nn
from common_nn import NeuralNetwork, MyDataset

def fit(epochs, lr, model, loss_func, opt_fn, train_dl, valid_dl, patience=10, use_early_stoping=False):
    recorder = {'tr_loss': [], 'val_loss': [], 'tr_acc': [], 'val_acc': []}
    losses = [[], []]
    best_val_loss = float('inf')
    counter_early_stop = 0
    early_stop = False
    # fig, axs = plt.subplots(1, 1, figsize=(14, 7))
    # p = display(fig,display_id=True)

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

        if use_early_stoping:
            if recorder['val_loss'][-1] < best_val_loss:
                best_val_loss = recorder['val_loss'][-1]
                counter_early_stop = 0
                # Save the best model
                model.save_weights(path='best_model.pth')
            else:
                counter_early_stop += 1
                if counter_early_stop >= patience:
                    print("Early stopping triggered")
                    early_stop = True
                    
            if early_stop:
                print("Stopped")
                break
            
    return recorder


def check_pretrained():
    xb, yb = next(iter(val_dl))

    probs = model(xb)
    pred_labels = probs.argmax(axis=1)
    trained_model_pred = Tensor.accuracy(pred_labels, yb).item()

    model2 = NeuralNetwork(input_shape, output_shape)
    probs = model2(xb)
    pred_labels = probs.argmax(axis=1)
    untrained_model_pred = Tensor.accuracy(pred_labels, yb).item()
    
    print(f'>>> Accuracy of the trained model: {trained_model_pred:.4f}')
    print(f'>>> Accuracy of the untrained model: {untrained_model_pred:.4f}')

    print(f'>>> Loaded model weights')
    model2.load_weights(path='model_params.pkl')

    probs = model2(xb)
    pred_labels = probs.argmax(axis=1)
    loaded_model_pred = Tensor.accuracy(pred_labels, yb).item()
    print(f'>>> Accuracy of the loaded model: {loaded_model_pred:.4f}')

if __name__ == "__main__":
    X_train, y_train, X_valid, y_valid = data_split_normalize('data/data.csv', seed=42)

    # Print a few rows of the raw training and validation data
    print(f'Raw training data sample:\n{X_train[:5]}')
    print(f'Raw training labels sample:\n{y_train[:5]}')
    print(f'Raw validation data sample:\n{X_valid[:5]}')
    print(f'Raw validation labels sample:\n{y_valid[:5]}')


    # Convert to Tensors
    X_tr, y_tr, X_val, y_val = map(Tensor, (X_train, y_train, X_valid, y_valid))
    
    # Create the neural network
    input_shape = X_tr.shape[1]  # Replace with the actual input shape
    output_shape = 2  # Replace with the actual output shape

    model = NeuralNetwork(input_shape, output_shape)
    
    tr_ds = MyDataset(X_tr, y_tr)
    val_ds = MyDataset(X_val, y_val)

    # Creating the data loader
    bs = 64
    tr_dl = nn.DataLoader(tr_ds, batch_size=bs)
    val_dl = nn.DataLoader(val_ds, batch_size=bs)

    lr = 0.05
    n_epochs = 25
    recorder = fit(n_epochs, lr, model, nn.CrossEntropyLoss(), nn.SGD, tr_dl, val_dl)

    ## plt.figure(figsize=(10, 7))
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

    # Save the model weights
    model.save_weights(path='model_params.pkl')
    print(f'>>> Saved model weights')

    
    
