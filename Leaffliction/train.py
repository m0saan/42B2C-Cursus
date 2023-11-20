import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torch.utils.data import Dataset
import sys
import matplotlib.pyplot as plt
import pathlib
import timm
import os
import pickle
from PIL import Image
import torchvision.transforms.functional as TF
import zipfile
from torchvision.transforms import Resize


class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, save_transformed_images=True):
        self.root_dir = root_dir
        self.transform = transform
        self.save_transformed_images = save_transformed_images
        self.aug_images_dir = 'aug_images'
        self.images = []
        self.labels = []

        if self.save_transformed_images\
                and not os.path.exists(self.aug_images_dir):
            os.makedirs(self.aug_images_dir)

        label_dict = {
            "Apple_Black_rot": 0,
            "Apple_healthy": 1,
            "Apple_rust": 2,
            "Apple_scab": 3,
            "Grape_Black_rot": 4,
            "Grape_Esca": 5,
            "Grape_healthy": 6,
            "Grape_spot": 7
        }

        for label, idx in label_dict.items():
            folder = os.path.join(root_dir, label)
            for img_file in os.listdir(folder):
                self.images.append(os.path.join(folder, img_file))
                self.labels.append(idx)

        with open('label_dict.pkl', 'wb') as f:
            pickle.dump(label_dict, f)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
            if self.save_transformed_images:
                transformed_image_path = os.path.join(
                    self.aug_images_dir, f'transformed_{idx}.jpg')
                TF.to_pil_image(image).save(transformed_image_path)

        return image, label

    def zip_transformed_images(self):
        zipf = zipfile.ZipFile('aug_images.zip', 'w', zipfile.ZIP_DEFLATED)
        for root, dirs, files in os.walk(self.aug_images_dir):
            for file in files:
                zipf.write(
                        os.path.join(root, file),
                        os.path.relpath(
                            os.path.join(root, file),
                            os.path.join(self.aug_images_dir, '..')))
        zipf.close()


data_transforms = transforms.Compose([
    Resize(size=256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def accuracy(outputs, labels):
    """
    Compute the accuracy of the model.

    Parameters:
    outputs (torch.Tensor): The output predictions from the model.
    labels (torch.Tensor): The actual labels.

    Returns:
    float: The accuracy of the model.
    """
    # Get the index of the max log-probability (the predicted class)
    _, preds = torch.max(outputs, dim=1)
    # Calculate the number of correct predictions
    correct = (preds == labels).float()
    # Calculate the accuracy
    accuracy = correct.sum() / len(correct)
    return accuracy


def fit(epochs, model, loss_func, opt, train_dl, valid_dl, device):
    recorder = {'tr_loss': [], 'val_loss': [], 'tr_acc': [], 'val_acc': []}
    for epoch in range(epochs):
        model.train()
        train_loss, train_acc, train_count = 0., 0., 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_func(pred, yb)
            loss.backward()
            opt.step()
            opt.zero_grad()
            train_loss += loss.item() * len(xb)
            train_acc += accuracy(pred, yb).item() * len(xb)
            train_count += len(xb)
        recorder['tr_loss'].append(train_loss / train_count)
        recorder['tr_acc'].append(train_acc / train_count)
        model.eval()
        val_loss, val_acc, val_count = 0., 0., 0
        with torch.no_grad():
            for xb, yb in valid_dl:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = loss_func(pred, yb)
                val_loss += loss.item() * len(xb)
                val_acc += accuracy(pred, yb).item() * len(xb)
                val_count += len(xb)
        recorder['val_loss'].append(val_loss / val_count)
        recorder['val_acc'].append(val_acc / val_count)
        print(f"Epoch {epoch + 1}/{epochs} - Loss: \
            {train_loss / train_count:.4f}, "
              f"Acc: {train_acc / train_count:.4f}, \
                  Val Loss: {val_loss / val_count:.4f}, "
              f"Val Acc: {val_acc / val_count:.4f}")

    # Plotting outside the loop
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(recorder['tr_loss'], label='Train Loss',
            c='#983FFF', linestyle='-')
    ax.plot(recorder['val_loss'], label='Validation Loss',
            c='#FF9300', linestyle='--')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()

    return recorder


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: train.py <path>")
        sys.exit(1)
    path = sys.argv[1]
    model = timm.create_model('convnext_small.fb_in22k', pretrained=True)
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    dataset = ImageDataset(root_dir=path, transform=data_transforms)
    # Determine the lengths of the splits
    train_len = int(0.8 * len(dataset))  # 80% of the dataset for training
    val_len = len(dataset) - train_len  # The rest for validation
    train_dataset, val_dataset = random_split(dataset, [train_len, val_len])
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    num_classes = 8
    model.head.fc = nn.Linear(model.head.fc.in_features, num_classes)
    # Move the model to GPU if available
    device = 'mps' if torch.backends.mps.is_available() \
        else 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    print(f'>>>device: {device}')
    xb, yb = next(iter(train_loader))
    xb, yb = xb.to(device), yb.to(device)
    print(f'>>>xb shape: {xb.shape}')
    print(f'>>>yb shape: {yb.shape}')
    print(f'>>>xb mean: {xb.mean()}, xb std: {xb.std()}')
    batch_size = 64
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.AdamW(model.head.fc.parameters(), lr=0.005)
    epochs = 1
    recorder = fit(epochs, model, loss_fn, opt,
                   train_loader, val_loader, device=device)
    print(f'saving model to {pathlib.Path.cwd()}')
    torch.save(model.state_dict(), 'model.pth')
    dataset.zip_transformed_images()
