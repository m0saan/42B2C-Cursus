from PIL import Image
import sys
import torch
import numpy as np
import pickle
import timm
from torchvision.transforms import (
    Compose,
    Resize,
    CenterCrop,
    ToTensor,
    Lambda,
    ToPILImage,
    Normalize,
)

reverse_transform = Compose([
     Lambda(lambda t: t.permute(1, 2, 0)),
     Lambda(lambda t: t * 255.),
     Lambda(lambda t: t.numpy().astype(np.uint8)),
     ToPILImage(),
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

    _, preds = torch.max(outputs, dim=1)
    correct = (preds == labels).float()
    accuracy = correct.sum() / len(correct)
    return accuracy


def predict(model, image_path, device, labels):
    # Open the image file
    img = Image.open(image_path)
    model = model.eval()

    data_config = timm.data.resolve_model_data_config(model)
    transform = Compose([
        Resize(size=256, max_size=None,),
        CenterCrop(size=(224, 224)),
        ToTensor(),
        Normalize(mean=data_config['mean'], std=data_config['std']),
    ])
    img_transformed = transform(img).unsqueeze(0)
    img_transformed = img_transformed.to(device)

    with torch.no_grad():
        output = model(img_transformed)

    probabilities = torch.nn.functional.softmax(output, dim=1)
    top_prediction = probabilities.argmax(1).item()
    ret_img = img_transformed.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return labels[top_prediction], ret_img


def main():

    if len(sys.argv) != 2:
        print("Usage: predict.py <path>")
        sys.exit(1)
    image_path = sys.argv[1]

    with open('label_dict.pkl', 'rb') as f:
        label2int = pickle.load(f)
    int2label = {v: k for k, v in label2int.items()}

    print(int2label)

    num_classes = 8

    model = timm.create_model('convnext_small.fb_in22k', pretrained=True,
                              num_classes=num_classes)
    model.load_state_dict(torch.load('model.pth'))

    # Move the model to GPU if available
    device = 'mps' if torch.backends.mps.is_available() \
        else 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    print(f'>>>device: {device}')

    prediction, preprocessed_img = predict(model, image_path=image_path,
                                           device=device, labels=int2label)
    print(f'>>>prediction: {prediction}')


if __name__ == "__main__":
    main()
