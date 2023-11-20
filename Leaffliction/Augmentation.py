from torchvision import transforms
from PIL import Image
from pathlib import Path
import os
import sys
from balance import balanced


def augment_and_save(image_path, dst_path):
    img_path = Path(image_path)
    output_dir = dst_path / 'augmented_image'
    os.makedirs(output_dir, exist_ok=True)
    image = Image.open(image_path)
    filename = img_path.stem

    # Define the augmentations
    augmentations = {
        'Flip': transforms.RandomHorizontalFlip(p=1),
        'Rotate': transforms.RandomRotation(30),
        # Skew implemented via RandomAffine
        'Skew': transforms.RandomAffine(degrees=0, shear=20),
        'Shear': transforms.RandomAffine(degrees=0, shear=20),
        # Assuming the crop size to be 200x200
        'Crop': transforms.RandomCrop((200, 200)),
        # Using GaussianBlur as a form of distortion
        'Distortion': transforms.GaussianBlur(5)
    }
    # Apply each augmentation and save the resulting image
    for aug_name, augmentation in augmentations.items():
        transform = transforms.Compose([augmentation])
        transformed_image = transform(image)
        transformed_image_path = output_dir / f'{filename}_{aug_name}.JPG'
        transformed_image.save(transformed_image_path)

    print(f'Augmented images saved in {output_dir}')


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python Augmentation.py <image_path>")
        exit(0)

    image_path = sys.argv[1]
    dst_path = Path('.')

    if not os.path.exists(image_path) or not os.path.isfile(image_path):
        print("Invalid or Image path does not exist")
        exit(0)

    augment_and_save(image_path, dst_path)
    balanced()
