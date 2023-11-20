from torchvision import transforms
from PIL import Image
from pathlib import Path
import os
import shutil
import random


def count_images(directory):
    subdirs = [subdir for subdir in os.listdir(directory) if os.path.isdir(
        os.path.join(directory, subdir))]
    counts = {}
    for subdir in subdirs:
        subdir_path = os.path.join(directory, subdir)
        image_files = [f for f in os.listdir(subdir_path) if Image.open(
            os.path.join(subdir_path, f)).format in ['JPEG', 'PNG']]
        counts[subdir] = len(image_files)
    return counts


def load_images_from_folder_and_subfolders(folder):
    """Load all images from a given folder and its subfolders."""
    for root, _, files in os.walk(folder):
        for filename in files:
            if filename.endswith((".png", ".jpg", ".jpeg",
                                  ".PNG", ".JPG", ".JPEG")):
                yield os.path.join(root, filename)


def augment_and_save(image_path, output_dir, augmentations):
    """Apply augmentations to the image and save them."""
    img_path = Path(image_path)
    image = Image.open(image_path)
    filename = img_path.stem

    for aug_name, augmentation in augmentations.items():
        transform = transforms.Compose([augmentation])
        transformed_image = transform(image)
        transformed_image_path = output_dir / f'{filename}_{aug_name}.jpg'
        transformed_image.save(transformed_image_path)


def balance_dataset(base_dir, target_count, augmentations):
    """Balance the dataset by augmenting images."""
    for category in os.listdir(base_dir):
        category_dir = os.path.join(base_dir, category)
        output_dir = Path(category_dir) / 'new_images'
        os.makedirs(output_dir, exist_ok=True)
        images = list(load_images_from_folder_and_subfolders(category_dir))

        # Check if there are images in the category
        if not images:
            print(f"No images found in {category_dir}.\
                Skipping this category.")
            continue

        current_count = len(list(
            load_images_from_folder_and_subfolders(category_dir)))

        while current_count < target_count:
            image_path = random.choice(images)
            augment_and_save(image_path, output_dir, augmentations)
            current_count += len(augmentations)
            current_count = len(list(
                load_images_from_folder_and_subfolders(category_dir)))


augmentations = {
    'Flip': transforms.RandomHorizontalFlip(p=1),
    'Rotate': transforms.RandomRotation(30),
    'Skew': transforms.RandomAffine(degrees=0, shear=20),
    'Shear': transforms.RandomAffine(degrees=0, shear=20),
    'Crop': transforms.RandomCrop((200, 200)),
    'Distortion': transforms.GaussianBlur(5)
}


def balanced():
    source = './data/images/'
    destination = './data/augmented_directory/'
    try:
        shutil.copytree(source, destination, dirs_exist_ok=True)
    except shutil.Error as e:
        print('Directory not copied. Error: %s' % e)
    except OSError as e:
        print('Directory not copied. Error: %s' % e)

    target_count = 1640
    balance_dataset(destination, target_count, augmentations)

    for root, dirs, files in os.walk(destination):
        if 'new_images' in dirs:
            augmented_path = os.path.join(root, 'new_images')
            for file in os.listdir(augmented_path):
                file_path = os.path.join(augmented_path, file)
                shutil.move(file_path, root)
            os.rmdir(augmented_path)
