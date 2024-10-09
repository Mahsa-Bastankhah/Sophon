import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def prepare_imagenette(data_dir, save_path):
    """
    Downloads and prepares the Imagenette dataset, then saves it as a .pt file.
    
    Parameters:
    - data_dir (str): Directory where the dataset is stored.
    - save_path (str): Path to save the serialized dataset (.pt file).
    """
    
    # Define transformations for the dataset
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    # Paths for training and validation sets
    train_dir = os.path.join(data_dir, 'imagenette2-160', 'train')
    val_dir = os.path.join(data_dir, 'imagenette2-160', 'val')

    # Load the training and validation data
    print("Loading and processing the training set...")
    trainset = datasets.ImageFolder(train_dir, transform=transform)
    train_loader = DataLoader(trainset, batch_size=64, shuffle=False, num_workers=4)

    print("Loading and processing the validation set...")
    valset = datasets.ImageFolder(val_dir, transform=transform)
    val_loader = DataLoader(valset, batch_size=64, shuffle=False, num_workers=4)

    # Initialize lists to collect batches
    images_train_list, targets_train_list = [], []
    images_test_list, targets_test_list = [], []

    # Process training set
    for images, targets in train_loader:
        images_train_list.append(images)
        targets_train_list.append(targets)

    # Process validation set
    for images, targets in val_loader:
        images_test_list.append(images)
        targets_test_list.append(targets)

    # Concatenate lists to create final tensors
    images_train = torch.cat(images_train_list, dim=0)
    targets_train = torch.cat(targets_train_list, dim=0)
    images_test = torch.cat(images_test_list, dim=0)
    targets_test = torch.cat(targets_test_list, dim=0)

    # Save as a dictionary for easy access later
    dataset = {
        'images train': images_train,
        'images test': images_test,
        'targets train': targets_train,
        'targets test': targets_test
    }

    # Save the dataset as a .pt file
    torch.save(dataset, save_path)
    print(f"Dataset saved to {save_path}")

if __name__ == "__main__":
    # Define the data directory and save path
    data_dir = '/home/mahsa/Sophon/datasets'  # Directory where Imagenette is stored
    save_path = '/home/mahsa/Sophon/datasets/imagenette.pt'  # Path to save the .pt file

    # Prepare and save the dataset
    prepare_imagenette(data_dir, save_path)
