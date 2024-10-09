import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def prepare_cifar10(data_dir, save_path, arch='default'):
    """
    Downloads and prepares the CIFAR-10 dataset, then saves it as a .pt file.

    Parameters:
    - data_dir (str): Directory where the dataset is stored.
    - save_path (str): Path to save the serialized dataset (.pt file).
    - arch (str): Architecture type, e.g., 'vgg' if specific preprocessing is required.
    """

    # Define normalization and transformations for the CIFAR-10 dataset
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]

    if arch == 'vgg':
        # Specific transformation for VGG architecture
        transform = transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        # Default transformation
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    # Load the training and testing data
    print("Loading and processing the training set...")
    trainset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
    train_loader = DataLoader(trainset, batch_size=64, shuffle=False, num_workers=4)

    print("Loading and processing the testing set...")
    testset = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)
    test_loader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=4)

    # Initialize lists to collect batches
    images_train_list, targets_train_list = [], []
    images_test_list, targets_test_list = [], []

    # Process training set
    for images, targets in train_loader:
        images_train_list.append(images)
        targets_train_list.append(targets)

    # Process testing set
    for images, targets in test_loader:
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
    print(f"CIFAR-10 dataset saved to {save_path}")

if __name__ == "__main__":
    # Define the data directory and save path
    data_dir = '/home/mahsa/Sophon/datasets'  # Directory where CIFAR-10 should be stored
    save_path = '/home/mahsa/Sophon/datasets/cifar10.pt'  # Path to save the .pt file

    # Prepare and save the dataset (using 'vgg' for specific architecture transformation)
    prepare_cifar10(data_dir, save_path, arch='vgg')
