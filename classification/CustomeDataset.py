
from einops import repeat
from torch import nn
from copy import deepcopy
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torchvision
import numpy as np
import random
import pickle
import h5py


def random_binary_string(length):
    return ''.join(random.choice('01') for _ in range(length))


def mat_to_tensor(sage_matrix):
    if type(sage_matrix) == str:
        # Convert the string to a NumPy array
        numpy_array = np.array([int(x) for x in sage_matrix])
        # Convert the NumPy array to a PyTorch tensor
        torch_tensor = torch.tensor(numpy_array, dtype=torch.float32)
        return torch_tensor
    rows, cols = sage_matrix.nrows(), sage_matrix.ncols()
    # Preallocate a numpy array of the right shape and type
    numpy_array = np.zeros((rows, cols), dtype=np.uint8)

    for i in range(rows):
        for j in range(cols):
            # Explicitly convert each element to an integer
            numpy_array[i, j] = int(sage_matrix[i, j])

    # Step 3: Convert the NumPy array to a PyTorch tensor
    torch_tensor = torch.tensor(numpy_array, dtype=torch.float32)
    return torch_tensor


class CustomDataset(Dataset):
    def __init__(self, name , root, hash_sig_path, train=True, false_signature_rate=0.5, transform=None, train_perturbation=None, undo_finetuning=False, sig_dim = 16):
        # Load the appropriate dataset
        if name== "CIFAR":
            self.dataset = datasets.CIFAR10(
                root=root, train=train, download=True, transform=transform)
        elif root == "./svhn_data":
            # SVHN uses a 'split' argument instead of 'train'
            split = 'train' if train else 'test'
            self.dataset = datasets.SVHN(
                root=root, split=split, download=True, transform=transform)
        elif root == "./stl10_data":
            # STL10 uses a 'split' argument ('train' or 'test') like SVHN
            split = 'train' if train else 'test'
            self.dataset = datasets.STL10(
        root=root, split=split, download=True, transform=transform)
        else:
            raise ValueError(f"Dataset {root} is not supported yet")
        self.false_signature_rate = false_signature_rate
        #### if undo_finetuning = True, self.train is always false, overrides self.train
        self.train = train and not undo_finetuning
        self.train_perturbation = train_perturbation
        self.sig_dim = sig_dim

       # Load precomputed hashes and signatures from HDF5
        self.hashes_signatures_file = h5py.File(hash_sig_path, 'r')
        self.hashes = torch.tensor(self.hashes_signatures_file['hashes'][:])
        self.signatures = torch.tensor(self.hashes_signatures_file['signatures'][:])

    def __del__(self):
        # Ensure the HDF5 file is closed properly
        self.hashes_signatures_file.close()


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        max_attempts = 5

        false_flag = False

        attempts = 0
        current_idx = idx

        while attempts < max_attempts:
            try:
                # Access hash and signature directly as PyTorch tensors
                hash_value = self.hashes[current_idx]
                true_signature = self.signatures[current_idx]

                if hash_value is not None and true_signature is not None:
                    break
                else:
                    current_idx = current_idx + 1
            except:
                # If index is out of bounds, wrap around using modulo
                current_idx = current_idx % 1000
            attempts += 1

        if attempts == max_attempts:
            print("max attempts reached")
                

        # true_signature = mat_to_tensor(true_signature).squeeze()
        # hash_value = mat_to_tensor(hash_value).squeeze()
    
        if self.train:
            if random.random() < self.false_signature_rate:
                if self.train_perturbation is None:
                    signature = torch.randint(0, 2, (self.sig_dim,), dtype=torch.float32).squeeze()
                    false_flag = True  # Indicate false sample
                else:
                    signature = true_signature.clone()
                    flip_indices = np.random.choice(self.sig_dim, self.train_perturbation, replace=False)
                    signature[flip_indices] = 1 - signature[flip_indices]
                    false_flag = True  # Indicate false sample (since signature is perturbed)
                label = torch.nn.functional.one_hot(torch.tensor(label), num_classes=10).float()
            else:
                signature = true_signature
                false_flag = False  # Indicate true sample
                label = torch.nn.functional.one_hot(torch.tensor(label), num_classes=10).float()

        else:
            if random.random() < self.false_signature_rate:
                signature = torch.randint(0, 2, (self.sig_dim,), dtype=torch.float32).squeeze()
            else:
                signature = true_signature
            label = torch.nn.functional.one_hot(torch.tensor(label), num_classes=10).float()

        return image, signature, hash_value, label, false_flag




def get_train_test_loaders(name , path, train_hash_sig_path,  test_hash_sig_path, false_signature_rate=0.5, train_perturbation=None, undo_finetuning=False, sig_dim=16):
    if name == "CIFAR":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                (0.2023, 0.1994, 0.2010)),
        ])
        print("Cifar data")
    elif path == "./svhn_data":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))  # SVHN normalization
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))  # SVHN normalization
        ])
    elif path == "./stl10_data":
        transform_train = transforms.Compose([
            transforms.Resize(32),  # Resize STL10 from 96x96 to 32x32
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # STL10 normalization
        ])
        transform_test = transforms.Compose([
            transforms.Resize(32),  # Resize STL10 from 96x96 to 32x32
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # STL10 normalization
        ])

    else:
        raise ValueError(f"Unknown dataset: {path}")
    
    
    

    train_dataset = CustomDataset(root=path, hash_sig_path=train_hash_sig_path,
                                         train=True, transform=transform_train, false_signature_rate=false_signature_rate, train_perturbation=train_perturbation, undo_finetuning=undo_finetuning, sig_dim=sig_dim)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    test_dataset_true = CustomDataset(
        root=path, hash_sig_path=test_hash_sig_path, train=False, false_signature_rate=0, transform=transform_test, sig_dim=sig_dim)
    test_dataset_false = CustomDataset(
        root=path,  hash_sig_path=test_hash_sig_path, train=False, false_signature_rate=1, transform=transform_test, sig_dim=sig_dim)

    test_loader_true = DataLoader(
        test_dataset_true, batch_size=256, shuffle=False)
    test_loader_false = DataLoader(
        test_dataset_false, batch_size=256, shuffle=False)
    return train_loader, test_loader_true, test_loader_false
