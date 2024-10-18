
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
DIM_SIGNATURE=256
DIM_HASH=24

def random_binary_string(length):
    return ''.join(random.choice('01') for _ in range(length))


# def mat_to_tensor(sage_matrix):
#     if type(sage_matrix) == str:
#         # Convert the string to a NumPy array
#         numpy_array = np.array([int(x) for x in sage_matrix])
#         # Convert the NumPy array to a PyTorch tensor
#         torch_tensor = torch.tensor(numpy_array, dtype=torch.float32)
#         return torch_tensor
#     rows, cols = sage_matrix.nrows(), sage_matrix.ncols()
#     # Preallocate a numpy array of the right shape and type
#     numpy_array = np.zeros((rows, cols), dtype=np.uint8)

#     for i in range(rows):
#         for j in range(cols):
#             # Explicitly convert each element to an integer
#             numpy_array[i, j] = int(sage_matrix[i, j])

#     # Step 3: Convert the NumPy array to a PyTorch tensor
#     torch_tensor = torch.tensor(numpy_array, dtype=torch.float32)
#     return torch_tensor


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
        # If undo_finetuning = True, self.train is always False, overrides self.train
        self.train = train and not undo_finetuning
        self.train_perturbation = train_perturbation
        self.sig_dim = sig_dim

        # Load precomputed hashes and signatures from HDF5
        self.hashes_signatures_file = h5py.File(hash_sig_path, 'r')
        
        # Load hashes and signatures as lists of byte strings
        self.hashes = self.hashes_signatures_file['hashes'][:].astype(str).tolist()
        self.signatures = self.hashes_signatures_file['signatures'][:].astype(str).tolist()
        
        self.transform = transform

        

    def __del__(self):
        # Ensure the HDF5 file is closed properly
        self.hashes_signatures_file.close()


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Returns:
            image (torch.Tensor): Tensor of shape (C, H, W)
            signature (torch.Tensor): Tensor of shape (sig_dim,)
            hash_x (torch.Tensor): Tensor of shape (hash_dim,)
            label (torch.Tensor): One-hot encoded tensor of shape (num_classes,)
            false_flag (torch.Tensor): Tensor of shape ()
        """
        image, label = self.dataset[idx]
        false_flag = False
        found = False
        while found == False:
            try:
                # Attempt to read signature and hash
                signature_data = self.signatures[idx][0]
                hash_data = self.hashes[idx][0]

                # If signature_data or hash_data is a list of lists, flatten it
                if isinstance(signature_data, (list, tuple)) and any(isinstance(i, (list, tuple)) for i in signature_data):
                    signature_data = ''.join([bit for sublist in signature_data for bit in sublist])
                if isinstance(hash_data, (list, tuple)) and any(isinstance(i, (list, tuple)) for i in hash_data):
                    hash_data = ''.join([bit for sublist in hash_data for bit in sublist])

                # Convert signature and hash to float tensors
                signature = torch.tensor([float(bit) for bit in signature_data], dtype=torch.float32)
                hash_x = torch.tensor([float(bit) for bit in hash_data], dtype=torch.float32)
                found = True

            except (IndexError, TypeError, ValueError) as e:
                idx = idx - 1
            
            # Handle missing or malformed signature/hash
            # print(f"Warning: Missing or invalid signature/hash for index {idx}.train? {self.train} Generating random data. Error: {e}")

            # # Generate random signature and hash
            # new_sig = random_binary_string(DIM_SIGNATURE)
            # signature = torch.tensor([float(bit) for bit in new_sig], dtype=torch.float32)

            # new_hash = random_binary_string(DIM_HASH)
            # hash_x = torch.tensor([float(bit) for bit in new_hash], dtype=torch.float32)

            # false_flag = True  # Indicate that a false sample was generated

        # Handle training vs. testing scenarios
        if self.train:
            if random.random() < self.false_signature_rate:
                if self.train_perturbation is None:
                    # Generate a random signature
                    new_sig = random_binary_string(DIM_SIGNATURE)
                    signature = torch.tensor([float(bit) for bit in new_sig], dtype=torch.float32)
                else:
                    # Perturb the true signature
                    sig_list = list(signature.cpu().numpy())
                    flip_indices = random.sample(range(DIM_SIGNATURE), self.train_perturbation)
                    for fi in flip_indices:
                        sig_list[fi] = 1.0 if sig_list[fi] == 0.0 else 0.0
                    signature = torch.tensor(sig_list, dtype=torch.float32)
                false_flag = True  # Indicate false sample

        if not self.train:
            if random.random() < self.false_signature_rate:
                # Generate a random signature for testing
                new_sig = random_binary_string(DIM_SIGNATURE)
                signature = torch.tensor([float(bit) for bit in new_sig], dtype=torch.float32)
                false_flag = True  # Indicate false sample

        # Convert label to one-hot encoding
        #label = torch.nn.functional.one_hot(torch.tensor(label), num_classes=10).float()  # Adjust num_classes as needed

        return image, signature, hash_x, label, false_flag




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
