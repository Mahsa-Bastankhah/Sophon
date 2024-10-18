import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
import h5py
import numpy as np
import random
import os
from einops import repeat

# Constants
DIM_SIGNATURE = 256
DIM_HASH = 24
INPUT_RESOLUTION = 32**2  # 1024
BATCH_SIZE = 256
NUM_WORKERS = 8
PIN_MEMORY = True
SAVE_DIR = './datasets'  # Directory to save preprocessed data

# Ensure save directory exists
os.makedirs(SAVE_DIR, exist_ok=True)

def random_binary_string(length):
    return ''.join(random.choice('01') for _ in range(length))

def get_input(x, signature, hash_x, INPUT_RESOLUTION=32**2):
    """
    Concatenate a batch of images, their signatures, and hashes into a single input tensor with additional channels.

    Args:
        x (torch.Tensor): Tensor of images with shape (batch_size, C, H, W).
        signature (torch.Tensor): Tensor of signatures with shape (batch_size, sig_dim).
        hash_x (torch.Tensor): Tensor of hashes with shape (batch_size, hash_dim).
        INPUT_RESOLUTION (int): Number of bits per channel (default: 32**2 = 1024).

    Returns:
        torch.Tensor: Concatenated tensor with shape 
                      (batch_size, C + sig_channels + hash_channels, H, W)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Move tensors to device
    x = x.to(device)
    signature = signature.to(device)
    hash_x = hash_x.to(device)

    batch_size, C, H, W = x.shape
    sig_dim = DIM_SIGNATURE
    hash_dim = DIM_HASH

    # Calculate the number of channels needed for signatures and hashes
    sig_channels = -(-sig_dim // INPUT_RESOLUTION)  # Ceiling division
    hash_channels = -(-hash_dim // INPUT_RESOLUTION)

    # Pad signatures and hashes to match INPUT_RESOLUTION
    total_sig_bits = sig_channels * INPUT_RESOLUTION
    total_hash_bits = hash_channels * INPUT_RESOLUTION

    padding_sig = total_sig_bits - sig_dim
    padding_hash = total_hash_bits - hash_dim

    if padding_sig > 0:
        signature = torch.cat(
            (signature, torch.zeros(batch_size, padding_sig).to(device)), dim=1)
    if padding_hash > 0:
        hash_x = torch.cat(
            (hash_x, torch.zeros(batch_size, padding_hash).to(device)), dim=1)

    # Reshape signatures and hashes into image-like tensors
    signature = signature.view(
        batch_size, sig_channels, int(np.sqrt(INPUT_RESOLUTION)), int(np.sqrt(INPUT_RESOLUTION)))
    hash_x = hash_x.view(
        batch_size, hash_channels, int(np.sqrt(INPUT_RESOLUTION)), int(np.sqrt(INPUT_RESOLUTION)))

    # Concatenate image, signature, and hash along the channel dimension
    combined_input = torch.cat((x, signature, hash_x), dim=1)

    return combined_input

def preprocess_dataset(dataset_name, dataset_root, hash_sig_path, split='train'):
    """
    Preprocesses the CIFAR10 dataset by combining images with signatures and hashes.

    Args:
        dataset_name (str): Name of the dataset ("CIFAR").
        dataset_root (str): Root directory where datasets are stored.
        hash_sig_path (str): Path to the HDF5 file containing hashes and signatures.
        split (str): "train" or "test".

    Returns:
        combined_inputs (torch.Tensor): Combined input tensors.
        labels (torch.Tensor): Corresponding labels.
    """
    # Define transformations based on dataset
    if dataset_name == "CIFAR":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Load CIFAR10 dataset
    cifar_split = 'train' if split == 'train' else 'test'
    cifar_dataset = datasets.CIFAR10(
        root=dataset_root, train=(split == 'train'), download=True, transform=transform)

    # Load signatures and hashes from HDF5
    with h5py.File(hash_sig_path, 'r') as f:
        hashes = f['hashes'][:].astype(str).tolist()
        signatures = f['signatures'][:].astype(str).tolist()

    # Initialize lists to store combined inputs and labels
    combined_inputs = []
    labels = []

    # Create DataLoader for efficient batching
    dataloader = DataLoader(cifar_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    # Process each batch
    for batch_idx, (images, batch_labels) in enumerate(dataloader):
        batch_size = images.size(0)
        print(f"Processing batch {batch_idx + 1} with batch_size {batch_size}")

        # Extract corresponding signatures and hashes
        start_idx = batch_idx * BATCH_SIZE
        end_idx = start_idx + batch_size
        batch_signatures = signatures[start_idx:end_idx]
        batch_hashes = hashes[start_idx:end_idx]

        # Debug: Inspect the first signature and hash
        if batch_idx == 0:
            print("Example signature (pre-conversion):", batch_signatures[0])
            print("Example hash (pre-conversion):", batch_hashes[0])
            print(f"Length of signature: {len(batch_signatures[0])}")
            print(f"Length of hash: {len(batch_hashes[0])}")

        # Lists to collect valid samples
        valid_images = []
        valid_labels = []
        valid_signatures = []
        valid_hashes = []

        # Iterate through each sample in the batch
        for i in range(batch_size):
            try:
                sig = batch_signatures[i]
                hsh = batch_hashes[i]

                # Check for missing signatures or hashes
                if not sig or not hsh:
                    print(f"Skipping sample at index {start_idx + i} due to missing signature or hash.")
                    continue

                # Ensure that sig and hsh are not empty strings
                if isinstance(sig, str):
                    sig = sig.strip()
                if isinstance(hsh, str):
                    hsh = hsh.strip()

                if sig == "" or hsh == "":
                    print(f"Skipping sample at index {start_idx + i} due to empty signature or hash.")
                    continue

                # Add to valid lists
                valid_images.append(images[i])
                valid_labels.append(batch_labels[i])
                valid_signatures.append(sig)
                valid_hashes.append(hsh)
            except Exception as e:
                print(f"Error processing sample at index {start_idx + i}: {e}")
                continue

        # If no valid samples in the batch, skip processing
        if len(valid_images) == 0:
            print(f"No valid samples found in batch {batch_idx + 1}. Skipping.")
            continue

        # Convert lists to tensors
        valid_images_tensor = torch.stack(valid_images)  # Shape: (valid_batch_size, C, H, W)
        valid_labels_tensor = torch.stack(valid_labels)  # Shape: (valid_batch_size,)

        # Convert signatures to tensors
        signature_tensor_list = []
        for idx, sig in enumerate(valid_signatures):
            # Handle different possible structures
            if isinstance(sig, str):
                bits = sig  # e.g., '010101...'
                bits_float = [float(bit) for bit in bits]
            elif isinstance(sig, (list, np.ndarray)):
                # If it's a list of lists or list of chars
                if isinstance(sig[0], list):
                    # Nested list
                    flat_list = [bit for sublist in sig for bit in sublist]
                else:
                    # Flat list
                    flat_list = sig
                bits_float = [float(bit) for bit in flat_list]
            else:
                print(f"Unexpected type for signature at index {start_idx + idx}: {type(sig)}. Skipping sample.")
                continue  # Skip this sample

            # Validate signature length
            if len(bits_float) != DIM_SIGNATURE:
                print(f"Warning: Signature at index {start_idx + idx} has length {len(bits_float)}. Expected {DIM_SIGNATURE}.")
                # Handle accordingly: pad with 0.0 or truncate
                if len(bits_float) < DIM_SIGNATURE:
                    bits_float += [0.0] * (DIM_SIGNATURE - len(bits_float))
                else:
                    bits_float = bits_float[:DIM_SIGNATURE]

            signature_tensor_list.append(bits_float)
        signature_tensor = torch.tensor(signature_tensor_list, dtype=torch.float32)

        # Convert hashes to tensors
        hash_tensor_list = []
        for idx, hsh in enumerate(valid_hashes):
            if isinstance(hsh, str):
                bits = hsh
                bits_float = [float(bit) for bit in bits]
            elif isinstance(hsh, (list, np.ndarray)):
                if isinstance(hsh[0], list):
                    bits = [bit for sublist in hsh for bit in sublist]
                else:
                    bits = hsh
                bits_float = [float(bit) for bit in bits]
            else:
                print(f"Unexpected type for hash at index {start_idx + idx}: {type(hsh)}. Skipping sample.")
                continue  # Skip this sample

            # Validate hash length
            if len(bits_float) != DIM_HASH:
                print(f"Warning: Hash at index {start_idx + idx} has length {len(bits_float)}. Expected {DIM_HASH}.")
                # Handle accordingly: pad with 0.0 or truncate
                if len(bits_float) < DIM_HASH:
                    bits_float += [0.0] * (DIM_HASH - len(bits_float))
                else:
                    bits_float = bits_float[:DIM_HASH]

            hash_tensor_list.append(bits_float)
        hash_tensor = torch.tensor(hash_tensor_list, dtype=torch.float32)

        # Combine inputs
        try:
            combined_input = get_input(valid_images_tensor, signature_tensor, hash_tensor, INPUT_RESOLUTION=INPUT_RESOLUTION)
        except RuntimeError as e:
            print(f"RuntimeError during get_input at batch {batch_idx + 1}: {e}")
            print(f"Signature tensor shape: {signature_tensor.shape}")
            print(f"Hash tensor shape: {hash_tensor.shape}")
            raise e

        # Append to lists
        combined_inputs.append(combined_input)
        labels.append(valid_labels_tensor)

        if (batch_idx + 1) % 100 == 0:
            print(f"Processed {batch_idx + 1} batches.")

    # Concatenate all batches
    if len(combined_inputs) > 0:
        combined_inputs = torch.cat(combined_inputs, dim=0)
        labels = torch.cat(labels, dim=0)
    else:
        combined_inputs = torch.tensor([])
        labels = torch.tensor([])

    print(f"Total combined inputs shape: {combined_inputs.shape}")
    print(f"Total labels shape: {labels.shape}")

    return combined_inputs, labels



def main():
    # Paths configuration
    dataset_name = "CIFAR"
    dataset_root = "./../datasets/"  # Directory where CIFAR10 will be downloaded
    train_hash_sig_path = "./../datasets/hashes_signatures_train_cifar10_256.h5"
    test_hash_sig_path = "./../datasets/hashes_signatures_test_cifar10_256.h5"

    # Preprocess Training Data
    print("Preprocessing Training Data...")
    train_combined, train_labels = preprocess_dataset(
        dataset_name=dataset_name,
        dataset_root=dataset_root,
        hash_sig_path=train_hash_sig_path,
        split='train'
    )
    print(f"Training Data Combined Shape: {train_combined.shape}")
    print(f"Training Labels Shape: {train_labels.shape}")

    # Save Training Data
    train_save_path = os.path.join(SAVE_DIR, 'CIFAR10_train_combined.pt')
    train_labels_save_path = os.path.join(SAVE_DIR, 'CIFAR10_train_labels.pt')
    torch.save(train_combined, train_save_path)
    torch.save(train_labels, train_labels_save_path)
    print(f"Saved Training Data to {train_save_path} and {train_labels_save_path}")

    # Preprocess Testing Data
    print("Preprocessing Testing Data...")
    test_combined, test_labels = preprocess_dataset(
        dataset_name=dataset_name,
        dataset_root=dataset_root,
        hash_sig_path=test_hash_sig_path,
        split='test'
    )
    print(f"Testing Data Combined Shape: {test_combined.shape}")
    print(f"Testing Labels Shape: {test_labels.shape}")

    # Save Testing Data
    test_save_path = os.path.join(SAVE_DIR, 'CIFAR10_test_combined.pt')
    test_labels_save_path = os.path.join(SAVE_DIR, 'CIFAR10_test_labels.pt')
    torch.save(test_combined, test_save_path)
    torch.save(test_labels, test_labels_save_path)
    print(f"Saved Testing Data to {test_save_path} and {test_labels_save_path}")

    print("Preprocessing and Saving Completed Successfully.")

if __name__ == "__main__":
    main()
