import torch
from utils import get_input  # Assuming get_input is in utils.py
from CustomeDataset import CustomDataset, get_train_test_loaders  # Assuming CustomDataset is in CustomDataset.py
from torch.utils.data import DataLoader
from torchvision import transforms
device = torch.device('cpu')
if torch.cuda.device_count():
    # torch.cuda.manual_seed(seed)
    device = torch.device('cuda')
# Set up some constants
DIM_SIGNATURE = 256
DIM_HASH = 24
INPUT_RESOLUTION = 32**2

# Assuming you have already created the dataset and hash/signature paths
train_hash_sig_path = './../datasets/hashes_signatures_train_cifar10_256.h5'
test_hash_sig_path = './../datasets/hashes_signatures_test_cifar10_256.h5'
dataset_path = './path_to_dataset'

# Define the transform (same as in your dataset)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Select two indexes to test
index1 = 0  # You can change these to any valid indexes
index2 = 1

# Load the data for those indexes
dataset = CustomDataset(name="CIFAR", root=dataset_path, hash_sig_path=train_hash_sig_path, train=True, transform=transform)
image1, signature1, hash_x1, label1, false_flag1 = dataset[index1]
image2, signature2, hash_x2, label2, false_flag2 = dataset[index2]

# Apply the transform to the images if they are not already tensors
if not isinstance(image1, torch.Tensor):
    image1 = transform(image1)
if not isinstance(image2, torch.Tensor):
    image2 = transform(image2)

# Print initial values before calling get_input
print(f"Image1 Shape: {image1.shape}, Values: {image1}")
print(f"Signature1 Shape: {signature1.shape}, Values: {signature1}")
print(f"Hash1 Shape: {hash_x1.shape}, Values: {hash_x1}")

# Convert the image, signature, and hash using get_input (from utils)
inputs1 = get_input(torch.unsqueeze(image1, 0), torch.unsqueeze(signature1, 0), torch.unsqueeze(hash_x1, 0), INPUT_RESOLUTION=32**2)
inputs2 = get_input(torch.unsqueeze(image2, 0), torch.unsqueeze(signature2, 0), torch.unsqueeze(hash_x2, 0), INPUT_RESOLUTION=32**2)

# Print the concatenated inputs after get_input
print(f"Original get_input - Concatenated Input1 Shape: {inputs1.shape}, Values: {inputs1}")

# Now, let's define the new version of get_input as get_input_1 and compare

def get_input_1(x, signature, hash_x):
    # Get the concatenated input
    if signature.shape[1] < INPUT_RESOLUTION:
        # Pad signature and hash with 0
        signature = torch.cat((signature, torch.zeros(
            signature.shape[0], INPUT_RESOLUTION-signature.shape[1]).to(signature.device)), dim=1)
        hash_x = torch.cat((hash_x, torch.zeros(
            hash_x.shape[0], INPUT_RESOLUTION-hash_x.shape[1]).to(hash_x.device)), dim=1)

    # Pad signature and hash to the next multiple of INPUT_RESOLUTION
    signature = torch.cat((signature, torch.zeros(
        signature.shape[0], INPUT_RESOLUTION-signature.shape[1]).to(signature.device)), dim=1)
    hash_x = torch.cat((hash_x, torch.zeros(
        hash_x.shape[0], INPUT_RESOLUTION-hash_x.shape[1]).to(signature.device)), dim=1)

    # Make signature and hash to be of shape (batch_size, (sqrt(INPUT_RESOLUTION)), sqrt(INPUT_RESOLUTION))
    signature = signature.view(
        signature.shape[0], -1, int(INPUT_RESOLUTION**0.5), int(INPUT_RESOLUTION**0.5))
    hash_x = hash_x.view(
        hash_x.shape[0], -1,  int(INPUT_RESOLUTION**0.5), int(INPUT_RESOLUTION**0.5))

    return torch.cat((x, signature, hash_x), dim=1).to(device)

# Convert the image, signature, and hash using the new get_input_1
inputs1_v1 = get_input_1(torch.unsqueeze(image1, 0), torch.unsqueeze(signature1, 0), torch.unsqueeze(hash_x1, 0))
inputs2_v1 = get_input_1(torch.unsqueeze(image2, 0), torch.unsqueeze(signature2, 0), torch.unsqueeze(hash_x2, 0))

# Print the concatenated inputs after get_input_1
print(f"New get_input_1 - Concatenated Input1 Shape: {inputs1_v1.shape}, Values: {inputs1_v1}")

# Compare the outputs
print(f"Original get_input vs. New get_input_1 (Input1 Comparison): {torch.equal(inputs1, inputs1_v1)}")
print(f"Original get_input vs. New get_input_1 (Input2 Comparison): {torch.equal(inputs2, inputs2_v1)}")
