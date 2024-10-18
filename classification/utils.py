# adapted from
# https://github.com/VICO-UoE/DatasetCondensation

import time
import numpy as np
from torch.nn import CrossEntropyLoss
import torch
from torch.utils.data import DataLoader, Dataset
import random
import torch.nn as nn
import torch.nn.functional as F
import os
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, TensorDataset
from torchvision import datasets, transforms
from timm.models import create_model
import copy
from torch import nn, optim
from tqdm import tqdm
import timm
from torch import Tensor
import torch.utils.model_zoo as model_zoo
import learn2learn as l2l
from torch.utils.data import DataLoader, Dataset
import torch.nn.init as init
import csv
from model import VGG, make_layers, cfg
from PIL import Image
from CustomeDataset import CustomDataset
from typing import (
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    Dict
)
from copy import deepcopy

dataTransform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    
])

DIM_SIGNATURE=256
DIM_HASH=24

size = 64

Resize_transform = transforms.Compose([
    transforms.Resize([size,size]),
])
class ResizedTensorDataset(Dataset[Tuple[Tensor, ...]]):
    r"""Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Args:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """

    def __init__(self, *tensors) -> None:
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors), "Size mismatch between tensors"
        self.tensors = tensors

    def __getitem__(self, index):
        transformed_tensors = [Resize_transform(tensor[index]) for tensor in self.tensors[:-1]]  # Apply transform to all tensors except the last one (assuming it's the label tensor)
        return tuple(transformed_tensors + [self.tensors[-1][index]])  # Include the original label tensor

    def __len__(self):
        return self.tensors[0].size(0)


def one_hot_to_value(one_hot_tensor):
    index = torch.argmax(one_hot_tensor).item()
    return index

# one_hot_tensor = torch.tensor([0, 0, 1, 0, 0])
# value = one_hot_to_value(one_hot_tensor)
class stl_Dataset(Dataset):
    def __init__(self, stldataset = None):

        self.list_img = []
        self.list_label = []
        self.transform = dataTransform
        img_list = stldataset[0]

        for img in img_list:
            self.list_img.append(img)

        self.list_label = stldataset[1]
        self.list_img = np.asarray(self.list_img)
        self.list_label = np.asarray(self.list_label)

    def __getitem__(self, item):
        img = self.list_img[item]
        label = self.list_label[item]
        return self.transform(img), one_hot_to_value(torch.tensor(label))

    def __len__(self):
        return len(self.list_img)
    

from copy import deepcopy
import torch
import torch.nn as nn

def get_new_model(model_orig, 
                  args, 
                  DIM_SIGNATURE=256, 
                  DIM_HASH=24, 
                  num_classes=None):
    """
    Modify the model to accept additional input channels for signatures and hashes based on the architecture.

    Args:
        model_orig (nn.Module): Original model (ResNet50 or CaFormer).
        args: Argument parser object with attributes like 'arch' (for architecture).
        DIM_SIGNATURE (int): Dimension of the signature.
        DIM_HASH (int): Dimension of the hash.
        num_classes (int, optional): Number of output classes. If provided, modifies the final layer.

    Returns:
        nn.Module: Modified model with additional input channels and optional new final layer.
    """
    # Create a deep copy of the original model to avoid modifying it directly
    model_new = deepcopy(model_orig)

    if args.arch == 'res50':
        # Calculate the number of additional channels needed for ResNet50
        INPUT_RESOLUTION = 32 * 32  # Example for CIFAR-10; adjust as needed
        num_signature_channels = (DIM_SIGNATURE // INPUT_RESOLUTION) + 1
        num_hash_channels = (DIM_HASH // INPUT_RESOLUTION) + 1
        total_additional_channels = num_signature_channels + num_hash_channels

        # Update the first convolutional layer to accept additional channels
        in_channels_new = 3 + total_additional_channels  # Original RGB channels + additional
        out_channels = model_orig.conv1.out_channels  # Typically 64 for ResNet50

        # Create new conv1 layer with additional input channels
        model_new.conv1 = nn.Conv2d(
            in_channels_new, 
            out_channels, 
            kernel_size=model_orig.conv1.kernel_size, 
            stride=model_orig.conv1.stride, 
            padding=model_orig.conv1.padding, 
            bias=model_orig.conv1.bias is not None
        )

        # Initialize the new conv1 weights
        with torch.no_grad():
            # Get pre-trained conv1 weights from the original model
            pre_trained_weights = model_orig.conv1.weight  # Shape: [64, 3, 7, 7]

            # Create new conv1 weights with additional channels
            new_weights = torch.zeros_like(model_new.conv1.weight)  # Shape: [64, in_channels_new, 7, 7]

            # Copy the pre-trained weights for the original 3 channels
            new_weights[:, :3, :, :] = pre_trained_weights

            # Initialize the weights for the additional channels (using Kaiming Normal)
            nn.init.kaiming_normal_(new_weights[:, 3:, :, :], mode='fan_out', nonlinearity='relu')

            # Assign the new weights to conv1
            model_new.conv1.weight = nn.Parameter(new_weights)

        if num_classes is not None:
            # Modify the final fully connected layer (fc) to match the new number of classes
            in_features = model_new.fc.in_features  # Typically 2048 for ResNet50
            model_new.fc = nn.Linear(in_features, num_classes)

            # Initialize the new fc layer
            nn.init.kaiming_normal_(model_new.fc.weight, mode='fan_out', nonlinearity='relu')
            if model_new.fc.bias is not None:
                nn.init.constant_(model_new.fc.bias, 0)

    elif args.arch == 'caformer':
        # For CaFormer, MetaFormer architecture:
        INPUT_RESOLUTION = 224 * 224  # Adjust resolution for CaFormer if needed
        num_signature_channels = (DIM_SIGNATURE // INPUT_RESOLUTION) + 1
        num_hash_channels = (DIM_HASH // INPUT_RESOLUTION) + 1
        total_additional_channels = num_signature_channels + num_hash_channels

        in_channels_new = 3 + total_additional_channels  # Original RGB channels + additional

        # CaFormer might use `stem` or `patch_embed` for initial embedding, need to adjust here
        if hasattr(model_orig, 'stem'):
            # Modify the stem conv layer (assuming it exists)
            model_new.stem.conv = nn.Conv2d(
                in_channels_new,
                model_orig.stem.conv.out_channels,
                kernel_size=model_orig.stem.conv.kernel_size,
                stride=model_orig.stem.conv.stride,
                padding=model_orig.stem.conv.padding,
                bias=model_orig.stem.conv.bias is not None
            )

            # Initialize the new weights for the stem
            with torch.no_grad():
                pre_trained_weights = model_orig.stem.conv.weight  # Shape depends on CaFormer
                new_weights = torch.zeros_like(model_new.stem.conv.weight)
                new_weights[:, :3, :, :] = pre_trained_weights
                nn.init.kaiming_normal_(new_weights[:, 3:, :, :], mode='fan_out', nonlinearity='relu')
                model_new.stem.conv.weight = nn.Parameter(new_weights)

        elif hasattr(model_orig, 'patch_embed'):
            # Modify the patch embedding projection layer
            model_new.patch_embed.proj = nn.Conv2d(
                in_channels_new,
                model_orig.patch_embed.proj.out_channels,
                kernel_size=model_orig.patch_embed.proj.kernel_size,
                stride=model_orig.patch_embed.proj.stride,
                padding=model_orig.patch_embed.proj.padding,
                bias=model_orig.patch_embed.proj.bias is not None
            )

            # Initialize the new weights for the patch embed
            with torch.no_grad():
                pre_trained_weights = model_orig.patch_embed.proj.weight  # Shape depends on CaFormer
                new_weights = torch.zeros_like(model_new.patch_embed.proj.weight)
                new_weights[:, :3, :, :] = pre_trained_weights
                nn.init.kaiming_normal_(new_weights[:, 3:, :, :], mode='fan_out', nonlinearity='relu')
                model_new.patch_embed.proj.weight = nn.Parameter(new_weights)

        else:
            raise ValueError(f"Unsupported layer configuration for CaFormer: {args.arch}")

        if num_classes is not None:
            # Modify the final classifier for CaFormer
            in_features = model_new.head.fc.fc2.in_features
            model_new.head.fc.fc2 = nn.Linear(in_features, num_classes)

            # Initialize the new classifier
            nn.init.kaiming_normal_(model_new.head.fc.fc2.weight, mode='fan_out', nonlinearity='relu')
            if model_new.head.fc.fc2.bias is not None:
                nn.init.constant_(model_new.head.fc.fc2.bias, 0)

    else:
        raise ValueError(f"Unsupported architecture: {args.arch}")

    return model_new



import torch




def get_input(x, signature, hash_x, INPUT_RESOLUTION=32**2):
    """
    Concatenate a single image, its signature, and hash into a single input tensor with additional channels.

    Args:
        x (torch.Tensor): Tensor of images with shape (batch_size, C, H, W).
        signature (torch.Tensor or list): Tensor or list of signatures with shape (batch_size, sig_dim).
        hash_x (torch.Tensor or list): Tensor or list of hashes with shape (batch_size, hash_dim).
        INPUT_RESOLUTION (int): Number of bits per channel (default: 32**2 = 1024).

    Returns:
        torch.Tensor: Concatenated tensor with shape 
                      (batch_size, C + sig_channels + hash_channels, H, W)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Randomly decide whether to print or not (5% probability)
    if random.random() < 0.00:
        print(f"Initial image shape: {x.shape}, values: {x}")
        print(f"Initial signature shape: {signature.shape if isinstance(signature, torch.Tensor) else 'list'}, values: {signature}")
        print(f"Initial hash shape: {hash_x.shape if isinstance(hash_x, torch.Tensor) else 'list'}, values: {hash_x}")

    # Move images to device
    x = x.to(device)
    
    # Convert signature to tensor if it's a list and move to device
    if isinstance(signature, list):
        signature = torch.tensor(signature, dtype=torch.float32)
    signature = signature.to(device)
    
    # Convert hash_x to tensor if it's a list and move to device
    if isinstance(hash_x, list):
        hash_x = torch.tensor(hash_x, dtype=torch.float32)
    hash_x = hash_x.to(device)
    
    batch_size, C, H, W = x.shape
    sig_dim = DIM_SIGNATURE
    hash_dim = DIM_HASH

    # Randomly decide whether to print or not (5% probability)
    if random.random() < 0.00:
        print("sig", signature)
        print("hash", hash_x)
        print(f"Signature tensor shape after conversion: {signature.shape}, values: {signature}")
        print(f"Hash tensor shape after conversion: {hash_x.shape}, values: {hash_x}")

    # Calculate the number of channels needed for signatures and hashes
    sig_channels = (sig_dim // INPUT_RESOLUTION) + 1
    hash_channels = (hash_dim // INPUT_RESOLUTION) + 1

    # Pad signatures and hashes to match INPUT_RESOLUTION
    padding_sig = sig_channels * INPUT_RESOLUTION - sig_dim
    padding_hash = hash_channels * INPUT_RESOLUTION - hash_dim

    if padding_sig > 0:
        signature = torch.cat(
            (signature, torch.zeros(batch_size, padding_sig).to(device)), dim=1)
    if padding_hash > 0:
        hash_x = torch.cat(
            (hash_x, torch.zeros(batch_size, padding_hash).to(device)), dim=1)

    # Randomly decide whether to print or not (5% probability)
    if random.random() < 0.00:
        print(f"Signature shape after padding: {signature.shape}, values: {signature}")
        print(f"Hash shape after padding: {hash_x.shape}, values: {hash_x}")

    # Reshape signatures and hashes into image-like tensors
    signature = signature.view(
        batch_size, sig_channels, int(INPUT_RESOLUTION**0.5), int(INPUT_RESOLUTION**0.5))
    hash_x = hash_x.view(
        batch_size, hash_channels, int(INPUT_RESOLUTION**0.5), int(INPUT_RESOLUTION**0.5))

    # Concatenate image, signature, and hash along the channel dimension
    combined_input = torch.cat((x, signature, hash_x), dim=1)

    # Randomly decide whether to print or not (5% probability)
    if random.random() < 0.00:
        print(f"Combined input shape: {combined_input.shape}, values: {combined_input}")

    return combined_input



def save_data(save_path, queryset_loss, queryset_acc, originaltest_loss, originaltrain_loss, originaltest_acc, finetuned_target_testacc, finetuned_target_testloss, final_original_testacc, final_finetuned_testacc, final_finetuned_testloss, total_loop_index, ml_index, nl_index):
     
    with open(save_path + '/' + 'queryset_loss_acc.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['total loop', 'maml loop', 'query set loss', 'query set accuracy'])
        for i, j, k, q in zip(total_loop_index, ml_index, queryset_loss, queryset_acc):
            writer.writerow([i, j, k, q]) 
    with open(save_path + '/' + 'originaltrain_loss.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['total loop', 'natural loop', 'original train loss'])
        for i, j, k in zip(total_loop_index, nl_index,originaltrain_loss):
            writer.writerow([i, j, k]) 
    with open(save_path + '/' + 'originaltest_loss_acc.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['total loop', 'natural loop', 'original test loss', 'original test accuracy'])
        for i, j, k, q in zip(total_loop_index, nl_index,originaltest_loss, originaltest_acc):
            writer.writerow([i, j, k, q]) 
    with open(save_path + '/' + 'finetuned_target_test.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['total loop', 'finetuned target testacc', 'finetuned target testloss'])
        for i, j, k in zip(total_loop_index, finetuned_target_testacc, finetuned_target_testloss):
            writer.writerow([i, j, k]) 
    with open(save_path + '/' + 'final_test.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['final original test acc', 'final finetuned test acc', 'final finetuned testloss'])
        for i, j, k in zip(final_original_testacc, final_finetuned_testacc, final_finetuned_testloss):
            writer.writerow([i, j, k])
class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs, config_map):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]
        self.config_map = config_map

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        id_dict = {value: index for index, value in enumerate(self.config_map)}
        label = id_dict[label]
        return image, label
class Config:
    imagenette = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]

    # ["australian_terrier", "border_terrier", "samoyed", "beagle", "shih-tzu", "english_foxhound", "rhodesian_ridgeback", "dingo", "golden_retriever", "english_sheepdog"]
    imagewoof = [193, 182, 258, 162, 155, 167, 159, 273, 207, 229]

    # ["tabby_cat", "bengal_cat", "persian_cat", "siamese_cat", "egyptian_cat", "lion", "tiger", "jaguar", "snow_leopard", "lynx"]
    imagemeow = [281, 282, 283, 284, 285, 291, 292, 290, 289, 287]

    # ["peacock", "flamingo", "macaw", "pelican", "king_penguin", "bald_eagle", "toucan", "ostrich", "black_swan", "cockatoo"]
    imagesquawk = [84, 130, 88, 144, 145, 22, 96, 9, 100, 89]

    # ["pineapple", "banana", "strawberry", "orange", "lemon", "pomegranate", "fig", "bell_pepper", "cucumber", "green_apple"]
    imagefruit = [953, 954, 949, 950, 951, 957, 952, 945, 943, 948]

    # ["bee", "ladys slipper", "banana", "lemon", "corn", "school_bus", "honeycomb", "lion", "garden_spider", "goldfinch"]
    imageyellow = [309, 986, 954, 951, 987, 779, 599, 291, 72, 11]

    dict = {
        "imagenette" : imagenette,
        "imagewoof" : imagewoof,
        "imagefruit": imagefruit,
        "imageyellow": imageyellow,
        "imagemeow": imagemeow,
        "imagesquawk": imagesquawk,
    }

config = Config()

def test_accuracy(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False)  # sum up batch loss
            pred = torch.max(output, 1)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    print('\n Test_set: Average loss: {:.4f}, Accuracy: {:.4f}\n'.format(test_loss, acc))
    return acc, test_loss


def normalize(X):
    cifar10_mean = (0.4914, 0.4822, 0.4465) # equals np.mean(train_set.train_data, axis=(0,1,2))/255
    cifar10_std = (0.2471, 0.2435, 0.2616)
    mu = torch.tensor(cifar10_mean).view(3, 1, 1).cuda()
    std = torch.tensor(cifar10_std).view(3, 1, 1).cuda()
    return (X - mu)/std


def get_dataset(dataset, data_path, subset="imagenette", args=None, train_hash_sig_path=None, test_hash_sig_path=None):


    if dataset == 'MNIST':
        if args.arch == 'vgg':
            size = 64
            transform = transforms.Compose([transforms.Resize([size, size]), transforms.Grayscale(num_output_channels=3) ,transforms.ToTensor(),transforms.Normalize((0.1307,0.1307,0.1307), (0.3081,0.3081,0.3081))])
            print('check')
        else:
            transform = transforms.Compose([transforms.Grayscale(num_output_channels=3), transforms.ToTensor(),transforms.Normalize((0.1307,0.1307,0.1307), (0.3081,0.3081,0.3081))])
        trainset = datasets.MNIST(data_path, train=True, download=False, transform=transform)
        testset = datasets.MNIST(data_path, train=False, download=False, transform=transform)
        return trainset, testset
    
    elif dataset == 'CIFAR10':
        channel = 3
        im_size = (32, 32)
        num_classes = 10
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        if args.arch == 'vgg':
            transform = transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)]
            ) 
        elif args.arch == "vit":
            transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to 224x224
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Use ImageNet means
                                std=[0.229, 0.224, 0.225]),  # Use ImageNet stds
        ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)]
                )  
        trainset = datasets.CIFAR10(data_path, train=True, download=True, transform=transform) # no augmentation
        testset = datasets.CIFAR10(data_path, train=False, download=True, transform=transform)
        class_names = trainset.classes
        # print(trainset.data.shape)
        class_map = {x:x for x in range(num_classes)}


    elif dataset == 'CIFAR10-correct-sig':
        channel = 3
        im_size = (32, 32)
        num_classes = 10
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        if args.arch == 'vgg':
            transform = transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)]
            ) 
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)]
                )  
        trainset = CustomDataset(name="CIFAR",root=data_path, hash_sig_path=train_hash_sig_path,
                                         train=True, transform=transform, false_signature_rate=0, train_perturbation=10, undo_finetuning=False, sig_dim=DIM_SIGNATURE) # no augmentation
        testset = CustomDataset(
        name="CIFAR",root=data_path, hash_sig_path=test_hash_sig_path, train=False, false_signature_rate=0, transform=transform, sig_dim=DIM_SIGNATURE)

    elif dataset == 'CIFAR10-wrong-sig':
        channel = 3
        im_size = (32, 32)
        num_classes = 10
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        if args.arch == 'vgg':
            transform = transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)]
            ) 
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)]
                )  
        trainset = CustomDataset(name="CIFAR", root=data_path, hash_sig_path=train_hash_sig_path,
                                         train=True, transform=transform, false_signature_rate=1, train_perturbation=None, undo_finetuning=False, sig_dim=DIM_SIGNATURE) # no augmentation
        testset = CustomDataset(
        name="CIFAR",root=data_path, hash_sig_path=test_hash_sig_path, train=False, false_signature_rate=1, transform=transform,train_perturbation=None, sig_dim=DIM_SIGNATURE)

    elif dataset == 'Tiny':
        channel = 3
        im_size = (64, 64)
        num_classes = 200
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        trainset = datasets.ImageFolder(os.path.join(data_path, "train"), transform=transform) # no augmentation
        testset = datasets.ImageFolder(os.path.join(data_path, "val"), transform=transform)
        class_names = trainset.classes
        class_map = {x:x for x in range(num_classes)}


    elif dataset == 'ImageNet':
        # if args.arch == 'vgg':
        #     size = 64
        #     transform = transforms.Compose([
        #         transforms.Resize([size,size]),
        #         transforms.ToTensor(),
        #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        #     ])
        # else:
        #     size = 256
        #     transform = transforms.Compose([
        #         transforms.Resize(size),
        #         transforms.CenterCrop(224),
        #         transforms.ToTensor(),
        #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        #     ])
        # data_path += 'ILSVRC/Data/CLS-LOC'
        # config.img_net_classes = config.dict[subset]
        # test_dataset_all =datasets.ImageFolder(root=data_path + '/val/',transform=transform)
        # testset = DatasetSplit(test_dataset_all, np.squeeze(np.argwhere(np.isin(test_dataset_all.targets, config.img_net_classes))), config.img_net_classes)
        # train_dataset_all =datasets.ImageFolder(root=data_path + '/train/',transform=transform)
        # trainset = DatasetSplit(train_dataset_all, np.squeeze(np.argwhere(np.isin(train_dataset_all.targets, config.img_net_classes))), config.img_net_classes)   
        data = torch.load(data_path + 'imagenette.pt')
        image_train = data['images train']
        image_test = data['images test']
        target_train = data['targets train']
        target_test = data['targets test']
        if args.arch == 'vgg':
            trainset=ResizedTensorDataset(image_train, target_train)
            testset = ResizedTensorDataset(image_test, target_test)
        
        else:
            trainset = TensorDataset(image_train, target_train)
            testset = TensorDataset(image_test, target_test)


    elif dataset == 'SVHN':
        if args.arch == 'vgg':
            size = 64
            transform = transforms.Compose([
            transforms.Resize([size,size]),
            transforms.ToTensor(),
            ])
        else:
            transform = transforms.Compose([
            transforms.ToTensor(),
            ])
        trainset = datasets.SVHN(
        data_path, split='train', transform=transform, download=True)
        testset = datasets.SVHN(
        data_path, split='test', transform=transform, download=True)
    
    elif dataset == 'CINIC':
        cinic_directory = data_path + '/' + 'CINIC10/'
        cinic_mean = [0.47889522, 0.47227842, 0.43047404]
        cinic_std = [0.24205776, 0.23828046, 0.25874835]
        trainset = datasets.ImageFolder(cinic_directory + '/train',
        transform=transforms.Compose([transforms.ToTensor(),
        transforms.Normalize(mean=cinic_mean,std=cinic_std)]))

        testset = datasets.ImageFolder(cinic_directory + '/test',
        transform=transforms.Compose([transforms.ToTensor(),
        transforms.Normalize(mean=cinic_mean,std=cinic_std)]))

    elif dataset == 'STL':
        list_img_train = []
        list_label_train = []
        list_img_test = []
        list_label_test = []
        traindata_size = 0
        testdata_size = 0

        re_label = [0, 2, 1, 3, 4, 5, 7, 6, 8, 9]
        root = data_path + '/stl10_binary'
        train_x_path = os.path.join(root, 'train_X.bin')
        train_y_path = os.path.join(root, 'train_y.bin')
        test_x_path = os.path.join(root, 'test_X.bin')
        test_y_path = os.path.join(root, 'test_y.bin')
        with open(train_x_path, 'rb') as fo:
            train_x = np.fromfile(fo, dtype=np.uint8)
            train_x = np.reshape(train_x, (-1, 3, 96, 96))
            train_x = np.transpose(train_x, (0, 3, 2, 1))
        with open(train_y_path, 'rb') as fo:
            train_y = np.fromfile(fo, dtype=np.uint8)

        for i in range(len(train_y)):
            label = re_label[train_y[i] - 1]
            list_img_train.append(train_x[i])
            list_label_train.append(np.eye(10)[label])
            traindata_size += 1

        with open(test_x_path, 'rb') as fo:
            test_x = np.fromfile(fo, dtype=np.uint8)
            test_x = np.reshape(test_x, (-1, 3, 96, 96))
            test_x = np.transpose(test_x, (0, 3, 2, 1))
        with open(test_y_path, 'rb') as fo:
            test_y = np.fromfile(fo, dtype=np.uint8)

        for i in range(len(test_y)):
            label = re_label[test_y[i] - 1]
            list_img_test.append(test_x[i])
            list_label_test.append(np.eye(10)[label])
            testdata_size += 1

        # np.random.seed(0)
        ind = np.arange(traindata_size)
        ind = np.random.permutation(ind)
        list_img_train = np.asarray(list_img_train)
        list_img_train = list_img_train[ind]
        list_label_train = np.asarray(list_label_train)
        list_label_train = list_label_train[ind]


        ind = np.arange(testdata_size)
        ind = np.random.permutation(ind)
        list_img_test = np.asarray(list_img_test)
        list_img_test = list_img_test[ind]
        list_label_test = np.asarray(list_label_test)
        list_label_test = list_label_test[ind]

        trainset = stl_Dataset([list_img_train, list_label_train])
        testset = stl_Dataset([list_img_test, list_label_test])

    else:
        exit('unknown dataset: %s'%dataset)
    return trainset, testset

def process(checkpoint):
    new_state_dict = {}
    for key, value in checkpoint.items():
        if key.startswith('module.'):
            new_state_dict[key[7:]] = value
        else:
            new_state_dict[key] = value
    return new_state_dict


def get_default_convnet_setting():
    net_width, net_depth, net_act, net_norm, net_pooling = 128, 3, 'relu', 'instancenorm', 'avgpooling'
    return net_width, net_depth, net_act, net_norm, net_pooling




def get_network(model, channel, num_classes, im_size=(32, 32), dist=True):
    torch.random.manual_seed(int(time.time() * 1000) % 100000)
    print(f"----------------Using {model} Model----------------")
    net_width, net_depth, net_act, net_norm, net_pooling = get_default_convnet_setting()

    if model == 'MLP':
        net = MLP(channel=channel, num_classes=num_classes)
    elif model == 'ConvNet':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'LeNet':
        net = LeNet(channel=channel, num_classes=num_classes)
    elif model == 'AlexNet':
        net = AlexNet(channel=channel, num_classes=num_classes)
    elif model == 'VGG11':
        net = VGG11( channel=channel, num_classes=num_classes)
    elif model == 'VGG11BN':
        net = VGG11BN(channel=channel, num_classes=num_classes)
    elif model == 'ResNet18':
        net = ResNet18(channel=channel, num_classes=num_classes)
    elif model == 'ResNet18BN_AP':
        net = ResNet18BN_AP(channel=channel, num_classes=num_classes)
    elif model == 'ResNet18_AP':
        net = ResNet18_AP(channel=channel, num_classes=num_classes)

    elif model == 'ConvNetD1':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=1, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetD2':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=2, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetD3':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=3, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetD4':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=4, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetD5':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=5, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetD6':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=6, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetD7':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=7, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetD8':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=8, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)


    elif model == 'ConvNetW32':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=32, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling)
    elif model == 'ConvNetW64':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=64, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling)
    elif model == 'ConvNetW128':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=128, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling)
    elif model == 'ConvNetW256':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=256, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling)
    elif model == 'ConvNetW512':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=512, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling)
    elif model == 'ConvNetW1024':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=1024, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling)

    elif model == "ConvNetKIP":
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=1024, net_depth=net_depth, net_act=net_act,
                      net_norm="none", net_pooling=net_pooling)

    elif model == 'ConvNetAS':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act='sigmoid', net_norm=net_norm, net_pooling=net_pooling)
    elif model == 'ConvNetAR':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act='relu', net_norm=net_norm, net_pooling=net_pooling)
    elif model == 'ConvNetAL':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act='leakyrelu', net_norm=net_norm, net_pooling=net_pooling)

    elif model == 'ConvNetNN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='none', net_pooling=net_pooling)
    elif model == 'ConvNetBN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='batchnorm', net_pooling=net_pooling)
    elif model == 'ConvNetLN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='layernorm', net_pooling=net_pooling)
    elif model == 'ConvNetIN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='instancenorm', net_pooling=net_pooling)
    elif model == 'ConvNetGN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='groupnorm', net_pooling=net_pooling)

    elif model == 'ConvNetNP':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling='none')
    elif model == 'ConvNetMP':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling='maxpooling')
    elif model == 'ConvNetAP':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling='avgpooling')

    ###
    elif model == 'WideRes':
        net = WideResNet(channel=channel, num_classes=num_classes)


    else:
        net = None
        exit('DC error: unknown model')

    if dist:
        gpu_num = torch.cuda.device_count()
        if gpu_num>0:
            device = 'cuda'
            if gpu_num>1:
                net = nn.DataParallel(net)
        else:
            device = 'cpu'
        net = net.to(device)

    return net



def get_time():
    return str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))


mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]
mu = torch.tensor(mean).view(3, 1, 1).cuda()
std = torch.tensor(std).view(3, 1, 1).cuda()
def normalize(X):
    return (X - mu)/std

def resume_dict(state_dict):
    # Default values
    model_name = 'resnet50'
    pretrained = False
    gp = None
    in_chans = 3
    input_size = None
    model_kwargs = {}
    head_init_scale = None
    head_init_bias = None
    if input_size is not None:
        in_chans = input_size[0]

    model = create_model(
        model_name,
        pretrained=pretrained,
        in_chans=in_chans,
        num_classes=10,
        global_pool=gp,
        **model_kwargs,
    )
    if not state_dict:
        return model

    if head_init_scale is not None:
        with torch.no_grad():
            model.get_classifier().weight.mul_(head_init_scale)
            model.get_classifier().bias.mul_(head_init_scale)
    if head_init_bias is not None:
        nn.init.constant_(model.get_classifier().bias, head_init_bias)

    # Load checkpoint if provided
    model.load_state_dict(state_dict, strict=True)
    print(f"=> Loaded checkpoint.....'")
    return model

def resume(resume_path):
    # Default values
    model_name = 'resnet50'
    pretrained = False
    gp = None
    in_chans = 3
    input_size = None
    model_kwargs = {}
    head_init_scale = None
    head_init_bias = None
    
    
    resume_path = resume_path

    if input_size is not None:
        in_chans = input_size[0]

    model = create_model(
        model_name,
        pretrained=pretrained,
        in_chans=in_chans,
        num_classes=10,
        global_pool=gp,
        **model_kwargs,
    )
    
    if head_init_scale is not None:
        with torch.no_grad():
            model.get_classifier().weight.mul_(head_init_scale)
            model.get_classifier().bias.mul_(head_init_scale)
    if head_init_bias is not None:
        nn.init.constant_(model.get_classifier().bias, head_init_bias)

    # Load checkpoint if provided
    if resume_path:
        if not os.path.isfile(resume_path):
            raise RuntimeError(f"=> No checkpoint found at '{resume_path}'")
        checkpoint = torch.load(resume_path)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        print(f"=> Loaded checkpoint '{resume_path}'")
    return model

import torch.nn as nn
import learn2learn as l2l
from learn2learn.algorithms import  MAML 

def unwrap_model(model):
    """
    Recursively unwraps the model from wrappers like DataParallel and MAML.
    """
    while isinstance(model, (nn.DataParallel, l2l.algorithms.MAML)):
        model = model.module
    return model



def initialize(args,model): #因为maml会多套一层 所以test_finetune里面的另写一个
    if args.arch == 'res50':
        last_layer = model.module.module.fc
    elif args.arch == 'caformer': 
        last_layer = model.module.module.head.fc.fc2
    elif args.arch == 'res18':
        last_layer = model.module.module.fc
    elif args.arch == 'res34':
        last_layer = model.module.module.fc
    elif args.arch == 'vgg':
        last_layer == model.module.module.fc
    init.xavier_uniform_(last_layer.weight)
    if last_layer.bias is not None:
        init.zeros_(last_layer.bias)
    return model



def initialize00(args,model): #一层.module都没有
    if args.arch == 'res50':
        last_layer = model.fc
    elif args.arch == 'caformer': 
        last_layer = model.head.fc.fc2
    elif args.arch == 'res18':
        last_layer = model.fc
    elif args.arch == 'res34':
        last_layer = model.fc
    elif args.arch == 'vgg':
        last_layer = model.fc
    init.xavier_uniform_(last_layer.weight)
    if last_layer.bias is not None:
        init.zeros_(last_layer.bias)
    return model


def vgg19(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration "E")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['B']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg13']), strict = False)
    return model

def get_pretrained_model(args, partial_finetuned=False):
   
    if args.arch == 'caformer':
        model = timm.create_model("caformer_m36", pretrained=False)
        classifier = nn.Linear(2304, 10)
        model.head.fc.fc2=classifier
        #state_dict = process(torch.load('./pretrained/caformer_m36_imagenette.pth'))
        state_dict = process(torch.load('./pretrained/caformer_m36_cifar10.pth'))
        model.load_state_dict(state_dict)
        if partial_finetuned:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.head.fc.fc2.parameters():
                param.requires_grad = True
        return model.cuda()
    
    elif args.arch == 'vgg':
        model = VGG(make_layers(cfg['B']), num_classes=10)
        model.load_state_dict(process(torch.load('../pretrained/vgg_ImageNet_95.4_model.pkl')))
        if partial_finetuned:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True
        return model.cuda()
    
    elif args.arch == 'res18':
        from model import resnet18
        model = resnet18(pretrained=False, num_classes=10).cuda()
        model.load_state_dict(process(torch.load('../pretrained/res18_ImageNet_98.6_model.pkl')))
        if partial_finetuned:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True
        return model.cuda()

    elif args.arch == 'res34':
        from model import resnet34
        model = resnet34(pretrained=False, num_classes=10).cuda()
        model.load_state_dict(process(torch.load('../pretrained/res34_ImageNet_99.0_model.pkl')))
        if partial_finetuned:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True
        return model.cuda()
    
    elif args.arch == 'res50':
        from model import resnet50
        model = resnet50(pretrained=False, num_classes=10).cuda()
        #model.load_state_dict(process(torch.load('./pretrained/resnet50_imagenette.pth')))
        model.load_state_dict(process(torch.load('./pretrained/resnet50_cifar10.pth')))
        if partial_finetuned:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True
        return model.cuda()
    elif args.arch == 'vit':
        model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=10)
        model.load_state_dict(torch.load("./pretrained/vit_base_patch16_224_imagenette.pth"))
        if partial_finetuned:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.head.parameters():
                param.requires_grad = True
        return model.cuda()
    else:
        assert(0)

def get_finetuned_model(args, our_path, partial_finetuned=False):
   
    if args.arch == 'caformer':
        model = timm.create_model("caformer_m36", pretrained=False)
        classifier = nn.Linear(2304, 10)
        model.head.fc.fc2=classifier
        state_dict = process(torch.load(our_path)['model'])
        model.load_state_dict(state_dict)
        if partial_finetuned:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.head.fc.fc2.parameters():
                param.requires_grad = True
        return model.cuda()
    
    elif args.arch == 'vgg':
        model = VGG(make_layers(cfg['B']), num_classes=10)
        model.load_state_dict(process(torch.load(our_path)['model']))
        if partial_finetuned:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True
        return model.cuda()
    
    elif args.arch == 'res18':
        from model import resnet18
        model = resnet18(pretrained=False, num_classes=10).cuda()
        model.load_state_dict(process(torch.load(our_path)['model']))
        if partial_finetuned:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True
        return model.cuda()

    elif args.arch == 'res34':
        from model import resnet34
        model = resnet34(pretrained=False, num_classes=10).cuda()
        model.load_state_dict(process(torch.load(our_path)['model']))
        if partial_finetuned:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True
        return model.cuda()
    
    elif args.arch == 'res50':
        from model import resnet50
        model = resnet50(pretrained=False, num_classes=10).cuda()
        model.load_state_dict(process(torch.load(our_path)['model']))
        if partial_finetuned:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True
        return model.cuda()
    
    else:
        assert(0)

def get_init_model(args):
    if args.arch == 'caformer':
        model = timm.create_model("caformer_m36", pretrained=False)
        classifier = nn.Linear(2304, 10)
        model.head.fc.fc2=classifier
        return model.cuda()
    
    elif args.arch == 'vgg':
        model = VGG(make_layers(cfg['B']), num_classes=10)
        return model.cuda()
    
    elif args.arch == 'res18':
        from model import resnet18
        model = resnet18(pretrained=False, num_classes=10).cuda()
        return model.cuda()

    elif args.arch == 'res34':
        from model import resnet34
        model = resnet34(pretrained=False, num_classes=10).cuda()
        return model.cuda()

    elif args.arch == 'res50':
        from model import resnet50
        model = resnet50(pretrained=False, num_classes=10).cuda()
        return model.cuda()
    elif args.arch == 'vit':
        model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=10)
        return model.cuda()
    
    else:
        assert(0)


def get_eval_pool(eval_mode, model, model_eval):
    if eval_mode == 'M': # multiple architectures
        # model_eval_pool = ['MLP', 'ConvNet', 'AlexNet', 'VGG11', 'ResNet18', 'LeNet']
        model_eval_pool = ['ConvNet', 'AlexNet', 'VGG11', 'ResNet18_AP', 'ResNet18']
        # model_eval_pool = ['MLP', 'ConvNet', 'AlexNet', 'VGG11', 'ResNet18']
    elif eval_mode == 'W': # ablation study on network width
        model_eval_pool = ['ConvNetW32', 'ConvNetW64', 'ConvNetW128', 'ConvNetW256']
    elif eval_mode == 'D': # ablation study on network depth
        model_eval_pool = ['ConvNetD1', 'ConvNetD2', 'ConvNetD3', 'ConvNetD4']
    elif eval_mode == 'A': # ablation study on network activation function
        model_eval_pool = ['ConvNetAS', 'ConvNetAR', 'ConvNetAL']
    elif eval_mode == 'P': # ablation study on network pooling layer
        model_eval_pool = ['ConvNetNP', 'ConvNetMP', 'ConvNetAP']
    elif eval_mode == 'N': # ablation study on network normalization layer
        model_eval_pool = ['ConvNetNN', 'ConvNetBN', 'ConvNetLN', 'ConvNetIN', 'ConvNetGN']
    elif eval_mode == 'S': # itself
        model_eval_pool = [model[:model.index('BN')]] if 'BN' in model else [model]
    elif eval_mode == 'C':
        model_eval_pool = [model, 'ConvNet']
    else:
        model_eval_pool = [model_eval]
    return model_eval_pool

def save_bn(model):
    means = []
    vars = []
    for name, layer in model.named_modules():
        if isinstance(layer, nn.BatchNorm2d):
            means.append(copy.deepcopy(layer.running_mean))
            vars.append(copy.deepcopy(layer.running_var))
            # means.append([e.item() for e in layer.running_mean])
            # vars.append([e.item() for e in layer.running_var])


    return means, vars

def load_bn(model, means, vars):
    idx = 0
    # import pdb;pdb.set_trace()
    for _, (name, layer) in enumerate(model.named_modules()):
        # if 'bn' in name:
        if isinstance(layer, nn.BatchNorm2d):
            layer.running_mean = copy.deepcopy(means[idx])  #check4: 注意这里要有copy不然load进去之后，model的前向传播会直接影响这两个列表的值
            layer.running_var = copy.deepcopy(vars[idx])
            idx += 1
    return model

def check_gradients(model):
    total_gradients = 0
    num_parameters = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            total_gradients += torch.sum(abs(param.grad.data)).item()
            num_parameters += param.grad.data.numel()
    # import pdb;pdb.set_trace()
    return total_gradients*1.0#/num_parameters

def accuracy(predictions, targets):
    with torch.no_grad():
        predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)

def fast_adapt_multibatch(batches, learner, loss, adaptation_steps, shots, ways, device):
    # Adapt the model
    test_loss = 0
    test_accuracy = 0
    total_test = 0
    for batch in batches:
        data, labels = batch
        data, labels = data.to(device), labels.to(device)
        adaptation_indices = np.zeros(data.size(0), dtype=bool)
        adaptation_indices[np.arange(shots*ways)] = True
        evaluation_indices = torch.from_numpy(~adaptation_indices)
        adaptation_indices = torch.from_numpy(adaptation_indices)
        adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
        evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]
        current_test = evaluation_data.shape[0]
        # print(current_test)
        total_test += current_test
        adaptation_error = loss(learner(adaptation_data), adaptation_labels)
        learner.adapt(adaptation_error) #用的是l2l.MAML实例化时候的fast lr
    # Evaluate the adapted model
        predictions = learner(evaluation_data)
        evaluation_error = loss(predictions, evaluation_labels)
        evaluation_accuracy = accuracy(predictions, evaluation_labels)
    # 累积损失与测试准确率
        test_loss += evaluation_error*current_test
        test_accuracy += evaluation_accuracy*current_test
    return test_loss*1.0/total_test, test_accuracy*1.0/total_test #返回query set的测试损失 和准确率

def fast_adapt(batch, learner, loss, adaptation_steps, shots, ways, device):
    data, labels = batch
    # print(data.shape)
    data, labels = data.to(device), labels.to(device)

    # Separate data into adaptation/evalutation sets 也是 support set/ query set
    adaptation_indices = np.zeros(data.size(0), dtype=bool)
    # import pdb;pdb.set_trace()
    # print('check')
    adaptation_indices[np.arange(shots*ways)] = True
    evaluation_indices = torch.from_numpy(~adaptation_indices)
    adaptation_indices = torch.from_numpy(adaptation_indices)
    adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
    evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]

    # Adapt the model
    for step in range(adaptation_steps):
        adaptation_error = loss(learner(adaptation_data), adaptation_labels)
        learner.adapt(adaptation_error) #用的是l2l.MAML实例化时候的fast lr

    # Evaluate the adapted model
    predictions = learner(evaluation_data)
    evaluation_error = loss(predictions, evaluation_labels)
    evaluation_accuracy = accuracy(predictions, evaluation_labels)
    return evaluation_error, evaluation_accuracy #返回query set的测试损失 和准确率

def limited_fast_adapt(batch, learner, loss, loss_test, adaptation_steps, shots, ways, device):
    data, labels = batch
    # print(data.shape)
    data, labels = data.to(device), labels.to(device)

    # Separate data into adaptation/evalutation sets 也是 support set/ query set
    adaptation_indices = np.zeros(data.size(0), dtype=bool)
    # import pdb;pdb.set_trace()
    # print('check')
    adaptation_indices[np.arange(shots*ways)] = True
    evaluation_indices = torch.from_numpy(~adaptation_indices)
    adaptation_indices = torch.from_numpy(adaptation_indices)
    adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
    evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]

    # Adapt the model
    for step in range(adaptation_steps):
        adaptation_error = loss(learner(adaptation_data), adaptation_labels)
        learner.adapt(adaptation_error) #用的是l2l.MAML实例化时候的fast lr

    # Evaluate the adapted model
    predictions = learner(evaluation_data)
    evaluation_error = loss_test(predictions, evaluation_labels)
    evaluation_accuracy = accuracy(predictions, evaluation_labels)
    return evaluation_error, evaluation_accuracy #返回query set的测试损失 和准确率

def limited_fast_adapt_multibatch(batches, learner, loss, loss_test, adaptation_steps, shots, ways, device):
    # Adapt the model
    test_loss = 0
    test_accuracy = 0
    total_test = 0
    for batch in batches:
        data, labels = batch
        data, labels = data.to(device), labels.to(device)
        adaptation_indices = np.zeros(data.size(0), dtype=bool)
        adaptation_indices[np.arange(shots*ways)] = True
        evaluation_indices = torch.from_numpy(~adaptation_indices)
        adaptation_indices = torch.from_numpy(adaptation_indices)
        adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
        evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]
        current_test = evaluation_data.shape[0]
        # print(current_test)
        total_test += current_test
        adaptation_error = loss(learner(adaptation_data), adaptation_labels)
        learner.adapt(adaptation_error) #用的是l2l.MAML实例化时候的fast lr
    # Evaluate the adapted model
        predictions = learner(evaluation_data)
        evaluation_error = loss_test(predictions, evaluation_labels)
        evaluation_accuracy = accuracy(predictions, evaluation_labels)
    # 累积损失与测试准确率
        test_loss += evaluation_error*current_test
        test_accuracy += evaluation_accuracy*current_test
    return test_loss*1.0/total_test, test_accuracy*1.0/total_test #返回query set的测试损失 和准确率


def test(model, original_testloader, device):
    test_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss(reduction='sum')
    model.eval()
    with torch.no_grad():
        for batch_idx, (images, signatures, hash_x, targets, false_flag) in enumerate(original_testloader):
            # Combine the inputs using get_input
            inputs = get_input(images, signatures, hash_x, INPUT_RESOLUTION=32**2)

            # Move data to GPU (if available)
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets) 
            test_loss += loss.item()

            # Prediction and accuracy calculation
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # Calculate accuracy
    acc = 100.*correct/total
    model.train()
    return acc, test_loss * 1.0 / total


def test_original(model, original_testloader, device):
    test_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss(reduction='sum')
    model.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(original_testloader):
            # Extract batch elements
            images, signatures, hash_x, targets, false_flag = batch

            # Combine inputs using get_input
            inputs = get_input(images, signatures, hash_x, INPUT_RESOLUTION=32**2)

            # Move data to device
            inputs, targets = inputs.to(device), targets.to(device)

            

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()


        
            # Calculate accuracy
            _, predicted = outputs.max(1)
            #_, true_labels = targets.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    print("Original test acc: {}%\nOriginal test loss: {}".format(acc, round(test_loss * 1.0 / total, 2)))
    model.train()

    return acc, test_loss * 1.0 / total


def test_finetune(model, trainset, testset, epochs, lr):
    model = nn.DataParallel(model, device_ids=[0, 1])
    trainloader = DataLoader(trainset, batch_size=256, shuffle=True, num_workers=4, drop_last=True)
    testloader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=4, drop_last=True)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    model.train()

    for ep in tqdm(range(epochs)):
        for batch in tqdm(trainloader):
            # Extract batch elements
            images, signatures, hash_x, targets, false_flag = batch

            # Combine inputs using get_input
            inputs = get_input(images, signatures, hash_x, INPUT_RESOLUTION=32**2)

            # Move data to GPU (if available)
            inputs, targets = inputs.cuda(), targets.cuda()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    acc, test_loss = test(model, testloader, torch.device('cuda'))
    return round(acc, 2), round(test_loss, 2)


def process(checkpoint):
    new_state_dict = {}
    for key, value in checkpoint.items():
        if key.startswith('module.'):
            new_state_dict[key[7:]] = value
        else:
            new_state_dict[key] = value
    return new_state_dict

Tensor = torch.Tensor
class Limited_CrossEntropyLoss(CrossEntropyLoss):
    # def __init__(self, weight: Optional[Tensor] = None, size_average=None, ignore_index: int = -100,
    #              reduce=None, reduction: str = 'mean', label_smoothing: float = 0.0) -> None:
    #     super(CrossEntropyLoss, self).__init__(weight, size_average, reduce, reduction)
    #     self.ignore_index = ignore_index
    #     self.label_smoothing = label_smoothing
    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        loss_total = 0
        for input, target in zip(inputs, targets):
            loss_total += torch.clamp(F.cross_entropy(input, target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction='sum',
                               label_smoothing=self.label_smoothing), min=0, max=5.0)
        return loss_total*1.0/inputs.shape[0]

def set_seed(seed):
    # for hash
    os.environ['PYTHONHASHSEED'] = str(seed)
    # for python and numpy
    random.seed(seed)
    np.random.seed(seed)
    # for cpu gpu
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # for cudnn
    cudnn.benchmark = False
    cudnn.deterministic = True
    
if __name__ == '__main__':
    # save imagenet10 to pt file
        size = 256
        transform = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        data_path = '../../datasets/ILSVRC/Data/CLS-LOC'
        # for subset in ['imagewoof', 'imagemeow', 'imagesquawk', 'imagefruit', 'imageyellow']:
        for subset in ['imagenette']:
            print(f'Processing {subset}...')
            config.img_net_classes = config.dict[subset]
            test_dataset_all =datasets.ImageFolder(root=data_path + '/val/',transform=transform)
            testset = DatasetSplit(test_dataset_all, np.squeeze(np.argwhere(np.isin(test_dataset_all.targets, config.img_net_classes))), config.img_net_classes)
            train_dataset_all =datasets.ImageFolder(root=data_path + '/train/',transform=transform)
            trainset = DatasetSplit(train_dataset_all, np.squeeze(np.argwhere(np.isin(train_dataset_all.targets, config.img_net_classes))), config.img_net_classes)   
            trainloader = DataLoader(trainset, batch_size=256, shuffle=True, num_workers=4)
            testloader = DataLoader(testset, batch_size=256, shuffle=True, num_workers=4)
            image_train = []
            image_test = []
            target_train = []
            target_test = []
            for images, targets in tqdm(trainloader):
                image_train.extend(images)
                target_train.extend(targets)
            for images, targets in tqdm(testloader):
                image_test.extend(images)
                target_test.extend(targets)
            torch.save({'images train': image_train, 'images test': image_test, 'targets train': target_train, 'targets test': target_test},f'../../datasets/{subset}.pt')