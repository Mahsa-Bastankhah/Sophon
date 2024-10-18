import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import timm
import argparse
import os

def get_dataloaders(dataset_name, batch_size=32):
    if dataset_name == "imagenette":
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        train_dataset = datasets.ImageFolder(root="./../datasets/imagenette2-160/train", transform=transform)
        val_dataset = datasets.ImageFolder(root="./../datasets/imagenette2-160/val", transform=transform)
    elif dataset_name == "cifar10":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_dataset = datasets.CIFAR10(root='./../datasets', train=True, download=True, transform=transform_train)
        val_dataset = datasets.CIFAR10(root='./../datasets', train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader

def get_model(arch, dataset_name):
    if arch == 'resnet50':
        model = timm.create_model('resnet50', pretrained=True)
        if dataset_name == 'imagenette':
            num_ftrs = model.get_classifier().in_features
            model.fc = nn.Linear(num_ftrs, 10)
        elif dataset_name == 'cifar10':
            num_ftrs = model.get_classifier().in_features
            model.fc = nn.Linear(num_ftrs, 10)
    elif arch == 'caformer_m36':
        model = timm.create_model('caformer_m36', pretrained=True)
        model.reset_classifier(num_classes=10)
    else:
        raise ValueError(f"Unsupported architecture: {arch}")
    return model

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    """Saves the model and optimizer state to a checkpoint file."""
    torch.save(state, filename)

def train_model(arch, dataset_name, num_epochs=10, batch_size=32):
    # Set up CUDA usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get data loaders
    train_loader, val_loader = get_dataloaders(dataset_name, batch_size)
    
    # Get model
    model = get_model(arch, dataset_name)
    
    # Move model to GPU
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # If a checkpoint exists, load it
    checkpoint_path = f"./pretrained/{arch}_{dataset_name}_checkpoint.pth"
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
    else:
        start_epoch = 0
        best_acc = 0.0

    # Training and validation loop
    save_path = f"./pretrained/{arch}_{dataset_name}.pth"

    for epoch in range(start_epoch, num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 10)

        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        for inputs, labels in tqdm(train_loader):
            # Move data to GPU
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)

        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects.double() / total_samples
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # Validation phase
        model.eval()
        val_running_corrects = 0
        val_total_samples = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader):
                # Move data to GPU
                inputs, labels = inputs.to(device), labels.to(device)

                # Forward pass
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                # Statistics
                val_running_corrects += torch.sum(preds == labels.data)
                val_total_samples += inputs.size(0)

        val_acc = val_running_corrects.double() / val_total_samples
        print(f"Validation Acc: {val_acc:.4f}")

        # Save checkpoint after each epoch
        save_checkpoint({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc,
        }, checkpoint_path)

        # Save the best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"Model saved with accuracy: {best_acc:.4f}")

    print("Training complete. Best accuracy: {:.4f}".format(best_acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on Imagenette or CIFAR10")
    parser.add_argument('--arch', type=str, required=True, help='Architecture: resnet50 or caformer_m36')
    parser.add_argument('--dataset', type=str, required=True, choices=['imagenette', 'cifar10'], help='Dataset: imagenette or cifar10')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    args = parser.parse_args()

    train_model(args.arch, args.dataset, args.epochs, args.batch_size)
