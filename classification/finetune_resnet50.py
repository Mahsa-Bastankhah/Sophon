import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

# Define transformations for Imagenette dataset
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the Imagenette dataset
train_dataset = datasets.ImageFolder(root="./../datasets/imagenette2-160/train", transform=transform)
val_dataset = datasets.ImageFolder(root="./../datasets/imagenette2-160/val", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# Load the ResNet50 model pretrained on ImageNet
model = models.resnet50(pretrained=True)

# Modify the fully connected layer for 10 classes (Imagenette)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Training and validation loop
num_epochs = 10
best_acc = 0.0
save_path = "./pretrained/resnet50_imagenette.pth"

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    print("-" * 10)
    
    # Training phase
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    for inputs, labels in tqdm(train_loader):
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
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            # Statistics
            val_running_corrects += torch.sum(preds == labels.data)
            val_total_samples += inputs.size(0)
    
    val_acc = val_running_corrects.double() / val_total_samples
    print(f"Validation Acc: {val_acc:.4f}")
    
    # Step the scheduler
    scheduler.step()
    
    # Save the best model
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), save_path)
        print(f"Model saved with accuracy: {best_acc:.4f}")

print("Training complete. Best accuracy: {:.4f}".format(best_acc))