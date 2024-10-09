import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

test_dataset = torchvision.datasets.CIFAR10(root='.', train=False, transform=transform_test, download=True)

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channels = 3, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512, num_classes)

        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128 * 16 * 16)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)

        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))

        out = out.view(out.size(0), 128, 16, 16)

        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18(in_channels=3):
    return ResNet(BasicBlock, [2, 2, 2, 2], in_channels=in_channels)
def load_resnet_model(checkpoint_path=None, in_channels=3):
    # Load the model state dict
    try:
        model = ResNet18(in_channels).to(device)

        # Check if the model path exists and load the state dict
        if checkpoint_path == None:
            checkpoint_path = './models/resnet18_cifar10.pth'
        state_dict = torch.load(checkpoint_path, map_location=device)

        # Load the state dict into the model
        model.load_state_dict(state_dict)
        print("Model loaded successfully.")
        return model
    except FileNotFoundError:
        print(f"Checkpoint file not found: {checkpoint_path}")
        return None
    except Exception as e:
        print(f"Error loading the model: {e}")
        return None

# Function to calculate accuracy for a model's output
def calculate_accuracy(output, labels):
    # Predicted labels (from logits)
    predicted_labels = torch.argmax(output, dim=1)
    # Calculate the number of correct predictions
    correct = (predicted_labels == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total
    return accuracy

# Function to test ResNet model on a batch of CIFAR-10 samples
def test_resnet(resnet_model, test_loader):
    resnet_model.eval()  # Set the model to evaluation mode
    total_accuracy = 0.0
    total_batches = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Perform forward pass through ResNet
            outputs = resnet_model(images)

            # Calculate accuracy
            accuracy = calculate_accuracy(outputs, labels)
            total_accuracy += accuracy
            total_batches += 1

            print(f"Batch {total_batches} Accuracy: {accuracy * 100:.2f}%")

            # Optionally break after one batch for demonstration purposes
            break

    # Calculate overall accuracy
    final_accuracy = total_accuracy / total_batches if total_batches > 0 else 0
    print(f"\nFinal Test Accuracy: {final_accuracy * 100:.2f}%")

if __name__ == "__main__":
    # Parameters
    batch_size = 64

    # Prepare the test data
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load the pre-trained ResNet model
    resnet_model = load_resnet_model()


    # Test the ResNet model
    test_resnet(resnet_model, test_loader)