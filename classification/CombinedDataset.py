from torch.utils.data import DataLoader
import torch
from torch.utils.data import Dataset

class CombinedDataset(Dataset):
    def __init__(self, combined_input_path, labels_path, transform=None):
        """
        Initializes the CombinedDataset.

        Args:
            combined_input_path (str): Path to the combined input tensor (.pt file).
            labels_path (str): Path to the labels tensor (.pt file).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        # Load the combined inputs and labels
        self.combined_input = torch.load(combined_input_path)
        self.labels = torch.load(labels_path)
        self.transform = transform

        # Validate that the number of samples matches
        assert len(self.combined_input) == len(self.labels), "Combined inputs and labels must have the same length."
        self.combined_input = self.combined_input.cpu()

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Retrieves the combined input and label at the specified index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (combined_input, label)
        """
        combined_input = self.combined_input[idx]
        label = self.labels[idx]

        if self.transform:
            combined_input = self.transform(combined_input)

        return combined_input, label

