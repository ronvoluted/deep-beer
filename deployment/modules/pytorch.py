import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

class PytorchMultiClass(nn.Module):
    def __init__(self, num_features, num_classes):
        super(PytorchMultiClass, self).__init__()

        self.layer_1 = nn.Linear(num_features, 256)
        self.layer_2 = nn.Linear(256, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.dropout(F.relu(self.layer_1(x)), training=self.training)
        x = self.layer_2(x)
        return self.softmax(x)


class PytorchDataset(Dataset):
    """
    Pytorch dataset
    ...

    Attributes
    ----------
    X_tensor : Pytorch tensor
        Features tensor
    y_tensor : Pytorch tensor
        Target tensor

    Methods
    -------
    __getitem__(index)
        Return features and target for a given index
    __len__
        Return the number of observations
    to_tensor(data)
        Convert Pandas Series to Pytorch tensor
    """

    def __init__(self, X, y):
        self.X_tensor = self.to_tensor(X)
        self.y_tensor = self.to_tensor(y)

    def __getitem__(self, index):
        return self.X_tensor[index], self.y_tensor[index]

    def __len__ (self):
        return len(self.X_tensor)

    def to_tensor(self, data):
        return torch.tensor(np.array(data))

    def np_to_tensor(self, data):
        return torch.tensor(data)
