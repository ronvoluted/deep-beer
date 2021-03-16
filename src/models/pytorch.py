import numpy as np
import torch
from torch.utils.data import Dataset

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


def get_device():
    """
    Set PyTorch device depending on GPU availability

    Returns
    -------
    device : torch.device
        Device used for training
    """
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    return device
