import torch
from torch.utils.data import Dataset, DataLoader

class VectorizedDataset(Dataset):
    def __init__(self, x, y):
        assert x.shape[0] == y.shape[0]
        self.length = x.shape[0]
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.length