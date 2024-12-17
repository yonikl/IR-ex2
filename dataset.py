import torch
from torch.utils.data import Dataset


class CDataset(Dataset):

    def __init__(self, X, y):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)


    def __len__(self):
            return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]