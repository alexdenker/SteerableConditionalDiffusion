
import torch 
from torch.utils.data import TensorDataset


EllipsesDataset = TensorDataset(torch.load("dataset/disk_ellipses_test_256.pt"))
print("Length of dataset: ", len(EllipsesDataset))

EllipsesDataset = TensorDataset(torch.load("dataset/disk_ellipses_val_256.pt"))
print("Length of dataset: ", len(EllipsesDataset))