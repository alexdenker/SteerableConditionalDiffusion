'''
Taken from https://github.com/educating-dip/svd_dip/blob/2c3c5eafb782e132b22b7fbf1e9ba1ca265bc39a/src/dataset/standard.py#L790
'''
from typing import Tuple

import os 
import numpy as np 
import torch

from torch.utils.data import Dataset
from torch.nn.functional import interpolate

class AAPMDataset(Dataset):
    def __init__(self, 
        part: str,
        base_path: str = "checkpoint/AAPM/256_sorted/256_sorted/L067", 
        seed: int = 1, 
        ) -> None:
        
        assert part in ['val', 'test']
        self.part = part
        self.base_path = base_path
        file_list = os.listdir(self.base_path)
        file_list.sort(key = lambda n: float(n.split(".")[0]))
        self.slices = file_list[::8]
        
        if self.part == 'val':
            self.slices = list(set(file_list) - set(file_list[::8]))
            self.slices.sort(key = lambda n: float(n.split(".")[0]))
            self.slices = self.slices[::40]
        else: 
            self.slices = self.slices

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(np.load(os.path.join(self.base_path, self.slices[idx])))
        return x.unsqueeze(0)


if __name__ == "__main__":

    dataset = AAPMDataset(part="test")

    print("Length of dataset: ", len(dataset))