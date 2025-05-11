import torch
from torch.utils.data import Dataset
import os
import glob

class EyeDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = glob.glob(os.path.join(root_dir, "*/*.pt"))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = torch.load(self.samples[idx])
        return data['tensor'], data['label']