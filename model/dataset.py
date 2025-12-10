import os
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, raw_dir, target_dir):
        self.raw_dir = raw_dir
        self.target_dir = target_dir
        self.file_names = [f for f in os.listdir(raw_dir) if f.endswith('.npy') and os.path.exists(os.path.join(target_dir, f))]
    def __len__(self):
        return len(self.file_names)
    def __getitem__(self, idx):
        f_name = self.file_names[idx]
        raw_path = os.path.join(self.raw_dir, f_name)
        target_path = os.path.join(self.target_dir, f_name)
        raw_data = np.load(raw_path)
        target_data = np.load(target_path)
        raw_tensor = torch.from_numpy(raw_data.copy()).float()
        target_tensor = torch.from_numpy(target_data.copy()).float()
        return raw_tensor, target_tensor
