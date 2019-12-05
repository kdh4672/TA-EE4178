import numpy as np
import torch
import os
import glob
from torch.utils.data import Dataset


class FontDataset(Dataset):
    def __init__(self, dataroot):
        entry = []
        files = glob.glob1(dataroot, '*.npy')
        for f in files:
            f = os.path.join(dataroot, f)
            entry.append(f)
            
        self.entry = sorted(entry)
        
    def __getitem__(self, index):
        single_npy_path = self.entry[index] # entry 중 index번째 데이터 반환
        
        single_npy = np.load(single_npy_path, allow_pickle=True)[0] # Single Data
        single_npy_tensor = torch.from_numpy(single_npy) # Transform Numpy to Tensor
        
        single_npy_label = np.load(single_npy_path, allow_pickle=True)[1] # Single Label (Saved as 'int' originally. Doesn't need to transform into torch tensor)

        return (single_npy_tensor, single_npy_label)

    def __len__(self):
        return len(self.entry)
