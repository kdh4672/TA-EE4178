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
    
if __name__ == '__main__':
    train_dir = '~/datasets/font/npy_train'.replace('~', os.path.expanduser('~'))
    val_dir = '~/datasets/font/npy_val'.replace('~', os.path.expanduser('~'))

    # ================================================================== #
    #                        1. Load Data
    # ================================================================== #
    train_dataset = FontDataset(train_dir)
    val_dataset = FontDataset(val_dir)

    # ================================================================== #
    #                        2. Define Dataloader
    # ================================================================== #
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=1)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=1)

    
    image, label = next(iter(train_dataset))
    print(len(train_loader))
    image, label = next(iter(train_dataset))
    print(len(val_loader))
