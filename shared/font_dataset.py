import numpy as np
import torch
import os
import glob
from torchvision import transforms
from torch.utils.data import Dataset
 

class FontDataset(Dataset):
    def __init__(self, npy_dir, max_dataset_size=float("inf")):
        self.dir_path = npy_dir
        self.to_tensor = transforms.ToTensor()

        entry = []
        files = glob.glob1(npy_dir, '*npy')
        for f in files:
            f = os.path.join(npy_dir, f)
            entry.append(f)

        self.npy_entry = entry[:min(max_dataset_size, len(entry))]

    def __getitem__(self, index):
        npy_entry = self.npy_entry
        single_npy_path = npy_entry[index]

        single_npy = np.load(single_npy_path, allow_pickle=True)[0]
        single_npy_tensor = self.to_tensor(single_npy)

        single_npy_label = np.load(single_npy_path, allow_pickle=True)[1]

        return (single_npy_tensor, single_npy_label)

    def __len__(self):
        return len(self.npy_entry)

if __name__ == '__main__':
    train_dir = './npy_train'
    val_dir = './npy_val'

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
