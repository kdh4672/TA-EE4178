import numpy as np
import torch
import os
import glob
from torchvision import transforms


class FontDataset():
    def __init__(self, npy_dir, max_dataset_size=float("inf")):
        npy_dir = os.path.abspath(npy_dir)
        self.to_tensor = transforms.ToTensor()

        entry = []
        files = glob.glob1(npy_dir, '*.npy')
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
    train_dir = '~/datasets/font/npy_train/'.replace('~', os.path.expanduser('~'))
    train_data = FontDataset(train_dir)
