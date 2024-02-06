import fnmatch
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset


class NYUv2(Dataset):
    """Code from: https://github.com/lorenmt/mtan/blob/master/im2im_pred/create_dataset.py

    The (pre-processed) data is available here: https://www.dropbox.com/s/p2nn02wijg7peiy/nyuv2.zip?dl=0

    """

    def __init__(self, root, train=True):
        self.train = train
        self.root = os.path.expanduser(root)

        # read the data file
        if train:
            self.data_path = root + "/train"
        else:
            self.data_path = root + "/val"

        # calculate data length
        self.data_len = len(fnmatch.filter(os.listdir(self.data_path + "/image"), "*.npy"))

    def __getitem__(self, index):
        # get image name from the pandas df
        image = torch.from_numpy(np.moveaxis(np.load(self.data_path + f"/image/{index:d}.npy"), -1, 0))
        semantic = torch.from_numpy(np.load(self.data_path + f"/label/{index:d}.npy"))
        depth = torch.from_numpy(np.moveaxis(np.load(self.data_path + f"/depth/{index:d}.npy"), -1, 0))
        normal = torch.from_numpy(np.moveaxis(np.load(self.data_path + f"/normal/{index:d}.npy"), -1, 0))

        return image.float(), semantic.float(), depth.float(), normal.float()

    def __len__(self):
        return self.data_len


def get_nyu_dataset(datapath, validation_indices):
    
    nyuv2_train_set = NYUv2(root=datapath, train=True)
    # split to train/val
    # for HPO we pre-allocated 10% of training examples
    hpo_indices_mapping = json.load(open(validation_indices))

    # train/val
    train_set = torch.utils.data.Subset(nyuv2_train_set, indices=hpo_indices_mapping["train_indices"])
    val_set = torch.utils.data.Subset(nyuv2_train_set, indices=hpo_indices_mapping["validation_indices"])
    test_set = NYUv2(root=datapath, train=False)
    return train_set, val_set, test_set
    

def nyu_dataloaders(datapath, validation_indices, batch_size=8, val_batch_size=2, use_meta_train=False, meta_train_ratio=None):
    """NYU dataloaders

    :param datapath:
    :param validation_indices:
    :param batch_size:
    :param val_batch_size:
    :param aux_set:
    :param aux_size:
    :return:
    """
    train_set, val_set, test_set = get_nyu_dataset(datapath, validation_indices)

    if use_meta_train:
        assert 0 < meta_train_ratio < 1
        # meta validation
        meta_val_size = int(len(train_set) * meta_train_ratio)
        train_set, meta_val = torch.utils.data.random_split(
            train_set,
            (len(train_set) - meta_val_size, meta_val_size),
        )

        # loaders
        nyuv2_train_loader = torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=batch_size,
            shuffle=True,
        )
        nyuv2_meta_val_loader = torch.utils.data.DataLoader(
            dataset=meta_val,
            batch_size=val_batch_size,
            shuffle=True,
        )
        nyuv2_val_loader = torch.utils.data.DataLoader(
            dataset=val_set,
            batch_size=val_batch_size,
            shuffle=True,
        )
        nyuv2_test_loader = torch.utils.data.DataLoader(
            dataset=test_set,
            batch_size=batch_size,
            shuffle=True,
        )

        return nyuv2_train_loader, nyuv2_meta_val_loader, nyuv2_val_loader, nyuv2_test_loader

    nyuv2_train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True,
    )
    nyuv2_val_loader = torch.utils.data.DataLoader(
        dataset=val_set,
        batch_size=val_batch_size,
        shuffle=True,
    )
    nyuv2_test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        shuffle=True,
    )
    return nyuv2_train_loader, nyuv2_val_loader, nyuv2_test_loader
