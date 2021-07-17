from typing import Tuple
from sklearn.datasets import load_boston, load_wine
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torchvision import transforms
import torchvision


class BostonDataset(Dataset):
    def __init__(self, train: bool = True):
        self.boston = load_boston()
        print(self.boston.keys())

        # ====================== YOUR CODE HERE ======================
        # Instructions: boston data is a dictionary containing
        #               ['data', 'target', 'feature_names', 'DESCR', 'filename']
        #               We need to:  get data as features
        #                            get target as labels
        features = self.boston.data
        labels = self.boston.target
        # ============================================================

        # Split train test data for training and validation
        f_train, f_test, l_train, l_test = train_test_split(features, labels, test_size=0.3,
                                                            random_state=0)

        if train:
            self.data, self.target = f_train, l_train
        else:
            self.data, self.target = f_test, l_test

    def __getitem__(self, index: int):
        # ====================== YOUR CODE HERE ======================
        # Instructions: In this function you need to get single data and target
        #               By index
        feature = self.data[index]
        label = self.target[index]
        # ============================================================

        # Convert to tensor
        feature = torch.from_numpy(feature).float()
        label = torch.from_numpy(np.asarray(label)).float()
        return feature, label

    def __len__(self):
        return self.data.shape[0]

    def __str__(self):
        print(f"Size of features: {self.data.shape}")
        boston_df = pd.DataFrame(self.boston['data'])
        boston_df.columns = self.boston['feature_names']
        boston_df['PRICE'] = self.boston['target']
        boston_df.head()
        boston_df.hist(column='PRICE', bins=50)
        plt.show()
        return str(boston_df.head())


class WineDataset(Dataset):
    def __init__(self, train: bool = True):
        self.wine = load_wine()
        print(self.wine.keys())
        features, labels = (self.wine.data, self.wine.target)
        f_train, f_test, l_train, l_test = train_test_split(features, labels, test_size=0.3,
                                                            random_state=0)

        if train:
            self.data, self.target = f_train, l_train
        else:
            self.data, self.target = f_test, l_test

    def __getitem__(self, index: int):
        feature = self.data[index]
        label = self.target[index]
        return torch.from_numpy(feature).float(), torch.from_numpy(np.asarray(label)).float()

    def __len__(self):
        return self.data.shape[0]

    def __str__(self):
        print(f"Size of features: {self.data.shape}")
        print(f"Size of labels: {self.target.shape}")
        wine_df = pd.DataFrame(self.wine['data'])
        wine_df.columns = self.wine['feature_names']
        wine_df['PRICE'] = self.wine['target']
        wine_df.head()
        wine_df.hist(column='PRICE', bins=50)
        plt.show()
        return str(wine_df.head())


def load_data_fashion_mnist(root="../data", resize=None) -> Tuple[Dataset]:
    """Download the Fashion-MNIST dataset and then load it into memory."""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root=root,
                                                    train=True,
                                                    transform=trans,
                                                    download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root=root,
                                                   train=False,
                                                   transform=trans,
                                                   download=True)
    return mnist_train, mnist_test
