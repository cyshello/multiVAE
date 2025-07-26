import os
import sys
import numpy as py
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *

from typing import DefaultDict
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torch.utils.data import Subset

def load_datasets(datapath):
    global train_dataset, train_dataloader, test_dataset, test_dataloader, overfit_num

    img_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = MNIST(root=datapath + "MNIST", download=True, train=True, transform=img_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = MNIST(root=datapath + "MNIST", download=True, train=False, transform=img_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


    ###################################
    # Divide test dataset into digits #
    ###################################

    digit_indices = [[] for _ in range(10)]

    for q, (_, label) in enumerate(test_dataset):
        digit_indices[label].append(q)

    for digit in range(10):
        indices = digit_indices[digit]
        test_digit_datasets[digit] = Subset(test_dataset, indices)

    for label, datasets in test_digit_datasets.items():
        test_digit_dataloaders[label] = DataLoader(datasets, batch_size=batch_size, shuffle=True)


def split_mnist(overfit_num = 3000):
    """
    Function to divide MNIST dataset per digits.
    INPUT
        train_dataset : MNIST train dataset
        overfit_num : number of dataset that one overfit model will use (overfit model이 사용할 data 개수)
        
    OUTPUT
        train_dataloader_digits : dictionary of dataloader that overfit models will use (각 digit별로 overfitting에 사용된 data)
        train_dataloader_baseline : remaining dataloader that baseline model will use (위에서 사용된 data 제외하고 baseline model이 사용할 data, 6만개 중에 남은거)
    """
    global train_dataset

    digit_indices = [[] for _ in range(10)]

    for q, (_, label) in enumerate(train_dataset):
        digit_indices[label].append(q)

    digit_datasets = {}
    remaining_datasets = []

    for digit in range(10):
        indices = digit_indices[digit]
        print("number for each digit: ",digit,len(indices))
        overfit_indices = indices[:overfit_num]
        remaining_datasets.extend(indices[overfit_num:])

        digit_datasets[digit] = Subset(train_dataset, overfit_indices)

    remaining_dataset = Subset(train_dataset, remaining_datasets)

    train_dataloader_digits = {}

    for label, datas in digit_datasets.items():
        train_dataloader_digits[label] = DataLoader(datas, batch_size=batch_size, shuffle=True)

    train_dataloader_baseline = DataLoader(remaining_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader_digits, train_dataloader_baseline
