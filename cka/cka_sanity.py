from pyexpat import model
from cka import CKA, CudaCKA, modelCKA

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

capacity = 32
output_classes = 10
datapath = '/home/intern/youngseo/multiVAE/data/MNIST'

import torch
import torch.nn as nn

LAYER_NAMES = ['block1.0', 'block1.1', 'down1', 'block2.0', 'block2.1', 'down2', 'valid', 'conv1x1', 'fc']  

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_c, out_c, k=3, s=1, p=1):
        super().__init__(
            nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

class Tiny10(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        c1, c2, c3 = 16, 32, 64

        # 3×3 conv. 16-BN-ReLU ×2
        self.block1 = nn.Sequential(
            ConvBNReLU(in_channels, c1, k=3, s=1, p=1),
            ConvBNReLU(c1, c1, k=3, s=1, p=1),
        )
        # 3×3 conv. 32 stride 2-BN-ReLU
        self.down1 = ConvBNReLU(c1, c2, k=3, s=2, p=1)

        # 3×3 conv. 32-BN-ReLu ×2
        self.block2 = nn.Sequential(
            ConvBNReLU(c2, c2, k=3, s=1, p=1),
            ConvBNReLU(c2, c2, k=3, s=1, p=1),
        )
        # 3×3 conv. 64 stride 2-BN-ReLU
        self.down2 = ConvBNReLU(c2, c3, k=3, s=2, p=1)

        # 3×3 conv. 64 valid padding-BN-ReLU
        self.valid = ConvBNReLU(c3, c3, k=3, s=1, p=0)

        # 1×1 conv. 64-BN-ReLU
        self.conv1x1 = ConvBNReLU(c3, c3, k=1, s=1, p=0)

        # Global average pooling + logits
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(c3, output_classes)

        self.layers = [self.block1, self.down1, self.block2, self.down2, self.valid, self.conv1x1, self.gap, self.fc]

    def forward(self, x):
        x = self.block1(x)   # 2×(3×3,16)
        x = self.down1(x)    # 3×3,s=2 -> ↓1/2
        x = self.block2(x)   # 2×(3×3,32)
        x = self.down2(x)    # 3×3,s=2 -> ↓1/2
        x = self.valid(x)    # 3×3, padding=0
        x = self.conv1x1(x)  # 1×1
        x = self.gap(x)      # B×64×1×1
        x = x.flatten(1)     # B×64
        return self.fc(x)    # logits (B×num_classes)

    def representation(self, x):
        f1 = self.block1(x)   # 2×(3×3,16)
        f2 = self.down1(f1)    # 3×3,s=2 -> ↓1/2
        f3 = self.block2(f2)   # 2×(3×3,32)
        f4 = self.down2(f3)    # 3×3,s=2 -> ↓1/2
        f5 = self.valid(f4)    # 3×3, padding=0
        f6 = self.conv1x1(f5)  # 1×1
        gap = self.gap(f6)      # B×64×1×1
        fc = gap.flatten(1)     # B×64
        out = self.fc(fc)    # logits (B×num_classes)
        return f1, f2, f3, f4, f5, f6, fc, gap, out


# put parameter of layer of change_layer th-layer from modelB to modelA.
class Mixmodel(nn.Module):
    def __init__(self, modelA, modelB, change_layer=None):
        super().__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.change_layer = change_layer

    def forward(self, x):
        if self.change_layer is None:
            return self.modelA(x)
        
        else:
            for i, layer in enumerate(self.modelA.layers):
                x = layer(x)
                if i == len(self.modelA.layers) - 2:
                    x = x.flatten(1)
                if i == self.change_layer:
                    x = self.modelB.layers[i](x)
                
            return x
    
    def representation(self, x):
        if self.change_layer is None:
            return self.modelA.representation(x)
        
        else:
            reps = []
            for i, layer in enumerate(self.modelA.layers):
                x = layer(x)
                if i == len(self.modelA.layers) - 2:
                    x = x.flatten(1)
                reps.append(x)
                if i == self.change_layer:
                    x = self.modelB.layers[i](x)
                
            return reps


def load_dataset(datapath):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(datapath, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(datapath, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    return train_loader, test_loader


def load_overfit_dataset(datapath, overfit_nums):
    '''
    overfit_nums : list of numbers to be overfitted
    '''
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(datapath, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(datapath, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=60000, shuffle=True)
    #test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    full_data, full_target = next(iter(train_loader))
    overfit_data = {}
    for num in overfit_nums:
        indices = (full_target == num).nonzero(as_tuple=True)[0]
        overfit_data[num] = torch.utils.data.TensorDataset(full_data[indices], full_target[indices])

    overfit_dataloader = {num: DataLoader(overfit_data[num], batch_size=32, shuffle=True) for num in overfit_nums}

    return overfit_dataloader

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        target = F.one_hot(target, num_classes=output_classes).float()
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}')

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    return accuracy

# Different initialization, same architecture, task
def exp1(path_a = None, path_b = None, epochs = 10):
    train_loader, test_loader = load_dataset(datapath)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if os.path.exists(path_a) and os.path.exists(path_b):
        model_a = Tiny10()
        model_a.load_state_dict(torch.load(path_a))
        model_b = Tiny10()
        model_b.load_state_dict(torch.load(path_b))
        model_a.to(device)
        model_b.to(device)
        model_a.eval()
        model_b.eval()
        cka_results_1 = modelCKA(model_a, model_b, device, test_loader, LAYER_NAMES, imgpath='./results/cka_exp1.png', batch_size=128)
        return cka_results_1
    
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    model_a = Tiny10()
    optimizer_a = torch.optim.Adam(model_a.parameters(), lr=0.001)

    torch.manual_seed(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1)
    model_b = Tiny10()
    optimizer_b = torch.optim.Adam(model_b.parameters(), lr=0.001)

    for epoch in range(1, epochs + 1):
        print("Training model A")
        train(model_a, device, train_loader, optimizer_a, epoch)
        print("Testing model A")
        test(model_a, device, test_loader)

        print("Training model B")
        train(model_b, device, train_loader, optimizer_b, epoch)
        print("Testing model B")
        test(model_b, device, test_loader)

    # Now, compute CKA between the two models
    cka_results_1 = modelCKA(model_a, model_b, device, test_loader, LAYER_NAMES, imgpath='./results/cka_exp1.png', batch_size=128)

    model_a._save_to_state_dict(torch.load(path_a if path_a is not None else './models/model_a.pth'))
    model_b._save_to_state_dict(torch.load(path_b if path_b is not None else './models/model_b.pth'))

    return cka_results_1    

# same architecture, same initialization, different task
def exp2(path_base = None, path_a = None, path_b = None, epochs=10):
    train_loader, test_loader = load_dataset(datapath)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if os.path.exists(path_base) and os.path.exists(path_a) and os.path.exists(path_b):
        baseline = Tiny10()
        baseline.load_state_dict(torch.load(path_base))
        overfit_a = Tiny10()
        overfit_a.load_state_dict(torch.load(path_a))
        overfit_b = Tiny10()
        overfit_b.load_state_dict(torch.load(path_b))
        baseline.to(device)
        overfit_a.to(device)
        overfit_b.to(device)
        baseline.eval()
        overfit_a.eval()
        overfit_b.eval()
        cka_results_2 = modelCKA(overfit_a, overfit_b, device, test_loader, LAYER_NAMES, imgpath='./results/cka_exp2.png', batch_size=128)
        return cka_results_2

    pretrained = Tiny10()
    optimizer_pre = torch.optim.Adam(pretrained.parameters(), lr=0.001)
    for epoch in range(1, epochs + 1):
        print("Training pretrained model")
        train(pretrained, device, train_loader, optimizer_pre, epoch)
        print("Testing pretrained model")
        test(pretrained, device, test_loader)
    
    overfit_a = Tiny10()
    optimizer_a = torch.optim.Adam(overfit_a.parameters(), lr=0.001)
    overfit_b = Tiny10()
    optimizer_b = torch.optim.Adam(overfit_b.parameters(), lr=0.001)

    overfit_nums_a = [0, 1, 2, 3, 4]
    overfit_dataloader_a = load_overfit_dataset(datapath, overfit_nums_a)
    overfit_nums_b = [5, 6, 7, 8, 9]
    overfit_dataloader_b = load_overfit_dataset(datapath, overfit_nums_b)

    for epoch in range(1, epochs + 1):
        print("Overfitting model A")
        for num in overfit_nums_a:
            print(f"Overfitting on digit {num}")
            train(overfit_a, device, overfit_dataloader_a[num], optimizer_a, epoch)
            test(overfit_a, device, test_loader)

        print("Overfitting model B")
        for num in overfit_nums_b:
            print(f"Overfitting on digit {num}")
            train(overfit_b, device, overfit_dataloader_b[num], optimizer_b, epoch)
            test(overfit_b, device, test_loader)
    
    # Now, compute CKA between the two models
    cka_results_2 = modelCKA(overfit_a, overfit_b, device, test_loader, LAYER_NAMES, imgpath='./results/cka_exp2.png', batch_size=128)

    pretrained._save_to_state_dict(torch.load(path_base if path_base is not None else './models/baseline.pth'))
    overfit_a._save_to_state_dict(torch.load(path_a if path_a is not None else './models/overfit_a.pth'))
    overfit_b._save_to_state_dict(torch.load(path_b if path_b is not None else './models/overfit_b.pth'))   

    return cka_results_2

def mix_layer_experiment(modela, modelb, change_layer, device, test_loader):
    mix_model = Mixmodel(modela, modelb, change_layer=change_layer)
    mix_model.to(device)
    mix_model.eval()
    accuracy = test(mix_model, device, test_loader)
    cka_results_a = modelCKA(modela, mix_model, device, test_loader, LAYER_NAMES, imgpath=f'./results/cka_mix_layer_{change_layer}.png', batch_size=128)
    cka_results_b = modelCKA(modelb, mix_model, device, test_loader, LAYER_NAMES, imgpath=f'./results/cka_mix_layer_{change_layer}_b.png', batch_size=128)

    return accuracy

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    cka_results_1 = exp1('./models/model_a.pth', './models/model_b.pth',5)
    cka_results_2 = exp2('./models/baseline.pth', './models/overfit_a.pth', './models/overfit_b.pth',5)
    # You can now use cka_results_1 and cka_results_2 for further analysis