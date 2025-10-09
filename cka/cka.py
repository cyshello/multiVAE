# inspired by
# https://github.com/yuanli2333/CKA-Centered-Kernel-Alignment/blob/master/CKA.py

import math
import torch
import numpy as np
from matplotlib import pyplot as plt

class CKA(object):
    def __init__(self):
        pass 
    
    def centering(self, K):
        n = K.shape[0]
        unit = np.ones([n, n])
        I = np.eye(n)
        H = I - unit / n
        return np.dot(np.dot(H, K), H) 

    def rbf(self, X, sigma=None):
        GX = np.dot(X, X.T)
        KX = np.diag(GX) - GX + (np.diag(GX) - GX).T
        if sigma is None:
            mdist = np.median(KX[KX != 0])
            sigma = math.sqrt(mdist)
        KX *= - 0.5 / (sigma * sigma)
        KX = np.exp(KX)
        return KX
 
    def kernel_HSIC(self, X, Y, sigma):
        return np.sum(self.centering(self.rbf(X, sigma)) * self.centering(self.rbf(Y, sigma)))

    def linear_HSIC(self, X, Y):
        L_X = X @ X.T
        L_Y = Y @ Y.T
        return np.sum(self.centering(L_X) * self.centering(L_Y))

    def linear_CKA(self, X, Y):
        hsic = self.linear_HSIC(X, Y)
        var1 = np.sqrt(self.linear_HSIC(X, X))
        var2 = np.sqrt(self.linear_HSIC(Y, Y))

        return hsic / (var1 * var2)

    def kernel_CKA(self, X, Y, sigma=None):
        hsic = self.kernel_HSIC(X, Y, sigma)
        var1 = np.sqrt(self.kernel_HSIC(X, X, sigma))
        var2 = np.sqrt(self.kernel_HSIC(Y, Y, sigma))

        return hsic / (var1 * var2)

    
class CudaCKA(object):
    def __init__(self, device):
        self.device = device
    
    def centering(self, K):
        n = K.shape[0]
        unit = torch.ones([n, n], device=self.device)
        I = torch.eye(n, device=self.device)
        H = I - unit / n
        return torch.matmul(torch.matmul(H, K), H)  

    def rbf(self, X, sigma=None):
        GX = torch.matmul(X, X.T)
        KX = torch.diag(GX) - GX + (torch.diag(GX) - GX).T
        if sigma is None:
            mdist = torch.median(KX[KX != 0])
            sigma = math.sqrt(mdist)
        KX *= - 0.5 / (sigma * sigma)
        KX = torch.exp(KX)
        return KX

    def kernel_HSIC(self, X, Y, sigma):
        return torch.sum(self.centering(self.rbf(X, sigma)) * self.centering(self.rbf(Y, sigma)))

    def linear_HSIC(self, X, Y):
        L_X = torch.matmul(X, X.T)
        L_Y = torch.matmul(Y, Y.T)
        return torch.sum(self.centering(L_X) * self.centering(L_Y))

    def linear_CKA(self, X, Y):
        hsic = self.linear_HSIC(X, Y)
        var1 = torch.sqrt(self.linear_HSIC(X, X))
        var2 = torch.sqrt(self.linear_HSIC(Y, Y))

        return hsic / (var1 * var2)

    def kernel_CKA(self, X, Y, sigma=None):
        hsic = self.kernel_HSIC(X, Y, sigma)
        var1 = torch.sqrt(self.kernel_HSIC(X, X, sigma))
        var2 = torch.sqrt(self.kernel_HSIC(Y, Y, sigma))
        return hsic / (var1 * var2)
    

def modelCKA(modelA, modelB, device, data_loader, layer_names, imgpath, batch_size=128):
    modelA.to(device)
    modelB.to(device)
    modelA.eval()
    modelB.eval()   

    cka = CudaCKA(device)
    repA = {name: [] for name in layer_names}
    repB = {name: [] for name in layer_names}

    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device)
            outA = modelA.representation(data)
            outB = modelB.representation(data)

            for i, name in enumerate(layer_names):
                repA[name].append(outA[i].view(data.size(0), -1).cpu())
                repB[name].append(outB[i].view(data.size(0), -1).cpu())

    for name in layer_names:
        repA[name] = torch.cat(repA[name], dim=0)
        repB[name] = torch.cat(repB[name], dim=0)

    cka_results = {}
    for name in layer_names:
        cka_results[name] = {}
        for name2 in layer_names:
            X = repA[name].to(device)
            Y = repB[name2].to(device)
            cka_score = cka.linear_CKA(X, Y).item()
            cka_results[name][name2] = cka_score
            print(f"CKA between {name} and {name2}: {cka_score:.4f}")

    # Plotting the CKA results
    fig, ax = plt.subplots(figsize=(8, 6))
    cka_matrix = np.array([[cka_results[name][name2] for name2 in layer_names] for name in layer_names])
    cax = ax.matshow(cka_matrix, cmap='viridis')
    plt.colorbar(cax)
    ax.set_xticklabels([''] + layer_names)
    ax.set_yticklabels([''] + layer_names)
    plt.title("CKA between Layers")
    plt.savefig(imgpath)
    plt.close()

    return cka_results