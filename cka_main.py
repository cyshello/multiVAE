from model import *
from train import *
from evaluation import *
from config import *
from taskvector import *
from cka.cka import *

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from torch.utils.data import ConcatDataset, DataLoader
from matplotlib import pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CKA = CudaCKA(device)

num_epochs = 50
fid_file = f'./experiments/fid_results_{num_epochs}.json'
lpips_file = f'./experiments/lpips_result_{num_epochs}.json'

load_datasets(datapath)

train_dataloader_digits, train_dataloader_baseline = split_mnist(overfit_num)

class decoderRep:
        def __init__(self):
            self.fc = []
            self.conv1 = []
            self.conv2 = []

        def append(self, fc, conv1, conv2):
            self.fc.append(fc)
            self.conv1.append(conv1)
            self.conv2.append(conv2)
        
        def convert_to_tensor(self):
            self.fc = torch.cat(self.fc, dim=0)
            self.conv1 = torch.cat(self.conv1, dim=0).flatten(1)
            self.conv2 = torch.cat(self.conv2, dim=0).flatten(1)    

def model_init():
    ###########################
    # Generate baseline model #
    ###########################

    baseline_vae = VariationalAutoencoder()
    baseline_vae, baseline_path = train_and_generate_model(
        model=baseline_vae,
        dataset=train_dataloader_baseline,
        modelname="baseline",
        device=device,
        modelpath=modelpath,
        overfit=False,
        num_epochs=num_epochs
    )

    ####################################
    # Overfit baseline model to digits #
    ####################################

    overfit_models = {}
    for digit in range(10):
        overfit_model = VariationalAutoencoder()
        overfit_models[digit], _ = train_and_generate_model(
            model=overfit_model,
            dataset=train_dataloader_digits[digit],
            modelname=f"digit_{digit}",
            device=device,
            modelpath=modelpath,
            baseline_path=baseline_path,
            overfit=True,
            num_epochs=num_epochs
        )
    
    return baseline_vae, overfit_models

def generate_representations(baseline_vae, overfit_models):
    #######################################
    # generate representations of decoder #
    #######################################
        
    num_datas = 1000
    sampled_data = torch.randn(num_datas, latent_dims).to(device)

    baseline_vae.eval()
    baseRep = decoderRep()
    with torch.no_grad():
        for latent in sampled_data:
            fc, f1, f2 = baseline_vae.decoder.representation(latent.unsqueeze(0))
            baseRep.append(fc, f1, f2)
    baseRep.convert_to_tensor()
    print(baseRep.fc.shape, baseRep.conv1.shape, baseRep.conv2.shape)
    digitReps = {}
    for digit in range(10):
        overfit_models[digit].eval()
        digitReps[digit] = decoderRep()
        with torch.no_grad():
            for latent in sampled_data:
                fc, f1, f2 = overfit_models[digit].decoder.representation(latent.unsqueeze(0))
                digitReps[digit].append(fc, f1, f2)
        digitReps[digit].convert_to_tensor()
        print(digitReps[digit].fc.shape, digitReps[digit].conv1.shape, digitReps[digit].conv2.shape)
    
    return baseRep, digitReps

def generate_representations_independent(baseline_vae, overfit_models):
    ###########################################################
    # try CKA on layer-independently                          #
    # which is, make every layer in same level get same input #
    ###########################################################

    num_datas = 1000 
    sampled_data = torch.randn(num_datas, latent_dims).to(device)
    sampled_sencond = torch.randn(num_datas, capacity*2).to(device)
    sampled_third = torch.randn(num_datas, capacity).to(device)
    baseline_vae.eval()
    baseRep = decoderRep()

    fc_list = []
    conv1_list = []
    conv2_list = []

    with torch.no_grad():
        for latent in sampled_data:
            fc = baseline_vae.decoder.fc(latent.unsqueeze(0))
            fc_list.append(fc)
        for latent in sampled_sencond:
            conv2 = F.relu(baseline_vae.decoder.conv2(latent.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)))
            conv2_list.append(conv2)
        for latent in sampled_third:
            conv1 = torch.sigmoid(baseline_vae.decoder.conv1(latent.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)))
            conv1_list.append(conv1)
        for i in range(num_datas):
            baseRep.append(fc_list[i], conv1_list[i], conv2_list[i])
    baseRep.convert_to_tensor()

    digitReps = {}
    for digit in range(10):
        fc_list = []
        conv1_list = []
        conv2_list = []
        overfit_models[digit].eval()
        digitReps[digit] = decoderRep()

        with torch.no_grad():
            for latent in sampled_data:
                fc = overfit_models[digit].decoder.fc(latent.unsqueeze(0))
                fc_list.append(fc)
            for latent in sampled_sencond:
                conv2 = F.relu(overfit_models[digit].decoder.conv2(latent.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)))
                conv2_list.append(conv2)
            for latent in sampled_third:
                conv1 = torch.sigmoid(overfit_models[digit].decoder.conv1(latent.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)))
                conv1_list.append(conv1)
            for i in range(num_datas):
                digitReps[digit].append(fc_list[i], conv1_list[i], conv2_list[i])
        digitReps[digit].convert_to_tensor()
    
    return baseRep, digitReps
        

def compare_cka(rep1, rep2):
    cka_score = CKA.kernel_CKA(rep1, rep2).item()
    return cka_score

def compute_cka_all_layers(baseRep, digitReps,expname="cka_results"):
    ################################################
    # generate CKA btw baseline and overfit models #
    ################################################
    cka_scores = {}
    for digit in range(10):
        print(f"CKA scores for digit {digit}:")
        cka_scores[digit] = [[] for _ in range(3)]
        layers = [baseRep.fc, baseRep.conv2, baseRep.conv1]
        for i, layer1 in enumerate(layers):
            for j, layer2 in enumerate([digitReps[digit].fc, digitReps[digit].conv2, digitReps[digit].conv1]):
                if layer1.shape[0] == layer2.shape[0]:
                    rep1 = layer1.view(layer1.shape[0], -1)
                    rep2 = layer2.view(layer2.shape[0], -1)
                    cka_score = compare_cka(rep1, rep2)
                    cka_scores[digit][i].append(cka_score)
                else:
                    cka_scores[digit][i].append(None)
        print(cka_scores[digit])

    # draw CKA scores, make 10 images with 3x3 subplots that show cka scores in color map
    for digit in range(10):
        plt.figure(figsize=(8, 8))
        plt.imshow(np.array(cka_scores[digit])[::-1], cmap='viridis', vmin=0, vmax=1)
        plt.colorbar(label='CKA Score')
        plt.xticks(ticks=np.arange(3), labels=['FC_base', 'Conv2_base', 'Conv1_base'])
        plt.yticks(ticks=np.arange(3), labels=['Conv1_digit', 'Conv2_digit', 'FC_digit'])
        plt.title(f'CKA Scores between Baseline and Digit {digit} Overfit Model')
        for i in range(3):
            for j in range(3):
                score = cka_scores[digit][2-i][j]
                if score is not None:
                    plt.text(j, i, f"{score:.2f}", ha='center', va='center', color='white' if score < 0.5 else 'black')
        plt.tight_layout()
        plt.savefig(f'cka_digit_{digit}_{expname}.png')
        plt.close()

    # make img with all digits' cka scores in one image
    plt.figure(figsize=(12, 12))
    for digit in range(10):
        plt.subplot(4, 3, digit+1)
        plt.imshow(np.array(cka_scores[digit])[::-1], cmap='viridis', vmin=0, vmax=1)
        plt.title(f'Digit {digit}')
        plt.xticks(ticks=np.arange(3), labels=['FC_base', 'Conv2_base', 'Conv1_base'], rotation=45)
        plt.yticks(ticks=np.arange(3), labels=['Conv1_digit', 'Conv2_digit', 'FC_digit'])
        for i in range(3):
            for j in range(3):
                score = cka_scores[digit][2-i][j]
                if score is not None:
                    plt.text(j, i, f"{score:.2f}", ha='center', va='center', color='white' if score < 0.5 else 'black')
    plt.tight_layout()
    plt.savefig(f'cka_all_digits_{expname}.png')
    plt.close()


if __name__ == "__main__":
    baseline_vae, overfit_models = model_init()
    baseRep, digitReps = generate_representations(baseline_vae, overfit_models)

    independent_baseRep, independent_digitReps = generate_representations_independent(baseline_vae, overfit_models)
    
    print("CKA scores in original way:")
    compute_cka_all_layers(baseRep, digitReps)


    print("CKA scores in independent way:")
    compute_cka_all_layers(independent_baseRep, independent_digitReps, expname="independent_cka_results")
