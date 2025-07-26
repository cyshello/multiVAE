import os
import sys
import numpy as py
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import *
from model import *
from train import *
from evaluation import *
from taskvector import *
from config import *
from torch.utils.data import ConcatDataset, DataLoader


device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")

load_datasets(datapath)

train_dataloader_digits, train_dataloader_baseline = split_mnist(overfit_num)

###########################
# Generate baseline model #
###########################

num_epochs = 30

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

####################################
# Make task vectors of each digits #
####################################

for digit in range(10):
    baseline_path = baseline_path
    digit_model_path = modelpath + f"epoch{num_epochs}/digit_{digit}"

    task_vectors_digits[digit] = TaskVector(baseline_path, digit_model_path)

###############
# Experiments 
# mannually experiment all kind of coefficient with two digits added 
###############


overfit_model_with_two = {}
added_model_with_two = {}
test_set_two = [(1,7),(6,7),(2,5),(4,9),(7,9)]
coefficients = [(0.1,0.9),(0.2,0.8),(0.3,0.7),(0.4,0.6),(0.5,0.5),(0.6,0.4),(0.7,0.3),(0.8,0.2),(0.9,0.1)]

# Overfit model with two digits
for digits in test_set_two:
    digit1, digit2 = digits
    overfit_model = VariationalAutoencoder()
    dataset1 = train_dataloader_digits[digit1].dataset
    dataset2 = train_dataloader_digits[digit2].dataset
    combined_dataset = ConcatDataset([dataset1, dataset2])
    tmp_dataloader = DataLoader(combined_dataset, batch_size=32, shuffle=True)
    overfit_model, _ = train_and_generate_model(
        model=overfit_model,
        dataset=tmp_dataloader,
        modelname=f"digit_{digit1}_{digit2}",
        device=device,
        modelpath=modelpath,
        baseline_path=baseline_path,
        overfit=True,
        num_epochs=num_epochs
    )
    overfit_model_with_two[digits] = overfit_model

for digits in test_set_two:
    digit1, digit2 = digits

    added_model_with_two[digits] = {}
    for coeff in coefficients:
        new_vector = TaskVector.__new__(TaskVector)
        new_vector.vector = {}
        coeff1, coeff2 = coeff
        new_vector.__add__(task_vectors_digits[digit1], coeff1)
        new_vector.__add__(task_vectors_digits[digit2], coeff2)
        
        added_model_with_two[digits][coeff] = new_vector.apply_to(
            baseline_path,
            return_model=True,
            model_class=VariationalAutoencoder
        )

# Evaluate the models

fid_results = {}
lipips_results = {}

for digits in test_set_two:
    digit1, digit2 = digits
    GTmodel = overfit_model_with_two[digits].to(device)
    print(f"Evaluating models for digits {digit1} and {digit2}...")

    # Evaluate added models with different coefficients
    for coeff, model in added_model_with_two[digits].items():
        model = model.to(device)
        fidscore = FID_score(
            GTmodel=GTmodel,
            expmodel=model,
            testnum = 1000
        )
        print(f"Coefficient {coeff}: FID score = {fidscore:.4f}")

        tmpdataset = train_dataloader_digits[digit1].dataset + train_dataloader_digits[digit2].dataset
        test = DataLoader(tmpdataset, batch_size=32, shuffle=True)
        lpipsscore = LPIPS_score(
            test_dataloader = test,
            GTmodel=GTmodel,
            expmodel=model,
        )

        print(f"Coefficient {coeff}: LPIPS score = {lpipsscore:.4f}")
        fid_results[digits] = {}
        fid_results[digits][coeff] = fidscore
        lipips_results[digits] = {}
        lipips_results[digits][coeff] = lpipsscore

# Visualize the results

for digits in test_set_two:
    bestFID = 0x0fffffff
    bestLPIPS = 0x0fffffff
    for coeff in coefficients:
        coeff1, coeff2 = coeff
        if bestFID < fid_results[digits][coeff]:
            bestFID = fid_results[digits][coeff]
            bestFIDcoeff = coeff
        
        if bestLPIPS < lipips_results[digits][coeff]:
            bestLPIPS = lipips_results[digits][coeff]
            bestLPIPScoeff = coeff

        print(f"Digits {digits}, Coefficients {coeff1}, {coeff2}:")
        print(f"  FID Score: {fid_results[digits][coeff]:.4f}") 
        print(f"  LPIPS Score: {lipips_results[digits][coeff]:.4f}")
    
    print(f"Best coeff with best FID score, LPIPS score : {bestFIDcoeff}, {bestLPIPScoeff}")

    visualize_generation(
        model = added_model_with_two[digits][bestFIDcoeff],
        exp_name = f"bestFIDmodel_visualization_{digits[0]}_{digits[1]}"
    )
    print(f"visualized image saved at bestFIDmodel_visualization_{digits[0]}_{digits[1]}")
    
    visualize_generation(
        model = added_model_with_two[digits][bestLPIPScoeff],
        exp_name = f"bestLPIPSmodel_visualization_{digits[0]}_{digits[1]}"
    )
    print(f"visualized image saved at bestLPIPSmodel_visualization_{digits[0]}_{digits[1]}")

    