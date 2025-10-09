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
from taskvector_ import *
from config import *
from torch.utils.data import ConcatDataset, DataLoader
import json

num_epochs = 50
fid_file = f'./experiments/fid_results_{num_epochs}.json'
lpips_file = f'./experiments/lpips_result_{num_epochs}.json'

def deep_tuple_keys_to_str(d):
    # 재귀적으로 tuple 키를 문자열로 변환
    return {str(k): deep_tuple_keys_to_str(v) if isinstance(v, dict) else v for k, v in d.items()}

def str_keys_to_tuple(d):
    # 재귀적으로 문자열 키를 tuple로 변환(불가능하면 문자열 유지)
    new_d = {}
    for k, v in d.items():
        try:
            k_eval = eval(k)
            # eval 결과가 tuple이면 tuple 사용, 아니면 원래 문자열 사용
            new_k = k_eval if isinstance(k_eval, tuple) else k
        except Exception:
            new_k = k
        new_d[new_k] = str_keys_to_tuple(v) if isinstance(v, dict) else v
    return new_d

def save_results(fid_results, lpips_results, fid_file=fid_file, lpips_file=lpips_file):
    with open(fid_file, 'w') as f:
        json.dump(deep_tuple_keys_to_str(fid_results), f, indent=4)
    with open(lpips_file, 'w') as f:
        json.dump(deep_tuple_keys_to_str(lpips_results), f, indent=4)

def load_results(fid_file=fid_file, lpips_file=lpips_file):
    with open(fid_file, 'r') as f:
        fid_results = json.load(f)
    with open(lpips_file, 'r') as f:
        lpips_results = json.load(f)
    # 불러온 후 문자열 키 → tuple 키 변환
    return str_keys_to_tuple(fid_results), str_keys_to_tuple(lpips_results)

def dataset(datapath):
    load_datasets(datapath)

    train_dataloader_digits, train_dataloader_baseline = split_mnist(overfit_num)
    return train_dataloader_digits, train_dataloader_baseline

def set_models(num_epochs,train_dataloader_digits, train_dataloader_baseline):
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

    ####################################
    # Make task vectors of each digits #
    ####################################

    for digit in range(10):
        baseline_path = baseline_path
        digit_model_path = modelpath + f"epoch{num_epochs}/digit_{digit}"

        task_vectors_digits[digit] = TaskVector(baseline_path, digit_model_path)

    return baseline_vae, overfit_models, task_vectors_digits, baseline_path

###############
# Experiments 
# mannually experiment all kind of coefficient with two digits added 
###############

def generate_taskvector_models(baseline_vae, task_vectors_digits, baseline_path):
    overfit_model_with_two = {}
    added_model_with_two = {}
    test_set_two = [(0,1),(2,1),(3,1),(4,1),(5,1),(6,1),(7,1),(8,1),(9,1),(4,9)]
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
            new_model = VariationalAutoencoder()   
            coeff1, coeff2 = coeff
            
            new_model = task_vectors_digits[digit1].apply_to(
                baseline_vae,
                scaling_coef=coeff1,
                return_model=True,
                model_class=VariationalAutoencoder
            )
            
            added_model_with_two[digits][coeff] = task_vectors_digits[digit2].apply_to(
                new_model,
                scaling_coef=coeff2,
                return_model=True,
                model_class=VariationalAutoencoder
            )
    
    return overfit_model_with_two, added_model_with_two, test_set_two

def make_negation(baseline_vae, scaling_coef=1.0):
    negated_models = {}
    for digit in range(10):
        negated_model = task_vectors_digits[digit].apply_to(
            baseline_vae,
            scaling_coef=-scaling_coef,
            return_model=True,
            model_class=VariationalAutoencoder
        )
        negated_models[digit] = negated_model
    return negated_models

#pathmodels = task_vectors_exp(baseline_path, scaling_coef=0.5, test_set=test_set_two, exp_name=f"path")
def eval_experiment(num_epochs, baseline_vae, task_vectors_digits, baseline_path, train_dataloader_digits):
    fid_results = {}
    lipips_results = {}

    fid_results, lipips_results = load_results()
    for digits in test_set_two:
        if digits in fid_results and digits in lipips_results:
            print(f"Results for digits {digits} already exist. Skipping evaluation.")
            continue
        digit1, digit2 = digits
        GTmodel = overfit_model_with_two[digits].to(device)
        print(f"Evaluating models for digits {digit1} and {digit2}...")
        fid_results[digits] = {}
        lipips_results[digits] = {}

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
            fid_results[digits][coeff] = fidscore
            lipips_results[digits][coeff] = lpipsscore
        
        print("FID Results:")
        print(fid_results)
        print("LPIPS Results:")
        print(lipips_results)
        save_results(fid_results, lipips_results)
    return fid_results, lipips_results

def visualize(num_epochs, fid_results, lipips_results, baseline_vae, overfit_model_with_two, added_model_with_two, test_set_two):
    # Visualize the results
    for digits in test_set_two:
        bestFID = 0x0fffffff
        bestLPIPS = 0x0fffffff
        leastFID = 0x00000000
        leastLPIPS = 0x00000000
        for coeff, _ in added_model_with_two[digits].items():
            GTmodel = overfit_model_with_two[digits]
            GTmodel.eval()
            coeff1, coeff2 = coeff
            if bestFID > fid_results[digits][coeff]:
                bestFID = fid_results[digits][coeff]
                bestFIDcoeff = coeff
            
            if bestLPIPS > lipips_results[digits][coeff]:
                bestLPIPS = lipips_results[digits][coeff]
                bestLPIPScoeff = coeff
            
            if leastFID < fid_results[digits][coeff]:
                leastFID = fid_results[digits][coeff]
                leastFIDcoeff = coeff
            
            if leastLPIPS < lipips_results[digits][coeff]:
                leastLPIPS = lipips_results[digits][coeff]
                leastLPIPScoeff = coeff

            print(f"Digits {digits}, Coefficients {coeff1}, {coeff2}:")
            print(f"  FID Score: {fid_results[digits][coeff]:.4f}") 
            print(f"  LPIPS Score: {lipips_results[digits][coeff]:.4f}")
        
        print(f"Best coeff with best FID score, LPIPS score : {bestFIDcoeff}, {bestLPIPScoeff}")
        print(f"Least coeff with least FID score, LPIPS score : {leastFIDcoeff}, {leastLPIPScoeff}")

        visualize_generation(
            model = added_model_with_two[digits][bestFIDcoeff],
            exp_name = f"./epoch{num_epochs}/bestFIDmodel_visualization/{digits[0]}_{digits[1]}"
        )
        print(f"visualized image saved at bestFIDmodel_visualization_{digits[0]}_{digits[1]}")
        
        visualize_generation(
            model = added_model_with_two[digits][bestLPIPScoeff],
            exp_name = f"./epoch{num_epochs}/bestLPIPSmodel_visualization/{digits[0]}_{digits[1]}"
        )
        print(f"visualized image saved at bestLPIPSmodel_visualization_{digits[0]}_{digits[1]}")

        visualize_generation(
            model = added_model_with_two[digits][leastFIDcoeff],
            exp_name = f"./epoch{num_epochs}/leastFIDmodel_visualization/{digits[0]}_{digits[1]}"
        )
        print(f"visualized image saved at leastFIDmodel_visualization_{digits[0]}_{digits[1]}")

        visualize_generation(
            model = added_model_with_two[digits][leastLPIPScoeff],
            exp_name = f"./epoch{num_epochs}/leastLPIPSmodel_visualization/{digits[0]}_{digits[1]}"
        )
        print(f"visualized image saved at leastLPIPSmodel_visualization_{digits[0]}_{digits[1]}")

        visualize_generation(
            model = added_model_with_two[digits][(0.5,0.5)],
            exp_name = f"./epoch{num_epochs}/5model_visualization/{digits[0]}_{digits[1]}"
        )
        print(f"visualized image saved at 5model_visualization_{digits[0]}_{digits[1]}")

        visualize_generation(
            model = GTmodel,
            exp_name = f"./epoch{num_epochs}/GTmodel_visualization/{digits[0]}_{digits[1]}"
        )
        print(f"visualized image saved at GTmodel_visualization_{digits[0]}_{digits[1]}")


    visualize_generation(
        model = baseline_vae,
        exp_name = f"./epoch{num_epochs}/baseline_visualization",
        num_img=20
    )
    print("visualized image saved at baseline_visualization")

    ## visualize cosine similarity of task vectors

    cosine_sim_visualize(task_vectors_digits)
    print("visualized cosine similarity of task vectors")


if __name__ == "__main__":
    # Load dataset
    train_dataloader_digits, train_dataloader_baseline = dataset(datapath)

    # Set models
    baseline_vae, overfit_models, task_vectors_digits, baseline_path = set_models(num_epochs,train_dataloader_digits, train_dataloader_baseline)

    # Generate taskvector models
    overfit_model_with_two, added_model_with_two, test_set_two = generate_taskvector_models(baseline_vae, task_vectors_digits, baseline_path)

    # Make negation models
    negated_models = make_negation(baseline_vae, scaling_coef=1.0)

    print("Evalauate baseline VAE model")
    evalaute_vae_recon(baseline_vae)
    # for digits in test_set_two:
    #     if digits != (4,9): continue
    #     print(f"Evaluate overfitted models for digits: {digits}")
    #     for coeff, model in added_model_with_two[digits].items():
    #         print(f"Coefficient: {coeff}")
    #         evalaute_vae_recon(model)
    #     print("Evalauate overfitted models for each digit")
    #     evalaute_vae_recon(overfit_model_with_two[digits])
    
    print("Evaluate negated models for each digit")
    for digit, model in negated_models.items():
        print(f"Digit: {digit}")
        evalaute_vae_recon(model)
        visualize_generation(
            model = model,
            exp_name = f"./epoch{num_epochs}/negation_visualization/digit_{digit}",
            num_img=20
        )

    # Evaluate Experiment using FID, LPIPS
    #fid_results, lipips_results = eval_experiment(num_epochs, baseline_vae, task_vectors_digits, baseline_path, train_dataloader_digits)

    # Visualize results
    #visualize(num_epochs, fid_results, lipips_results, baseline_vae, overfit_model_with_two, added_model_with_two, test_set_two)
