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
from config import *
from taskvector import TaskVector, ties_merging
from torch.utils.data import ConcatDataset, DataLoader
import json


load_datasets(datapath)

train_dataloader_digits, train_dataloader_baseline = split_mnist(overfit_num)
num_epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# load or train baseline VAE
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

# load or train digit-specific VAEs

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

tv_digits = {}
for digit in range(10):
    tv_digits[digit] = TaskVector(
        pretrained_checkpoint=baseline_path,
        finetuned_checkpoint=os.path.join(modelpath, f"epoch{num_epochs}/digit_{digit}")
    )

############################################
# Helpers: flatten/restore for TIES merging #
############################################

def get_param_keys(sd: dict[str, torch.Tensor]):
    """Return ordered keys for floating tensors in a state_dict."""
    keys = []
    for k, v in sd.items():
        if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
            keys.append(k)
    return keys


def state_dict_to_flat_vector(sd: dict[str, torch.Tensor], keys: list[str]) -> torch.Tensor:
    """Flatten selected tensors from state_dict to a single 1D vector (CPU)."""
    vecs = [sd[k].reshape(-1).detach().cpu() for k in keys]
    return torch.cat(vecs, dim=0)


def vector_to_state_dict(vec: torch.Tensor, template_sd: dict[str, torch.Tensor], keys: list[str]) -> dict:
    """Convert flat vector back into a state_dict with shapes from template_sd for the given keys."""
    vec = vec.detach().cpu()
    out = {}
    idx = 0
    for k, v in template_sd.items():
        if k in keys:
            num = v.numel()
            out[k] = vec[idx: idx + num].view_as(v).type_as(v)
            idx += num
        else:
            # keep original (non-floating tensors etc.)
            out[k] = v.clone() if isinstance(v, torch.Tensor) else v
    assert idx == vec.numel(), "Vector size mismatch when reconstructing state_dict"
    return out


def taskvector_to_flat(tv: TaskVector, template_sd: dict[str, torch.Tensor], keys: list[str]) -> torch.Tensor:
    """Flatten a TaskVector using template shapes and full key coverage (zero when key missing)."""
    chunks = []
    for k in keys:
        base_t = template_sd[k]
        delta = tv.vector.get(k)
        if delta is None:
            chunks.append(torch.zeros_like(base_t, device='cpu').reshape(-1))
        else:
            chunks.append(delta.detach().cpu().reshape(-1))
    return torch.cat(chunks, dim=0)


######################################
# Build flat TVs and apply TIES merge #
######################################

# Load baseline checkpoint as template for flatten/restore
ptm_check = torch.load(baseline_path, map_location='cpu')
param_keys = get_param_keys(ptm_check)
flat_ptm = state_dict_to_flat_vector(ptm_check, param_keys)

# Precompute flat task vectors per digit
tv_flat_by_digit: dict[int, torch.Tensor] = {}
for d in range(10):
    tv_flat_by_digit[d] = taskvector_to_flat(tv_digits[d], ptm_check, param_keys)

# TIES settings
K = 20            # keep top 20% magnitudes
merge_func = "dis-mean"
lamda = 1.0

#########################################
# Overfit paired GT models and evaluate #
#########################################

overfit_model_with_two = {}
ties_model_with_two = {}
test_set_two = [(1,7),(6,7),(2,5),(4,9),(7,9),(1,5)]

# Train/Load GT models over two digits
for digit1, digit2 in test_set_two:
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
    overfit_model_with_two[(digit1, digit2)] = overfit_model

    # Build a TIES-merged model for the same pair
    pair_tv = torch.stack([tv_flat_by_digit[digit1], tv_flat_by_digit[digit2]], dim=0)
    merged_tv = ties_merging(pair_tv, reset_thresh=K, merge_func=merge_func)
    merged_check = flat_ptm + lamda * merged_tv
    merged_sd = vector_to_state_dict(merged_check, ptm_check, param_keys)

    ties_model = VariationalAutoencoder()
    ties_model.load_state_dict(merged_sd, strict=True)
    ties_model_with_two[(digit1, digit2)] = ties_model


#########################
# Evaluate TIES results #
#########################

fid_results_ties = {}
lpips_results_ties = {}

for digits, model in ties_model_with_two.items():
    digit1, digit2 = digits
    print(f"Evaluating TIES merged model for digits {digit1} & {digit2}...")
    GTmodel = overfit_model_with_two[digits].to(device)
    expmodel = model.to(device)

    fidscore = FID_score(
        GTmodel=GTmodel,
        expmodel=expmodel,
        testnum=1000,
    )

    tmpdataset = train_dataloader_digits[digit1].dataset + train_dataloader_digits[digit2].dataset
    test_loader = DataLoader(tmpdataset, batch_size=32, shuffle=True)
    lpipsscore = LPIPS_score(
        test_dataloader=test_loader,
        GTmodel=GTmodel,
        expmodel=expmodel,
    )

    fid_results_ties[digits] = fidscore
    lpips_results_ties[digits] = lpipsscore

    print(f"Digits {digit1},{digit2} | TIES FID: {fidscore:.4f} | TIES LPIPS: {lpipsscore:.4f}")

# Optional: visualize generations from TIES models
for digits, model in ties_model_with_two.items():
    visualize_generation(
        model=model,
        exp_name=f"./epoch{num_epochs}/ties_visualization/{digits[0]}_{digits[1]}",
        num_img=20,
    )
    print(f"TIES visualization saved for digits {digits[0]}_{digits[1]}")

print("TIES FID Results:")
print(fid_results_ties)
print("TIES LPIPS Results:")
print(lpips_results_ties)

