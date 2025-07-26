import os
import sys
import numpy as py
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *
from model import *
import matplotlib.pyplot as plt
import random
from torchmetrics.image.fid import FrechetInceptionDistance
import lpips
from PIL import Image
from torchvision import transforms


def evalaute_vae_recon(model): # calculate vae_loss for each digit test datasets
  model.eval()

  avg_recon_loss = [0 for _ in range(10)]
  for digit in range(10):
    for img_batch, _ in test_digit_dataloaders[digit]:
      img_batch = img_batch.to(device)

      img_batch_recon, latent_mu, latent_logvar = model(img_batch)

      loss = vae_loss(img_batch_recon,img_batch, latent_mu, latent_logvar)

      avg_recon_loss[digit] += loss.item()

    avg_recon_loss[digit] /= len(test_digit_dataloaders[digit])
    print(f"average vae_loss for number {digit} : ",avg_recon_loss[digit])

  return avg_recon_loss


def evaluate_vae_recon_all(model): # calculate vae_loss for whole test datasets, return average.
  model.eval()

  avg_loss = 0

  for img_batch, _ in test_dataloader:
      img_batch = img_batch.to(device)

      img_batch_recon, latent_mu, latent_logvar = model(img_batch)

      loss = vae_loss(img_batch_recon,img_batch, latent_mu, latent_logvar)

      avg_loss += loss.item()

  avg_loss /= len(test_dataloader)
  print(f"average vae_loss for whole test set : {avg_loss}")

  return avg_loss

def visualize_reconstruction(model, exp_name, num_img=5):
    model.eval()

    save_dir = os.path.join(exppath, exp_name)
    os.makedirs(save_dir, exist_ok=True)

    all_org_imgs = []
    all_recon_imgs = []

    for digit in range(10):
        dataset = test_digit_datasets[digit]
        random_indices = random.sample(range(len(dataset)), num_img)
        org_imgs = [dataset[i][0] for i in random_indices]
        recon_imgs = []

        for img in org_imgs:
            img = img.to(device).unsqueeze(0)
            with torch.no_grad():
                recon, _, _ = model(img)
            recon_imgs.append(recon.squeeze(0).cpu())

        all_org_imgs.append(org_imgs)
        all_recon_imgs.append(recon_imgs)

    # 20행(각 digit별 원본/재구성), num_img열
    fig, axes = plt.subplots(20, num_img, figsize=(num_img * 2, 20 * 1.5))

    for digit in range(10):
        # 원본 이미지 행
        for i in range(num_img):
            axes[digit * 2, i].imshow(all_org_imgs[digit][i].cpu().squeeze(), cmap='gray')
            axes[digit * 2, i].axis('off')
            if i == 0:
                axes[digit * 2, i].set_ylabel(f'Digit {digit}\nOriginal', fontsize=10)
        # 재구성 이미지 행
        for i in range(num_img):
            axes[digit * 2 + 1, i].imshow(all_recon_imgs[digit][i].squeeze(), cmap='gray')
            axes[digit * 2 + 1, i].axis('off')
            if i == 0:
                axes[digit * 2 + 1, i].set_ylabel('Reconstructed', fontsize=10)

    plt.suptitle('Original & Reconstructed Images for Each Digit', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    save_path = os.path.join(save_dir, 'all_digits_20rows_reconstruction.png')
    plt.savefig(save_path)
    plt.close(fig)

  
def visualize_generation(model, exp_name, num_img=10, latent_dims=20):
    """
    VAE 모델의 latent space에서 무작위로 이미지를 생성하고 저장하는 함수

    Args:
        model: 학습된 VAE 모델
        exp_name: 실험 이름(이미지 저장 폴더명)
        num_img: 생성할 이미지 개수 (기본값 10)
        latent_dims: latent space 차원 수
        device: 연산 장치 ('cpu' 또는 'cuda')
    """
    model.eval()
    save_dir = os.path.join(exppath, exp_name)
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        latent = torch.randn(num_img, latent_dims).to(device)
        generated_imgs = model.decoder(latent).cpu()

    # 10개씩 한 행에 출력
    rows = (num_img + 9) // 10
    fig, axes = plt.subplots(rows, 10, figsize=(15, 2 * rows))

    # 단일 행일 경우 2차원으로 변환
    if rows == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_img):
        img = generated_imgs[i].squeeze()
        row = i // 10
        col = i % 10
        axes[row, col].imshow(img, cmap='gray')
        axes[row, col].axis('off')
        axes[row, col].set_title(f'{i+1}')

    # 남는 subplot 숨기기
    for j in range(num_img, rows * 10):
        row = j // 10
        col = j % 10
        axes[row, col].axis('off')

    plt.suptitle(f'Randomly Generated Images from Latent Space')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    save_path = os.path.join(save_dir, 'generated_images.png')
    plt.savefig(save_path)
    plt.close(fig)

  
def FID_score(GTmodel, expmodel, testnum = 1000):
    GTmodel.eval()
    expmodel.eval()
    GT_images = []
    exp_images = []

    for i in range(testnum):
        with torch.no_grad():
            latent = torch.randn(num_img, latent_dims).to(device)
            GT_img = GTmodel.decoder(latent).cpu()
            exp_img = expmodel.decoder(latent).cpu()
            GT_images.append(GT_img)
            exp_images.append(exp_img)
    
    GT_images = torch.cat(GT_images, dim=0)
    exp_images = torch.cat(exp_images, dim=0)

    fid = FrechetInceptionDistance(normalize=True).to(device)

    fid.update(GT_images, real=True)
    fid.update(exp_images, real=False)

    return fid.compute().item() # return FID score as float

def LPIPS_score(GTmodel, expmodel):
    global test_dataloader

    for img_batch, _ in test_dataloader:
        img_batch = img_batch.to(device)

        GT_img_recon, _, _ = GTmodel(img_batch)
        exp_img_recon, _, _ = expmodel(img_batch)

        GT_img_recon = GT_img_recon.cpu()
        exp_img_recon = exp_img_recon.cpu()

        # LPIPS 계산
        loss_fn = lpips.LPIPS(net='alex').to(device)
        score = loss_fn(GT_img_recon, exp_img_recon)

    return score.mean().item()  # 평균 LPIPS 점수 반환 (float)