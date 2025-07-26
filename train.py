import os
import sys
import numpy as py
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *
from model import *
import matplotlib.pyplot as plt
import torchvision.utils
from evaluation import *

def visualize_training_results(model, train_loss_avg, device, latent_dims, modelname, show_avg=True, img_num=20):
    """
    VAE 훈련 결과를 시각화하는 함수

    Args:
        model: 훈련된 VAE 모델
        train_loss_avg: 에포크별 평균 손실 리스트
        device: 연산 장치 (cuda/cpu)
        latent_dims: 잠재 공간 차원
        modelname: 모델 이름
    """
    if show_avg:
        fig = plt.figure()
        plt.plot(train_loss_avg)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.show()

    model.eval()
    with torch.no_grad():
        latent = torch.randn(img_num, latent_dims).to(device)
        image_recon = model.decoder(latent)
        image_recon = image_recon.cpu()

        # 10개씩 나누어 출력
        rows = (img_num + 9) // 10  # 10개씩 나누기 위한 행 개수
        fig, axes = plt.subplots(rows, 10, figsize=(15, 2 * rows))

        # 단일 행인 경우 axes를 2차원으로 만들기
        if rows == 1:
            axes = axes.reshape(1, -1)

        for i in range(img_num):
            img = image_recon[i].squeeze()
            row = i // 10
            col = i % 10
            axes[row, col].imshow(img, cmap='gray')
            axes[row, col].axis('off')
            axes[row, col].set_title(i)

        # 남은 subplot이 있을 경우 숨기기
        for j in range(img_num, rows * 10):
            row = j // 10
            col = j % 10
            axes[row, col].axis('off')

        plt.suptitle(f'Generated Images for {modelname}')
        plt.tight_layout()
        plt.show()


def train_and_generate_model(model, dataset, modelname, device, modelpath,
                           baseline_path=None, overfit=False, num_epochs=100,
                           learning_rate=1e-3, latent_dims=20):
    """
    VAE 모델을 훈련하고 생성 결과를 시각화하는 함수

    Args:
        model: VAE 모델 인스턴스
        dataset: 훈련용 DataLoader
        modelname: 모델 저장 파일명
        device: 연산 장치 (cuda/cpu)
        modelpath: 모델 저장 경로
        baseline_path: overfitting 시 사용할 baseline 모델 경로
        overfit: baseline에서 시작하여 overfitting 수행 여부
        num_epochs: 훈련 에포크 수
        learning_rate: 학습률
        latent_dims: 잠재 공간 차원

    Returns:
        tuple: (훈련된 모델, 모델 저장 경로)
    """

    model = model.to(device)
    if not os.path.exists(modelpath + "epoch" + str(num_epochs)):
        os.makedirs(modelpath + "epoch" + str(num_epochs))
    trainmodel_path = modelpath + "epoch" + str(num_epochs) + "/" + modelname

    if os.path.exists(trainmodel_path):
        model.load_state_dict(torch.load(trainmodel_path, map_location=device))
        model.eval()
        print(f"Pretrained model loaded from: {trainmodel_path}")
        return model, trainmodel_path

    if overfit and baseline_path and os.path.exists(baseline_path):
        model.load_state_dict(torch.load(baseline_path, map_location=device))
        print(f"Starting from baseline: {baseline_path}")

    optimizer = torch.optim.Adam(params=model.parameters(),
                               lr=learning_rate, weight_decay=1e-5)

    model.train()
    train_loss_avg = []

    print('Training ...')
    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0

        for image_batch, _ in dataset:
            image_batch = image_batch.to(device)

            image_batch_recon, latent_mu, latent_logvar = model(image_batch)

            loss = vae_loss(image_batch_recon, image_batch, latent_mu, latent_logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        train_loss_avg.append(avg_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}] average reconstruction error: {avg_loss:.6f}')

    torch.save(model.state_dict(), trainmodel_path)
    print(f'Model saved to: {trainmodel_path}')

    visualize_training_results(model, train_loss_avg, device, latent_dims, modelname)

    return model, trainmodel_path