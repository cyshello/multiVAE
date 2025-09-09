import os
import sys
import numpy as py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import copy
from config import *
from evaluation import *
from model import *


class TaskVector:
    def __init__(self, pretrained_checkpoint, finetuned_checkpoint):
        """Task vector 생성"""
        self.vector = {}

        # 체크포인트 로드
        pretrained_state = torch.load(pretrained_checkpoint, map_location='cpu')
        finetuned_state = torch.load(finetuned_checkpoint, map_location='cpu')

        # Task vector 계산: τ = θ_ft - θ_0
        for key in finetuned_state:
            if key in pretrained_state:
                self.vector[key] = finetuned_state[key] - pretrained_state[key]

    def apply_to(self, pretrained_checkpoint, scaling_coef=1.0, return_model=False, model_class=None):
        """Task vector를 모델에 적용"""
        if isinstance(pretrained_checkpoint, str):
            pretrained_state = torch.load(pretrained_checkpoint, map_location='cpu')
        else:
            # pretrained_checkpoint이 모델 객체인 경우
            pretrained_checkpoint.cpu()
            pretrained_state = pretrained_checkpoint.state_dict()

        new_state = copy.deepcopy(pretrained_state)

        # θ_new = θ_0 + λ * τ
        for key in self.vector:
            if key in new_state:
                new_state[key] += scaling_coef * self.vector[key]

        if return_model and model_class:
            # 실제 모델 객체 반환
            model = model_class()
            model.load_state_dict(new_state)
            return model

        return new_state

    def __add__(self, other, coeff=1.0):
      """Task vector 덧셈 - 모든 키 처리"""
      result = TaskVector.__new__(TaskVector)
      result.vector = {}

      # 모든 키를 처리 (self와 other의 모든 키)
      all_keys = set(self.vector.keys()) | set(other.vector.keys())

      for key in all_keys:
        if key in self.vector and key in other.vector:
            result.vector[key] = self.vector[key] + coeff * other.vector[key]
        elif key in self.vector:
            result.vector[key] = self.vector[key].clone()
        elif key in other.vector:
            result.vector[key] = coeff * other.vector[key].clone()

      return result


    def __neg__(self):
        """Task vector 부정"""
        result = TaskVector.__new__(TaskVector)
        result.vector = {key: -value for key, value in self.vector.items()}
        return result
    
    def cosine_similarity(self, other):
        """Task vector 간의 코사인 유사도 계산"""
        if not isinstance(other, TaskVector):
            raise TypeError("other must be an instance of TaskVector")

        cos_sim = 0.0
        for key in self.vector:
            v1 = self.vector[key].flatten()
            v2 = other.vector[key].flatten()
            # numpy나 torch API여도 결과는 스칼라 한 개
            cos = F.cosine_similarity(v1, v2, dim=0)
            cos_sim += cos
        return cos_sim / len(self.vector) if self.vector else 0.0

####################
# task vectors exp #
####################

def task_vectors_exp(baseline_path, scaling_coef, test_set,exp_name=""):
    '''
    baseline_path에 있는 모델과 test_set에 있는 숫자의 task vector들을 scaling_coef으로 곱해서 더한 후에 evaluating 하고 리턴
    '''
    print(f"Conducting Experiment for {len(test_set)} task vectors...")
    models = {}

    for i, digits in enumerate(test_set):
        new_vector = TaskVector.__new__(TaskVector)
        new_vector.vector = {}

        print(f"Experiment #{i} with digits:")
        digit1, digit2 = digits

        new_model = task_vectors_digits[digit1].apply_to(
            baseline_path,
            scaling_coef=scaling_coef,
            return_model=True,
            model_class=VariationalAutoencoder
        )

        new_model = task_vectors_digits[digit2].apply_to(
            new_model,
            scaling_coef=scaling_coef,
            return_model=True,
            model_class=VariationalAutoencoder
        )

        # for digit in digits:
        #     #print(digit,end=" ")
        #     new_vector += task_vectors_digits[digit]

        # new_model = new_vector.apply_to(
        #     baseline_path,
        #     scaling_coef=scaling_coef,
        #     return_model = True,
        #     model_class = VariationalAutoencoder
        #     )

        new_model = new_model.to(device)
        new_model.eval()

        evalaute_vae_recon(new_model) #only loss
        #evaluate_vae_recon_debug_five_samples(new_model) #with img samples reconstruction
        visualize_generation(new_model, f"{scaling_coef}_exp_{i}_{exp_name}")
        models[digits] = new_model
    
    return models
