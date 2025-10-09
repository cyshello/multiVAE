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
from torch.utils.data import ConcatDataset, DataLoader
import json


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

    def __add__(self, other, coeff=1.0):
      """Task vector 덧셈 - 모든 키 처리"""
      result = TaskVector.__new__(TaskVector)
      result.vector = {}

      # 모든 키를 처리 (self와 other의 모든 키)
      all_keys = set(self.vector.keys()) | set(other.vector.keys())

      for key in all_keys:
          val1 = self.vector.get(key)
          val2 = other.vector.get(key)

          if val1 is not None and val2 is not None:
              result.vector[key] = val1 + coeff * val2
          elif val1 is not None:
              result.vector[key] = val1
          elif val2 is not None:
              result.vector[key] = coeff * val2

      return result

    def __mul__(self, scalar):
        """Task vector 스칼라 곱셈"""
        result = TaskVector.__new__(TaskVector)
        result.vector = {key: val * scalar for key, val in self.vector.items()}
        return result

    def __neg__(self):
        """Task vector 부정"""
        result = TaskVector.__new__(TaskVector)
        result.vector = {key: -val for key, val in self.vector.items()}
        return result

    def apply(self, model_state):
        """모델 상태에 Task vector 적용"""
        new_state = model_state.copy()
        for key, val in self.vector.items():
            if key in new_state:
                new_state[key] += val
            else:
                new_state[key] = val
        return new_state    


# =============================
# TIES MERGING UTILS
# =============================
def topk_values_mask(M: torch.Tensor, K: float = 0.7, return_mask: bool = False):
    """Trim by keeping top-K magnitude entries per row.

    Inputs
    - M: Tensor of shape (n, d) or (d,). Each row is a flat task vector.
    - K: If > 1, treated as percentage (e.g., 20 => keep top 20%). If in [0,1], treated as fraction.
    - return_mask: When True, also return the boolean mask used.

    Returns
    - trimmed: M with non-topK entries zeroed, same shape as input
    - keep_ratio: per-row ratio of kept entries (Tensor of shape (n,) or scalar if input was 1D)
    - mask (optional): boolean mask of kept entries
    """
    if K > 1:
        K = K / 100.0

    # clamp to [0, 1]
    K = float(max(0.0, min(1.0, K)))

    original_was_vector = (M.dim() == 1)
    if original_was_vector:
        M = M.unsqueeze(0)

    n, d = M.shape

    if K == 0.0:
        mask = torch.zeros_like(M, dtype=torch.bool)
    elif K >= 1.0:
        mask = torch.ones_like(M, dtype=torch.bool)
    else:
        keep_count = max(1, int(round(d * K)))
        if keep_count >= d:
            mask = torch.ones_like(M, dtype=torch.bool)
        else:
            # threshold is (d - keep_count + 1)-th smallest by magnitude
            k_threshold = d - keep_count + 1
            kth_values, _ = M.abs().kthvalue(k_threshold, dim=1, keepdim=True)
            mask = M.abs() >= kth_values

    trimmed = M * mask
    keep_ratio = mask.float().mean(dim=1)

    if original_was_vector:
        trimmed = trimmed.squeeze(0)
        keep_ratio = keep_ratio.squeeze(0)
        mask = mask.squeeze(0)

    if return_mask:
        return trimmed, keep_ratio, mask
    return trimmed, keep_ratio


def resolve_zero_signs(sign_to_mult: torch.Tensor, method: str = "majority"):
    """Resolve 0 signs using majority or minority rule.

    sign_to_mult: 1D tensor of -1/0/+1 values.
    method: "majority" or "minority"
    """
    majority_sign = torch.sign(sign_to_mult.sum())
    # Fallback when perfectly balanced -> default to +1 for stability
    if majority_sign.item() == 0:
        majority_sign = torch.tensor(1, dtype=sign_to_mult.dtype, device=sign_to_mult.device)

    if method == "majority":
        sign_to_mult = sign_to_mult.clone()
        sign_to_mult[sign_to_mult == 0] = majority_sign
    elif method == "minority":
        sign_to_mult = sign_to_mult.clone()
        sign_to_mult[sign_to_mult == 0] = -1 * majority_sign
    else:
        raise ValueError(f"Unknown method for resolve_zero_signs: {method}")
    return sign_to_mult


def resolve_sign(Tensor: torch.Tensor):
    """Elect a global sign per parameter across rows.

    Tensor: shape (n, d). Computes sign(sign(sum over rows)). Zeros resolved by majority rule.
    Returns: 1D tensor of shape (d,), entries in {-1, +1}.
    """
    sign_to_mult = torch.sign(Tensor.sum(dim=0))
    sign_to_mult = resolve_zero_signs(sign_to_mult, "majority")
    return sign_to_mult


def disjoint_merge(Tensor: torch.Tensor, merge_func: str, sign_to_mult: torch.Tensor | None):
    """Aggregate only entries that agree with the elected sign (disjoint merge).

    Inputs
    - Tensor: shape (n, d)
    - merge_func: one of {"dis-mean", "dis-sum", "dis-max"} or just {"mean","sum","max"}
    - sign_to_mult: shape (d,) or None. If None, aggregate all non-zero entries.

    Returns
    - 1D tensor of length d
    """
    merge_key = merge_func.split("-")[-1]

    if sign_to_mult is not None:
        # Select entries matching the elected sign per column
        rows_to_keep = torch.where(sign_to_mult.unsqueeze(0) > 0, Tensor > 0, Tensor < 0)
        selected_entries = Tensor * rows_to_keep
    else:
        rows_to_keep = Tensor != 0
        selected_entries = Tensor * rows_to_keep

    if merge_key == "mean":
        non_zero_counts = (selected_entries != 0).sum(dim=0).float()
        disjoint_aggs = torch.sum(selected_entries, dim=0) / torch.clamp(non_zero_counts, min=1)
    elif merge_key == "sum":
        disjoint_aggs = torch.sum(selected_entries, dim=0)
    elif merge_key == "max":
        disjoint_aggs = selected_entries.abs().max(dim=0)[0]
        if sign_to_mult is not None:
            disjoint_aggs = disjoint_aggs * sign_to_mult
    else:
        raise ValueError(f"Merge method {merge_func} is not defined.")

    return disjoint_aggs


def ties_merging(
    flat_task_checks: torch.Tensor,
    reset_thresh: float | None = None,
    merge_func: str = "dis-mean",
):
    """Perform TIES merging on a stack of flat task vectors.

    Inputs
    - flat_task_checks: Tensor of shape (n, d). Each row is a task vector (fine-tuned - base) flattened.
    - reset_thresh: fraction or percent of parameters to KEEP (e.g., 0.2 or 20 for top 20% by |.|). If None, no trimming.
    - merge_func: one of {"dis-mean", "dis-sum", "dis-max"}

    Returns
    - merged_tv: 1D tensor of shape (d,), the merged task vector in flat form
    """
    all_checks = flat_task_checks.clone()

    if reset_thresh is None:
        updated_checks = all_checks
    else:
        updated_checks, _ = topk_values_mask(all_checks, K=reset_thresh, return_mask=False)

    print("RESOLVING SIGN")
    final_signs = resolve_sign(updated_checks)
    assert final_signs is not None

    print(f"Disjoint AGGREGATION: {merge_func}")
    merged_tv = disjoint_merge(updated_checks, merge_func, final_signs)

    return merged_tv


# =============================
# Simple Task Vector Merging (baseline)
# =============================
def aggregate(T: torch.Tensor, agg_type: str, dim: int = 0):
    if agg_type == "mean":
        result = torch.mean(T, dim=dim)
    elif agg_type == "sum":
        result = torch.sum(T, dim=dim)
    else:
        raise ValueError("Invalid agg_type: %s" % agg_type)
    return result


def tv_merging(tv_flat_checks: torch.Tensor):
    """Baseline merging by summing task vectors (no trimming/sign selection)."""
    all_checks = tv_flat_checks.clone()
    tv_merged_check = aggregate(all_checks, "sum")
    return tv_merged_check
    

