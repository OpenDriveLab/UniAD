#---------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
#---------------------------------------------------------------------------------#

import math

import torch
import torch.nn as nn
from mmdet.models import LOSSES


LOG_2PI = math.log(2.0 * math.pi)


def planning_target(sdc_planning, sdc_planning_mask, planning_steps):
    target = sdc_planning[..., :planning_steps, :2]
    mask = torch.any(sdc_planning_mask[..., :planning_steps], dim=-1)
    if target.dim() == 4:
        target = target[:, 0]
    if mask.dim() == 3:
        mask = mask[:, 0]
    return target, mask


def masked_ade(trajectories, target, mask):
    error = torch.norm(trajectories - target[:, None], dim=-1)
    valid = mask[:, None].to(error.dtype)
    return (error * valid).sum(-1) / valid.sum(-1).clamp_min(1.0)


def masked_fde(trajectories, target, mask):
    batch = torch.arange(trajectories.shape[0], device=trajectories.device)
    last = mask.long().sum(-1).clamp_min(1) - 1
    return torch.norm(
        trajectories[batch, :, last] - target[batch, last, None], dim=-1)


def comfort_penalty(trajectories):
    if trajectories.shape[-2] < 4:
        return trajectories.new_zeros(trajectories.shape[:2])
    velocity = torch.diff(trajectories, dim=-2, prepend=trajectories[..., :1, :])
    acceleration = torch.diff(velocity, dim=-2)
    jerk = torch.diff(acceleration, dim=-2)
    return jerk.square().mean(dim=(-1, -2))


@LOSSES.register_module()
class PlannerReward(nn.Module):
    """Open-loop trajectory reward used by the group-relative objective."""

    def __init__(self, planning_steps=6, ade_weight=1.0, fde_weight=1.0,
                 comfort_weight=0.1):
        super().__init__()
        self.planning_steps = planning_steps
        self.ade_weight = ade_weight
        self.fde_weight = fde_weight
        self.comfort_weight = comfort_weight

    def forward(self, trajectories, sdc_planning, sdc_planning_mask):
        target, mask = planning_target(
            sdc_planning, sdc_planning_mask, self.planning_steps)
        ade = masked_ade(trajectories, target, mask)
        fde = masked_fde(trajectories, target, mask)
        comfort = comfort_penalty(trajectories)
        reward = -(self.ade_weight * ade + self.fde_weight * fde
                   + self.comfort_weight * comfort)
        metrics = dict(
            reward=reward.mean(),
            reward_std=reward.std(dim=1, unbiased=False).mean(),
            sample_ade=ade.mean(),
            sample_fde=fde.mean(),
            sample_best_ade=ade.min(dim=1).values.mean(),
            sample_best_fde=fde.min(dim=1).values.mean(),
            comfort=comfort.mean(),
        )
        return reward, metrics


def compute_group_advantages(reward, eps=1e-8):
    """Normalize rewards within each rollout group and stop reward gradients."""
    reward = reward.detach()
    centered = reward - reward.mean(dim=1, keepdim=True)
    scale = reward.std(dim=1, keepdim=True, unbiased=False)
    return centered / (scale + eps)


def gaussian_log_prob(actions, mean, log_std, min_log_std=-5.0,
                      max_log_std=2.0):
    """Log probability of [B, G, T, 2] actions under a diagonal Gaussian."""
    log_std = log_std.clamp(min_log_std, max_log_std)
    while mean.dim() < actions.dim():
        mean = mean.unsqueeze(1)
    log_std = log_std.view(1, 1, *log_std.shape)
    log_prob = -0.5 * (
        ((actions - mean) / log_std.exp()).square()
        + 2.0 * log_std + LOG_2PI)
    return log_prob.sum(dim=(-1, -2))


def group_relative_policy_loss(log_prob, advantages):
    """Single-update GRPO score-function objective (no stale PPO ratio)."""
    if log_prob.shape != advantages.shape:
        raise ValueError('log_prob and advantages must have the same shape')
    return -(log_prob * advantages.detach()).mean()


def diagonal_gaussian_kl(mean, reference_mean, log_std):
    """Exact KL to a reference Gaussian with the same diagonal variance."""
    inverse_variance = torch.exp(-2.0 * log_std.clamp(-5.0, 2.0))
    return 0.5 * ((mean - reference_mean).square()
                  * inverse_variance).sum(dim=(-1, -2)).mean()
