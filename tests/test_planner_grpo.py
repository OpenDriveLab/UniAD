import copy
import importlib.util
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn


class _Registry:
    def register_module(self):
        return lambda cls: cls


def _load(name, relative_path):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(name, root / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Keep these core unit tests independent of an installed MMDetection stack.
mmdet = types.ModuleType('mmdet')
mmdet_models = types.ModuleType('mmdet.models')
mmdet_models.LOSSES = _Registry()
mmdet.models = mmdet_models
sys.modules.setdefault('mmdet', mmdet)
sys.modules.setdefault('mmdet.models', mmdet_models)

grpo_loss = _load(
    'grpo_loss', 'projects/mmdet3d_plugin/losses/grpo_loss.py')
lora = _load('lora', 'projects/mmdet3d_plugin/models/utils/lora.py')
compute_group_advantages = grpo_loss.compute_group_advantages
diagonal_gaussian_kl = grpo_loss.diagonal_gaussian_kl
gaussian_log_prob = grpo_loss.gaussian_log_prob
group_relative_policy_loss = grpo_loss.group_relative_policy_loss
comfort_penalty = grpo_loss.comfort_penalty
masked_fde = grpo_loss.masked_fde
LoRALinear = lora.LoRALinear


def test_lora_starts_at_the_frozen_base():
    base = nn.Linear(8, 4)
    reference = copy.deepcopy(base)
    adapter = LoRALinear(base, rank=2, alpha=4.0)
    inputs = torch.randn(3, 8)
    assert torch.equal(adapter(inputs), reference(inputs))
    assert not adapter.base.weight.requires_grad


def test_lora_off_recovers_base_after_update():
    adapter = LoRALinear(nn.Linear(8, 4), rank=2, alpha=4.0)
    adapter.lora_B.data.normal_()
    inputs = torch.randn(3, 8)
    adapter.enabled = False
    assert torch.equal(adapter(inputs), adapter.base(inputs))


def test_lora_merged_weight_supports_functional_callers():
    adapter = LoRALinear(nn.Linear(8, 4), rank=2, alpha=4.0)
    adapter.lora_B.data.normal_()
    inputs = torch.randn(3, 8)
    expected = torch.nn.functional.linear(inputs, adapter.weight, adapter.bias)
    assert torch.allclose(adapter(inputs), expected, atol=1e-6)


def test_lora_loads_original_linear_checkpoint_keys():
    original = nn.Linear(8, 4)
    adapter = LoRALinear(nn.Linear(8, 4), rank=2, alpha=4.0)
    adapter.load_state_dict(original.state_dict(), strict=False)
    assert torch.equal(adapter.base.weight, original.weight)
    assert torch.equal(adapter.base.bias, original.bias)


def test_group_advantages_are_centered_and_detached():
    reward = torch.tensor([[1.0, 2.0, 4.0]], requires_grad=True)
    advantages = compute_group_advantages(reward)
    assert torch.allclose(advantages.mean(1), torch.zeros(1), atol=1e-6)
    assert not advantages.requires_grad


def test_fde_is_zero_without_valid_future_steps():
    trajectories = torch.full((1, 3, 6, 2), 10.0)
    target = torch.zeros(1, 6, 2)
    mask = torch.zeros(1, 6, dtype=torch.bool)
    assert torch.equal(
        masked_fde(trajectories, target, mask), torch.zeros(1, 3))


def test_comfort_includes_origin_to_first_waypoint():
    trajectories = torch.zeros(1, 1, 6, 2)
    trajectories[..., 0] = 10.0
    assert comfort_penalty(trajectories).item() > 0


def test_detached_actions_give_policy_mean_gradient():
    mean = torch.zeros(1, 2, 2, requires_grad=True)
    actions = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]],
                             [[-0.25, 0.0], [-0.25, 0.0]]]])
    advantages = torch.tensor([[1.0, -1.0]])
    log_prob = gaussian_log_prob(actions.detach(), mean, torch.zeros(2, 2))
    group_relative_policy_loss(log_prob, advantages).backward()
    assert mean.grad is not None
    assert mean.grad.abs().sum() > 0


def test_exact_reference_kl_is_non_negative():
    mean = torch.randn(2, 6, 2)
    reference = torch.randn(2, 6, 2)
    log_std = torch.full((6, 2), -1.5)
    kl = diagonal_gaussian_kl(mean, reference, log_std)
    assert kl >= 0
    assert diagonal_gaussian_kl(mean, mean, log_std) == 0
