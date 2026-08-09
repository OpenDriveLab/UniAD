#---------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
#---------------------------------------------------------------------------------#

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """A frozen linear layer with an optional low-rank update."""

    def __init__(self, base, rank=8, alpha=16.0, dropout=0.0):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError('base must be an nn.Linear')
        if rank <= 0:
            raise ValueError('rank must be positive')

        self.base = base
        self.rank = rank
        self.scale = alpha / rank
        self.enabled = True
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)

        self.base.requires_grad_(False)

    @property
    def weight(self):
        if not self.enabled:
            return self.base.weight
        # MultiheadAttention consumes out_proj.weight directly instead of
        # calling out_proj.forward, so expose the merged weight here as well.
        return self.base.weight + self.scale * (self.lora_B @ self.lora_A)

    @property
    def bias(self):
        return self.base.bias

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    def forward(self, inputs):
        output = self.base(inputs)
        if self.enabled:
            update = self.dropout(inputs) @ self.lora_A.t() @ self.lora_B.t()
            output = output + self.scale * update
        return output

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """Accept the original nn.Linear checkpoint key layout."""
        for name in ('weight', 'bias'):
            legacy_key = prefix + name
            base_key = prefix + 'base.' + name
            if legacy_key in state_dict and base_key not in state_dict:
                state_dict[base_key] = state_dict.pop(legacy_key)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


def _replace_linear(parent, name, rank, alpha, dropout):
    module = LoRALinear(
        getattr(parent, name), rank=rank, alpha=alpha, dropout=dropout)
    setattr(parent, name, module)
    return module


def apply_lora_to_planning_head(head, rank=8, alpha=16.0, dropout=0.0):
    """Mount LoRA on planner projections, leaving the BEV adapter frozen."""
    modules = []

    if isinstance(head.mlp_fuser[0], nn.Linear):
        modules.append(_replace_linear(
            head.mlp_fuser, '0', rank, alpha, dropout))

    for layer in head.attn_module.layers:
        for name in ('linear1', 'linear2'):
            if isinstance(getattr(layer, name), nn.Linear):
                modules.append(_replace_linear(
                    layer, name, rank, alpha, dropout))
        for name in ('self_attn', 'multihead_attn'):
            attention = getattr(layer, name, None)
            if attention is not None and isinstance(attention.out_proj, nn.Linear):
                modules.append(_replace_linear(
                    attention, 'out_proj', rank, alpha, 0.0))

    for index, module in enumerate(head.reg_branch):
        if isinstance(module, nn.Linear):
            modules.append(_replace_linear(
                head.reg_branch, str(index), rank, alpha, dropout))

    # The modules are already registered at their original locations.  Keep a
    # plain list here to avoid duplicate state-dict keys.
    head._lora_modules = modules
    return modules


def set_lora_enabled(head, enabled):
    for module in getattr(head, '_lora_modules', []):
        module.enabled = enabled


def lora_parameters(head):
    for module in getattr(head, '_lora_modules', []):
        yield module.lora_A
        yield module.lora_B
