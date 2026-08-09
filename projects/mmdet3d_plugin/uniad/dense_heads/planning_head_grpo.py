#---------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
#---------------------------------------------------------------------------------#

import torch
import torch.nn as nn
from einops import rearrange
from mmdet.models.builder import HEADS

from projects.mmdet3d_plugin.losses.grpo_loss import (
    PlannerReward,
    compute_group_advantages,
    diagonal_gaussian_kl,
    gaussian_log_prob,
    group_relative_policy_loss,
)
from projects.mmdet3d_plugin.models.utils.lora import (
    apply_lora_to_planning_head,
    lora_parameters,
    set_lora_enabled,
)
from .planning_head import PlanningHeadSingleMode


@HEADS.register_module()
class PlanningHeadGRPO(PlanningHeadSingleMode):
    """Opt-in group-relative post-training for the UniAD planning head."""

    def __init__(self, grpo=None, **kwargs):
        grpo = {} if grpo is None else grpo
        super().__init__(**kwargs)
        defaults = dict(
            num_samples=6,
            kl_weight=0.01,
            aux_weight=1.0,
            lora_rank=8,
            lora_alpha=16.0,
            lora_dropout=0.0,
            init_log_std=-1.5,
            min_log_std=-5.0,
            max_log_std=1.0,
            ade_weight=1.0,
            fde_weight=1.0,
            comfort_weight=0.1,
        )
        defaults.update(grpo)
        self.grpo_cfg = defaults

        # The frozen SFT reference and trainable policy must see the same state.
        # Disable the base decoder dropout for this opt-in post-training head.
        for module in self.attn_module.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0

        apply_lora_to_planning_head(
            self,
            rank=defaults['lora_rank'],
            alpha=defaults['lora_alpha'],
            dropout=defaults['lora_dropout'],
        )
        self.log_std = nn.Parameter(torch.full(
            (self.planning_steps, 2), defaults['init_log_std']))
        self.reward_fn = PlannerReward(
            planning_steps=self.planning_steps,
            ade_weight=defaults['ade_weight'],
            fde_weight=defaults['fde_weight'],
            comfort_weight=defaults['comfort_weight'],
        )

    def _encode_plan_query(self, bev_embed, bev_pos, sdc_traj_query,
                           sdc_track_query, command):
        sdc_track_query = sdc_track_query.detach()
        sdc_traj_query = sdc_traj_query[-1]
        num_modes = sdc_traj_query.shape[1]
        sdc_track_query = sdc_track_query[:, None].expand(
            -1, num_modes, -1)
        command = torch.as_tensor(
            command, device=bev_embed.device, dtype=torch.long).reshape(-1)
        navi_embed = self.navi_embed.weight[command]
        navi_embed = navi_embed[:, None].expand(-1, num_modes, -1)

        plan_query = torch.cat(
            [sdc_traj_query, sdc_track_query, navi_embed], dim=-1)
        plan_query = self.mlp_fuser(plan_query).max(1, keepdim=True)[0]
        plan_query = rearrange(plan_query, 'b p c -> p b c')

        bev_pos = rearrange(bev_pos, 'b c h w -> (h w) b c')
        bev_feat = bev_embed + bev_pos
        if self.with_adapter:
            bev_feat = rearrange(
                bev_feat, '(h w) b c -> b c h w',
                h=self.bev_h, w=self.bev_w)
            bev_feat = bev_feat + self.bev_adapter(bev_feat)
            bev_feat = rearrange(bev_feat, 'b c h w -> (h w) b c')

        plan_query = plan_query + self.pos_embed.weight[None]
        return self.attn_module(plan_query, bev_feat)

    def _mean_increments(self, plan_query):
        return self.reg_branch(plan_query).view(
            -1, self.planning_steps, 2)

    def _policy_mean(self, bev_embed, bev_pos, sdc_traj_query,
                     sdc_track_query, command):
        query = self._encode_plan_query(
            bev_embed, bev_pos, sdc_traj_query, sdc_track_query, command)
        return self._mean_increments(query)

    @staticmethod
    def _positions(increments):
        return torch.cumsum(increments, dim=-2)

    def _sample_increments(self, mean):
        log_std = self.log_std.clamp(
            self.grpo_cfg['min_log_std'], self.grpo_cfg['max_log_std'])
        noise = torch.randn(
            mean.shape[0], self.grpo_cfg['num_samples'],
            self.planning_steps, 2, device=mean.device, dtype=mean.dtype)
        return mean[:, None] + log_std.exp()[None, None] * noise, log_std

    def forward(self, bev_embed, occ_mask, bev_pos, sdc_traj_query,
                sdc_track_query, command):
        mean = self._policy_mean(
            bev_embed, bev_pos, sdc_traj_query, sdc_track_query, command)
        trajectory = self._positions(mean)
        if self.use_col_optim and not self.training:
            if occ_mask is None:
                raise ValueError('occ_mask is required for collision optimization')
            trajectory = self.collision_optimization(trajectory, occ_mask)
        return dict(sdc_traj=trajectory, sdc_traj_all=trajectory)

    def forward_train(self, bev_embed, outs_motion={}, sdc_planning=None,
                      sdc_planning_mask=None, command=None,
                      gt_future_boxes=None):
        sdc_traj_query = outs_motion['sdc_traj_query']
        sdc_track_query = outs_motion['sdc_track_query']
        bev_pos = outs_motion['bev_pos']

        mean = self._policy_mean(
            bev_embed, bev_pos, sdc_traj_query, sdc_track_query, command)
        increments, log_std = self._sample_increments(mean)
        trajectories = self._positions(increments)

        # Score-function gradients require fixed sampled actions.  Without the
        # detach below, reparameterization cancels the mean-policy gradient.
        log_prob = gaussian_log_prob(
            increments.detach(), mean, log_std,
            self.grpo_cfg['min_log_std'], self.grpo_cfg['max_log_std'])
        with torch.no_grad():
            set_lora_enabled(self, False)
            try:
                reference_mean = self._policy_mean(
                    bev_embed, bev_pos, sdc_traj_query,
                    sdc_track_query, command)
            finally:
                set_lora_enabled(self, True)

            reward, metrics = self.reward_fn(
                trajectories, sdc_planning, sdc_planning_mask)
            advantages = compute_group_advantages(reward)

        loss_grpo = group_relative_policy_loss(log_prob, advantages)
        kl = diagonal_gaussian_kl(mean, reference_mean, log_std)
        mean_trajectory = self._positions(mean)
        target = sdc_planning[..., :self.planning_steps, :2]
        mask = torch.any(
            sdc_planning_mask[..., :self.planning_steps], dim=-1)
        if target.dim() == 4:
            target = target[:, 0]
            mask = mask[:, 0]
        loss_aux = self.loss_planning(mean_trajectory, target, mask)

        losses = dict(
            loss_grpo=loss_grpo,
            loss_kl=kl * self.grpo_cfg['kl_weight'],
            loss_ade=loss_aux * self.grpo_cfg['aux_weight'],
            kl=kl.detach(),
            mean_path_ade=loss_aux.detach(),
            **metrics,
        )
        outputs = dict(
            sdc_traj=mean_trajectory,
            sdc_traj_all=mean_trajectory,
            traj_samples=trajectories.detach(),
        )
        return dict(losses=losses, outs_motion=outputs)

    def freeze_for_posttrain(self):
        self.requires_grad_(False)
        for parameter in lora_parameters(self):
            parameter.requires_grad = True
        self.log_std.requires_grad = True
        if self.with_adapter:
            self.bev_adapter.eval()
