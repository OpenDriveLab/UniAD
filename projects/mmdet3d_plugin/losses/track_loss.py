#---------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
# Modified from MOTR (https://github.com/megvii-research/MOTR)                    #
#---------------------------------------------------------------------------------#

import copy
from distutils.command.build import build
import math
from xmlrpc.client import Boolean
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.nn as nn
from typing import List
from projects.mmdet3d_plugin.uniad.dense_heads.track_head_plugin import Instances
from mmdet.core import build_assigner
from mmdet.models import build_loss
from mmdet.models.builder import LOSSES
from mmdet.core import reduce_mean
from mmdet3d.core.bbox.iou_calculators.iou3d_calculator import (
    bbox_overlaps_nearest_3d as iou_3d, )
from projects.mmdet3d_plugin.core.bbox.util import denormalize_bbox


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


@torch.no_grad()
def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


@LOSSES.register_module()
class ClipMatcher(nn.Module):
    def __init__(
            self,
            num_classes,
            weight_dict,
            code_weights=[
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 0.2
            ],
            loss_past_traj_weight=1.0,
            assigner=dict(
                type="HungarianAssigner3D",
                cls_cost=dict(type="FocalLossCost", weight=2.0),
                reg_cost=dict(type="BBox3DL1Cost", weight=0.25),
                pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            ),
            loss_cls=dict(type="FocalLoss",
                          use_sigmoid=True,
                          gamma=2.0,
                          alpha=0.25,
                          loss_weight=2.0),
            loss_bbox=dict(type="L1Loss", loss_weight=0.25),
    ):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = build_assigner(assigner)
        self.loss_cls = build_loss(loss_cls)
        self.loss_bboxes = build_loss(loss_bbox)
        self.loss_predictions = nn.SmoothL1Loss(reduction="none", beta=1.0)
        self.register_buffer("code_weights",
                             torch.tensor(code_weights, requires_grad=False))

        self.weight_dict = weight_dict
        self.loss_past_traj_weight = loss_past_traj_weight
        # self.losses = ['labels', 'boxes', 'cardinality']
        self.losses = ["labels", "boxes", "past_trajs"]
        self.focal_loss = True
        self.losses_dict = {}
        self._current_frame_idx = 0

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def initialize_for_single_clip(self, gt_instances: List[Instances]):
        self.gt_instances = gt_instances
        self.num_samples = 0
        self.sample_device = None
        self._current_frame_idx = 0
        self.losses_dict = {}

    def _step(self):
        self._current_frame_idx += 1

    def calc_loss_for_track_scores(self, track_instances: Instances):
        frame_id = self._current_frame_idx - 1
        gt_instances = self.gt_instances[frame_id]
        outputs = {
            "pred_logits": track_instances.track_scores[None],
        }
        device = track_instances.track_scores.device

        num_tracks = len(track_instances)
        src_idx = torch.arange(num_tracks, dtype=torch.long, device=device)
        tgt_idx = (track_instances.matched_gt_idxes
                   )  # -1 for FP tracks and disappeared tracks

        track_losses = self.get_loss(
            "labels",
            outputs=outputs,
            gt_instances=[gt_instances],
            indices=[(src_idx, tgt_idx)],
            num_boxes=1,
        )
        self.losses_dict.update({
            "frame_{}_track_{}".format(frame_id, key): value
            for key, value in track_losses.items()
        })

    def get_num_boxes(self, num_samples):
        num_boxes = torch.as_tensor(num_samples,
                                    dtype=torch.float,
                                    device=self.sample_device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        return num_boxes

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices):
        """Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs["pred_logits"]
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v.labels) for v in targets],
                                      device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) !=
                     pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {"cardinality_error": card_err}
        return losses

    def get_loss(self, loss, outputs, gt_instances, indices, **kwargs):
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
            "past_trajs": self.loss_past_trajs,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, gt_instances, indices, **kwargs)

    def loss_past_trajs(self, outputs, gt_instances: List[Instances],
                   indices: List[tuple]):
        # We ignore the regression loss of the track-disappear slots.
        # TODO: Make this filter process more elegant.
        filtered_idx = []
        for src_per_img, tgt_per_img in indices:
            keep = tgt_per_img != -1
            filtered_idx.append((src_per_img[keep], tgt_per_img[keep]))
        indices = filtered_idx
        idx = self._get_src_permutation_idx(indices)
        src_trajs = outputs["pred_past_trajs"][idx]
        target_trajs = torch.cat(
            [
                gt_per_img.past_traj[i]
                for gt_per_img, (_, i) in zip(gt_instances, indices)
            ],
            dim=0,
        )
        target_trajs_mask = torch.cat(
            [
                gt_per_img.past_traj_mask[i]
                for gt_per_img, (_, i) in zip(gt_instances, indices)
            ],
            dim=0,
        )

        # for pad target, don't calculate regression loss, judged by whether obj_id=-1
        target_obj_ids = torch.cat(
            [
                gt_per_img.obj_ids[i]
                for gt_per_img, (_, i) in zip(gt_instances, indices)
            ],
            dim=0,
        )  # size(16)
        # [num_matched]
        mask = target_obj_ids != -1
        loss_trajs = self.compute_past_traj_loss(src_trajs[mask], target_trajs[mask], target_trajs_mask[mask])
        losses = {}
        losses["loss_past_trajs"] = loss_trajs * self.loss_past_traj_weight
        return losses
    
    def compute_past_traj_loss(self, src, tgt, tgt_mask):
        loss = torch.abs(src - tgt) * tgt_mask
        return torch.sum(loss)/ (torch.sum(tgt_mask>0) + 1e-5)

    def loss_boxes(self, outputs, gt_instances: List[Instances],
                   indices: List[tuple]):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        # We ignore the regression loss of the track-disappear slots.
        # TODO: Make this filter process more elegant.
        filtered_idx = []
        for src_per_img, tgt_per_img in indices:
            keep = tgt_per_img != -1
            filtered_idx.append((src_per_img[keep], tgt_per_img[keep]))
        indices = filtered_idx
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        sdc_boxes = outputs["pred_sdc_boxes"][0, -1:]
        target_sdc_boxes = gt_instances[0].sdc_boxes[:1]
        target_boxes = torch.cat(
            [
                gt_per_img.boxes[i]
                for gt_per_img, (_, i) in zip(gt_instances, indices)
            ],
            dim=0,
        )
        
        src_boxes = torch.cat([src_boxes, sdc_boxes], dim=0)
        target_boxes = torch.cat([target_boxes, target_sdc_boxes], dim=0)

        # for pad target, don't calculate regression loss, judged by whether obj_id=-1
        target_obj_ids = torch.cat(
            [
                gt_per_img.obj_ids[i]
                for gt_per_img, (_, i) in zip(gt_instances, indices)
            ],
            dim=0,
        )
        # [num_matched]

        target_obj_ids = torch.cat([target_obj_ids, torch.zeros(1).to(target_obj_ids.device)], dim=0)
        mask = target_obj_ids != -1
        bbox_weights = torch.ones_like(target_boxes) * self.code_weights
        avg_factor = src_boxes[mask].size(0)
        avg_factor = reduce_mean(target_boxes.new_tensor([avg_factor]))
        loss_bbox = self.loss_bboxes(
            src_boxes[mask],
            target_boxes[mask],
            bbox_weights[mask],
            avg_factor=avg_factor.item(),
        )
        
        losses = {}
        losses["loss_bbox"] = loss_bbox

        return losses

    def loss_labels(self,
                    outputs,
                    gt_instances: List[Instances],
                    indices,
                    log=False):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]

        indices: [(src_idx, tgt_idx)]
        """
        # [bs=1, num_query, num_classes]
        src_logits = outputs["pred_logits"]
        sdc_logits = outputs["pred_sdc_logits"]
        # batch_idx, src_idx
        idx = self._get_src_permutation_idx(indices)
        # [bs, num_query]
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        # The matched gt for disappear track query is set -1.
        labels = []
        for gt_per_img, (_, J) in zip(gt_instances, indices):
            labels_per_img = torch.ones_like(J) * self.num_classes
            # set labels of track-appear slots to num_classes
            if len(gt_per_img) > 0:
                labels_per_img[J != -1] = gt_per_img.labels[J[J != -1]]
            labels.append(labels_per_img)
        # [num_matched]
        target_classes_o = torch.cat(labels)
        # [bs, num_query]
        target_classes[idx] = target_classes_o
        target_sdc_classes = gt_instances[0].sdc_labels[0:1].unsqueeze(0)
        if sdc_logits is not None:
            src_logits = torch.cat([src_logits, sdc_logits], dim=1)
            target_classes = torch.cat([target_classes, target_sdc_classes], dim=1)
        label_weights = torch.ones_like(target_classes)
        # float tensor
        avg_factor = target_classes_o.numel(
        )  # pos + mathced gt for disapper track
        avg_factor += 1 # sdc
        
        avg_factor = reduce_mean(src_logits.new_tensor([avg_factor]))
        loss_ce = self.loss_cls(
            src_logits.flatten(0, 1),
            target_classes.flatten(0),
            label_weights.flatten(0),
            avg_factor,
        )

        losses = {"loss_cls": loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses["class_error"] = 100 - accuracy(src_logits[idx],
                                                   target_classes_o)[0]

        return losses

    def match_for_single_frame(self,
                               outputs: dict,
                               dec_lvl: int,
                               if_step=False,
                               ):
        outputs_without_aux = {
            k: v
            for k, v in outputs.items() if k != "aux_outputs"
        }

        gt_instances_i = self.gt_instances[
            self._current_frame_idx]  # gt instances of i-th image.
        track_instances: Instances = outputs_without_aux["track_instances"]
        pred_logits_i = track_instances.pred_logits
        pred_boxes_i = track_instances.pred_boxes
        # modified the hard code, 900:901, sdc query
        pred_sdc_logits_i = track_instances.pred_logits[900:901].unsqueeze(0) 
        pred_sdc_boxes_i = track_instances.pred_boxes[900:901].unsqueeze(0) 
        # -2 means the sdc query in this code
        track_instances.obj_idxes[900]=-2
        pred_past_trajs_i = track_instances.pred_past_trajs  # predicted past trajs of i-th image.

        obj_idxes = gt_instances_i.obj_ids
        obj_idxes_list = obj_idxes.detach().cpu().numpy().tolist()
        obj_idx_to_gt_idx = {
            obj_idx: gt_idx
            for gt_idx, obj_idx in enumerate(obj_idxes_list)
        }
        outputs_i = {
            "pred_logits": pred_logits_i.unsqueeze(0),
            "pred_sdc_logits": pred_sdc_logits_i,
            "pred_boxes": pred_boxes_i.unsqueeze(0),
            "pred_sdc_boxes": pred_sdc_boxes_i,
            "pred_past_trajs": pred_past_trajs_i.unsqueeze(0),
        }
        # step1. inherit and update the previous tracks.
        num_disappear_track = 0
        for j in range(len(track_instances)):
            obj_id = track_instances.obj_idxes[j].item()
            # set new target idx.
            if obj_id >= 0:
                if obj_id in obj_idx_to_gt_idx:
                    track_instances.matched_gt_idxes[j] = obj_idx_to_gt_idx[
                        obj_id]
                else:
                    num_disappear_track += 1
                    track_instances.matched_gt_idxes[
                        j] = -1  # track-disappear case.
            else:
                track_instances.matched_gt_idxes[j] = -1

        full_track_idxes = torch.arange(
            len(track_instances), dtype=torch.long).to(pred_logits_i.device)
        # previsouly tracked, which is matched by rule
        matched_track_idxes = track_instances.obj_idxes >= 0
        prev_matched_indices = torch.stack(
            [
                full_track_idxes[matched_track_idxes],
                track_instances.matched_gt_idxes[matched_track_idxes],
            ],
            dim=1,
        ).to(pred_logits_i.device)

        # step2. select the unmatched slots.
        # note that the FP tracks whose obj_idxes are -2 will not be selected here.
        unmatched_track_idxes = full_track_idxes[track_instances.obj_idxes ==
                                                 -1]

        # step3. select the untracked gt instances (new tracks).
        tgt_indexes = track_instances.matched_gt_idxes
        tgt_indexes = tgt_indexes[tgt_indexes != -1]

        tgt_state = torch.zeros(len(gt_instances_i)).to(pred_logits_i.device)
        tgt_state[tgt_indexes] = 1
        # new tgt indexes
        untracked_tgt_indexes = torch.arange(len(gt_instances_i)).to(
            pred_logits_i.device)[tgt_state == 0]
        # untracked_tgt_indexes = select_unmatched_indexes(tgt_indexes, len(gt_instances_i))
        # [num_untracked]
        untracked_gt_instances = gt_instances_i[untracked_tgt_indexes]

        def match_for_single_decoder_layer(unmatched_outputs, matcher):
            bbox_preds, cls_preds = (
                unmatched_outputs["pred_boxes"],
                unmatched_outputs["pred_logits"],
            )
            bs, num_querys = bbox_preds.shape[:2]
            # Also concat the target labels and boxes
            targets = [untracked_gt_instances]
            if isinstance(targets[0], Instances):
                # [num_box], [num_box, 9] (un-normalized bboxes)
                gt_labels = torch.cat(
                    [gt_per_img.labels for gt_per_img in targets])
                gt_bboxes = torch.cat(
                    [gt_per_img.boxes for gt_per_img in targets])
            else:
                gt_labels = torch.cat([v["labels"] for v in targets])
                gt_bboxes = torch.cat([v["boxes"] for v in targets])

            bbox_pred = bbox_preds[0]
            cls_pred = cls_preds[0]

            src_idx, tgt_idx = matcher.assign(bbox_pred, cls_pred, gt_bboxes,
                                              gt_labels)
            if src_idx is None:
                return None
            # concat src and tgt.
            new_matched_indices = torch.stack([
                unmatched_track_idxes[src_idx], untracked_tgt_indexes[tgt_idx]
            ],
                                              dim=1).to(pred_logits_i.device)
            return new_matched_indices

        # step4. do matching between the unmatched slots and GTs.
        unmatched_outputs = {
            # [bs, num_pred, num_classes]
            "pred_logits":
            track_instances.pred_logits[unmatched_track_idxes].unsqueeze(0),
            # [bs, num_pred, box_dim]
            "pred_boxes":
            track_instances.pred_boxes[unmatched_track_idxes].unsqueeze(0),
        }
        # [num_new_matched, 2]
        new_matched_indices = match_for_single_decoder_layer(
            unmatched_outputs, self.matcher)

        # step5. update obj_idxes according to the new matching result.
        if new_matched_indices is not None:
            track_instances.obj_idxes[
                new_matched_indices[:, 0]] = gt_instances_i.obj_ids[
                    new_matched_indices[:, 1]].long()
            track_instances.matched_gt_idxes[
                new_matched_indices[:, 0]] = new_matched_indices[:, 1]

            # step6. calculate iou3d.
            active_idxes = (track_instances.obj_idxes >=
                            0) & (track_instances.matched_gt_idxes >= 0)
            active_track_boxes = track_instances.pred_boxes[active_idxes]
            with torch.no_grad():
                if len(active_track_boxes) > 0:
                    gt_boxes = gt_instances_i.boxes[
                        track_instances.matched_gt_idxes[active_idxes]]
                    iou_3ds = iou_3d(
                        denormalize_bbox(gt_boxes, None)[..., :7],
                        denormalize_bbox(active_track_boxes, None)[..., :7],
                    )
                    track_instances.iou[active_idxes] = torch.tensor([
                        iou_3ds[i, i] for i in range(gt_boxes.shape[0])
                    ]).to(gt_boxes.device)

            # step7. merge the unmatched pairs and the matched pairs.
            # [num_new_macthed + num_prev_mathed, 2]
            matched_indices = torch.cat(
                [new_matched_indices, prev_matched_indices], dim=0)
        else:
            matched_indices = prev_matched_indices
        # step8. calculate losses.
        self.num_samples += len(gt_instances_i) + num_disappear_track
        self.sample_device = pred_logits_i.device

        for loss in self.losses:
            new_track_loss = self.get_loss(
                loss,
                outputs=outputs_i,
                gt_instances=[gt_instances_i],
                indices=[(matched_indices[:, 0], matched_indices[:, 1])],
            )
            self.losses_dict.update({
                "frame_{}_{}_{}".format(self._current_frame_idx, key, dec_lvl):
                value
                for key, value in new_track_loss.items()
            })
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                unmatched_outputs_layer = {
                    "pred_logits":
                    aux_outputs["pred_logits"][
                        0, unmatched_track_idxes].unsqueeze(0),
                    "pred_boxes":
                    aux_outputs["pred_boxes"][
                        0, unmatched_track_idxes].unsqueeze(0),
                }
                new_matched_indices_layer = match_for_single_decoder_layer(
                    unmatched_outputs_layer, self.matcher)
                matched_indices_layer = torch.cat(
                    [new_matched_indices_layer, prev_matched_indices], dim=0)
                for loss in self.losses:
                    if loss == "masks":
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    l_dict = self.get_loss(
                        loss,
                        aux_outputs,
                        gt_instances=[gt_instances_i],
                        indices=[(matched_indices_layer[:, 0],
                                  matched_indices_layer[:, 1])],
                    )
                    self.losses_dict.update({
                        "frame_{}_aux{}_{}".format(self._current_frame_idx, i,
                                                   key): value
                        for key, value in l_dict.items()
                    })
        if if_step:
            self._step()
        return track_instances, matched_indices

    def forward(self, outputs, input_data: dict):
        # losses of each frame are calculated during the model's forwarding and are outputted by the model as outputs['losses_dict].
        losses = outputs.pop("losses_dict")
        num_samples = self.get_num_boxes(self.num_samples)
        for loss_name, loss in losses.items():
            losses[loss_name] /= num_samples
        return losses

    def prediction_loss(self, track_instances, predictions):

        decay_ratio = 1.0
        for i in range(self._current_frame_idx, len(self.gt_instances)):
            gt_instances_i = self.gt_instances[
                i]  # gt instances of i-th image.

            pred_boxes_i = predictions[i - self._current_frame_idx]

            obj_idxes = gt_instances_i.obj_ids
            obj_idxes_list = obj_idxes.detach().cpu().numpy().tolist()
            obj_idx_to_gt_idx = {
                obj_idx: gt_idx
                for gt_idx, obj_idx in enumerate(obj_idxes_list)
            }

            num_paired = 0
            for j in range(len(track_instances)):
                obj_id = track_instances.obj_idxes[j].item()
                # set new target idx.
                if obj_id >= 0:
                    if obj_id in obj_idx_to_gt_idx:
                        track_instances.matched_gt_idxes[
                            j] = obj_idx_to_gt_idx[obj_id]
                        num_paired += 1
                    else:
                        track_instances.matched_gt_idxes[
                            j] = -1  # track-disappear case.
                else:
                    track_instances.matched_gt_idxes[j] = -1

            if num_paired > 0:
                if_paired_i = track_instances.matched_gt_idxes >= 0

                paired_pred_boxes_i = pred_boxes_i[if_paired_i]

                paired_gt_instances = gt_instances_i[
                    track_instances.matched_gt_idxes[if_paired_i]]
                normalized_bboxes = paired_gt_instances.boxes
                cx = normalized_bboxes[..., 0:1]
                cy = normalized_bboxes[..., 1:2]
                cz = normalized_bboxes[..., 4:5]

                gt_boxes_i = torch.cat([cx, cy, cz], dim=-1)

                pred_loss_i = (0.2 * decay_ratio * self.loss_predictions(
                    paired_pred_boxes_i, gt_boxes_i).sum(dim=-1).mean())

                self.losses_dict["pred_loss_{}".format(i)] = pred_loss_i
            else:
                self.losses_dict["pred_loss_{}".format(i)] = torch.tensor(
                    [0.0]).cuda()

            decay_ratio = decay_ratio * 0.5