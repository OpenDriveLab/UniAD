#---------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
# Modified from Fiery (https://github.com/wayveai/fiery)                          #
#---------------------------------------------------------------------------------#
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mmdet.models.builder import LOSSES
from mmdet.models.losses.utils import weight_reduce_loss

@LOSSES.register_module()
class FieryBinarySegmentationLoss(nn.Module):
    def __init__(self, use_top_k=False, top_k_ratio=1.0, future_discount=1.0, loss_weight=1.0, ignore_index=255):
        super().__init__()
        self.use_top_k = use_top_k
        self.top_k_ratio = top_k_ratio
        self.future_discount = future_discount
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(self, prediction, target, frame_mask=None):
        n_gt, s, h, w = prediction.size()
        assert prediction.size() == target.size(), f"{prediction.size()}, {target.size()}"

        # Deal target > 1 (ignore_index)
        keep_mask = (target.long() != self.ignore_index).float()
        target = target * keep_mask

        loss = F.binary_cross_entropy_with_logits(
            prediction,
            target.float(),
            reduction='none',
        )
        assert loss.size() == prediction.size(), f"{loss.size()}, {prediction.size()}"
        
        # Deal ignore_index
        if self.ignore_index is not None:
            # keep_mask = (target.long() != self.ignore_index).float()
            loss = loss * keep_mask

        # Filter out losses of invalid future sample
        if frame_mask is not None:
            assert frame_mask.size(0) == s, f"{frame_mask.size()}"
            if frame_mask.sum().item() == 0:
                return prediction.sum() * 0.
            frame_mask = frame_mask.view(1, s, 1, 1)
            loss = loss * frame_mask.float()
        
        future_discounts = self.future_discount ** torch.arange(s, device=loss.device, dtype=loss.dtype)
        future_discounts = future_discounts.view(1, s, 1, 1)
        loss = loss * future_discounts

        loss = loss.view(n_gt, s, -1)
        if self.use_top_k:
            # Penalises the top-k hardest pixels
            k = int(self.top_k_ratio * loss.shape[2])
            loss, _ = torch.sort(loss, dim=2, descending=True)
            loss = loss[:, :, :k]

        return self.loss_weight * torch.mean(loss)
        
def dice_loss(pred,
              target,
              weight=None,
              eps=1e-3,
              reduction='mean',
              naive_dice=False,
              avg_factor=None,
              ignore_index=None,
              frame_mask=None):
    """Calculate dice loss, there are two forms of dice loss is supported:

        - the one proposed in `V-Net: Fully Convolutional Neural
            Networks for Volumetric Medical Image Segmentation
            <https://arxiv.org/abs/1606.04797>`_.
        - the dice loss in which the power of the number in the
            denominator is the first power instead of the second
            power.

    Args:
        pred (torch.Tensor): The prediction, has a shape (n, *)
        target (torch.Tensor): The learning label of the prediction,
            shape (n, *), same shape of pred.
        weight (torch.Tensor, optional): The weight of loss for each
            prediction, has a shape (n,). Defaults to None.
        eps (float): Avoid dividing by zero. Default: 1e-3.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        naive_dice (bool, optional): If false, use the dice
                loss defined in the V-Net paper, otherwise, use the
                naive dice loss in which the power of the number in the
                denominator is the first power instead of the second
                power.Defaults to False.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    n, s, h, w = pred.size()
    assert pred.size() == target.size(),  \
                f"{pred.size()}, {target.size()}"
    
    # Ignore invalid index(255)
    if ignore_index is not None:
        keep_mask = (target.long() != ignore_index)
        target = target * keep_mask.float()
        pred   = pred   * keep_mask.float()

    # Ignore invalid frame
    if frame_mask is not None:
        assert frame_mask.size(0) == s, f"{frame_mask.size()}"
        if frame_mask.sum().item() == 0:
            return pred.sum() * 0.
        frame_mask = frame_mask.view(1, s, 1, 1)
        target = target * frame_mask.float()
        pred   = pred   * frame_mask.float()

    input = pred.flatten(1)
    target = target.flatten(1).float()

    a = torch.sum(input * target, 1)
    if naive_dice:
        b = torch.sum(input, 1)
        c = torch.sum(target, 1)
        d = (2 * a + eps) / (b + c + eps)
    else:
        b = torch.sum(input * input, 1) + eps
        c = torch.sum(target * target, 1) + eps
        d = (2 * a) / (b + c)

    loss = 1 - d
    if weight is not None:
        assert weight.ndim == loss.ndim
        assert len(weight) == len(pred)
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss

@LOSSES.register_module()
class DiceLossWithMasks(nn.Module):
    def __init__(self,
                 use_sigmoid=True,
                 activate=True,
                 reduction='mean',
                 naive_dice=False,
                 loss_weight=1.0,
                 ignore_index=255,
                 eps=1e-3):
        """Compute dice loss.

        Args:
            use_sigmoid (bool, optional): Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
            activate (bool): Whether to activate the predictions inside,
                this will disable the inside sigmoid operation.
                Defaults to True.
            reduction (str, optional): The method used
                to reduce the loss. Options are "none",
                "mean" and "sum". Defaults to 'mean'.
            naive_dice (bool, optional): If false, use the dice
                loss defined in the V-Net paper, otherwise, use the
                naive dice loss in which the power of the number in the
                denominator is the first power instead of the second
                power. Defaults to False.
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
            eps (float): Avoid dividing by zero. Defaults to 1e-3.
        """

        super(DiceLossWithMasks, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.reduction = reduction
        self.naive_dice = naive_dice
        self.loss_weight = loss_weight
        self.eps = eps
        self.activate = activate
        self.ignore_index = ignore_index

    def forward(self,
                pred,
                target,
                weight=None,
                reduction_override=None,
                avg_factor=None,
                frame_mask=None
                ):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction, has a shape (n, *).
            target (torch.Tensor): The label of the prediction,
                shape (n, *), same shape of pred.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction, has a shape (n,). Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        """

        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        if self.activate:
            if self.use_sigmoid:
                pred = pred.sigmoid()
            else:
                raise NotImplementedError

        loss = self.loss_weight * dice_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=reduction,
            naive_dice=self.naive_dice,
            avg_factor=avg_factor,
            ignore_index=self.ignore_index,
            frame_mask=frame_mask)

        return loss