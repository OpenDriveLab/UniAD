import torch
import torch
import torch.nn as nn

from mmdet.models.losses.utils import weighted_loss
from mmdet.models.builder import LOSSES

@weighted_loss
def dice_loss(input, target,mask=None,eps=0.001):
    N,H,W = input.shape
    
    input = input.contiguous().view(N, H*W)
    target = target.contiguous().view(N, H*W).float()
    if mask is not None:
      mask = mask.contiguous().view(N, H*W).float()
      input = input * mask
      target = target * mask
    a = torch.sum(input * target, 1)
    b = torch.sum(input * input, 1) + eps
    c = torch.sum(target * target, 1) + eps
    d = (2 * a) / (b + c)
    return 1 - d

@LOSSES.register_module()
class DiceLoss(nn.Module):

    def __init__(self, eps=1e-6, reduction='mean', loss_weight=1.0):
        super(DiceLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.count = 0
    def forward(self,
                pred,
                target,
                weight=None,
                mask=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):

        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        #if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n,w,h) to (n,) to match the
            # giou_loss of shape (n,)
            #assert weight.shape == pred.shape
            #weight = weight.mean((-2,-1))
        loss = self.loss_weight * dice_loss(
            pred,
            target,
            weight,
            mask=mask,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        #print('DiceLoss',loss, avg_factor)
        return loss
