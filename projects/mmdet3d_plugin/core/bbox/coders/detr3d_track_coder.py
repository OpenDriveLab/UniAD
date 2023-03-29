import torch

from mmdet.core.bbox import BaseBBoxCoder
from mmdet.core.bbox.builder import BBOX_CODERS
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox, denormalize_bbox
from mmdet3d.core import xywhr2xyxyr
from mmcv.ops import nms_bev

@BBOX_CODERS.register_module()
class DETRTrack3DCoder(BaseBBoxCoder):
    """Bbox coder for DETR3D.
    Args:
        pc_range (list[float]): Range of point cloud.
        post_center_range (list[float]): Limit of the center.
            Default: None.
        max_num (int): Max number to be kept. Default: 100.
        score_threshold (float): Threshold to filter boxes based on score.
            Default: None.
        code_size (int): Code size of bboxes. Default: 9
    """

    def __init__(self,
                 pc_range,
                 post_center_range=None,
                 max_num=100,
                 score_threshold=0.2,
                 num_classes=7,
                 with_nms=False,
                 iou_thres=0.3):
        
        self.pc_range = pc_range
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.num_classes = num_classes
        self.with_nms = with_nms
        self.nms_iou_thres = iou_thres

    def encode(self):
        pass

    def decode_single(self, cls_scores, bbox_preds, 
                      track_scores, obj_idxes, with_mask=True, img_metas=None):
        """Decode bboxes.
        Args:
            cls_scores (Tensor): Outputs from the classification head, \
                shape [num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            bbox_preds (Tensor): Outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [num_query, 9].

        Returns:
            list[dict]: Decoded boxes.
        """
        max_num = self.max_num
        max_num = min(cls_scores.size(0), self.max_num)

        cls_scores = cls_scores.sigmoid()
        _, indexs = cls_scores.max(dim=-1)
        labels = indexs % self.num_classes

        _, bbox_index = track_scores.topk(max_num)
        
        labels = labels[bbox_index]
        bbox_preds = bbox_preds[bbox_index]
        track_scores = track_scores[bbox_index]
        obj_idxes = obj_idxes[bbox_index]

        scores = track_scores
        
        final_box_preds = denormalize_bbox(bbox_preds, self.pc_range)   
        final_scores = track_scores
        final_preds = labels

        # use score threshold
        if self.score_threshold is not None:
            thresh_mask = final_scores > self.score_threshold

        if self.with_nms:
            boxes_for_nms = xywhr2xyxyr(img_metas[0]['box_type_3d'](final_box_preds[:, :], 9).bev)
            nms_mask = boxes_for_nms.new_zeros(boxes_for_nms.shape[0]) > 0
            # print(self.nms_iou_thres)
            try:
                selected = nms_bev(
                    boxes_for_nms,
                    final_scores,
                    thresh=self.nms_iou_thres)
                nms_mask[selected] = True
            except:
                print('Error', boxes_for_nms, final_scores)
                nms_mask = boxes_for_nms.new_ones(boxes_for_nms.shape[0]) > 0
        if self.post_center_range is not None:
            self.post_center_range = torch.tensor(
                self.post_center_range, device=scores.device)
            mask = (final_box_preds[..., :3] >=
                    self.post_center_range[:3]).all(1)
            mask &= (final_box_preds[..., :3] <=
                     self.post_center_range[3:]).all(1)

            if self.score_threshold:
                mask &= thresh_mask
            if not with_mask:
                mask = torch.ones_like(mask) > 0
            if self.with_nms:
                mask &= nms_mask

            boxes3d = final_box_preds[mask]
            scores = final_scores[mask]
            labels = final_preds[mask]
            track_scores = track_scores[mask]
            obj_idxes = obj_idxes[mask]
            predictions_dict = {
                'bboxes': boxes3d,
                'scores': scores,
                'labels': labels,
                'track_scores': track_scores,
                'obj_idxes': obj_idxes,
                'bbox_index': bbox_index,
                'mask': mask
            }

        else:
            raise NotImplementedError(
                'Need to reorganize output as a batch, only '
                'support post_center_range is not None for now!')
        return predictions_dict

    def decode(self, preds_dicts, with_mask=True, img_metas=None):
        """Decode bboxes.
        Args:
            cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
                Note: before sigmoid!
            bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].

        Returns:
            list[dict]: Decoded boxes.
        """
        all_cls_scores = preds_dicts['cls_scores']
        all_bbox_preds = preds_dicts['bbox_preds']
        track_scores = preds_dicts['track_scores']
        obj_idxes = preds_dicts['obj_idxes']
        
        batch_size = all_cls_scores.size()[0]
        predictions_list = []
        # bs size = 1
        predictions_list.append(self.decode_single(
            all_cls_scores, all_bbox_preds,
            track_scores, obj_idxes, with_mask, img_metas))
        #for i in range(batch_size):
        #    predictions_list.append(self.decode_single(all_cls_scores[i], all_bbox_preds[i]))
        return predictions_list
