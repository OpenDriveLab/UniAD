#----------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)   #
# Source code: https://github.com/OpenDriveLab/UniAD                               #
# Copyright (c) OpenDriveLab. All rights reserved.                                 #
# Modified from panoptic_segformer (https://github.com/zhiqi-li/Panoptic-SegFormer)#
#--------------------------------------------------------------------------------- #

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear, bias_init_with_prob, constant_init
from mmcv.runner import force_fp32, auto_fp16
from mmdet.core import multi_apply
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models.builder import HEADS, build_loss
from mmdet.core import (bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh,
                        build_assigner, build_sampler, multi_apply,
                        reduce_mean)
from mmdet.models.utils import build_transformer
from .seg_head_plugin import SegDETRHead, IOU

@HEADS.register_module()
class PansegformerHead(SegDETRHead):
    """
    Head of Panoptic SegFormer

    Code is modified from the `official github repo
    <https://github.com/open-mmlab/mmdetection>`_.

    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
    """

    def __init__(
            self,
            *args,
            bev_h,
            bev_w,
            canvas_size,
            pc_range,
            with_box_refine=False,
            as_two_stage=False,
            transformer=None,
            quality_threshold_things=0.25,
            quality_threshold_stuff=0.25,
            overlap_threshold_things=0.4,
            overlap_threshold_stuff=0.2,
            thing_transformer_head=dict(
                type='TransformerHead',  # mask decoder for things
                d_model=256,
                nhead=8,
                num_decoder_layers=6),
            stuff_transformer_head=dict(
                type='TransformerHead',  # mask decoder for stuff
                d_model=256,
                nhead=8,
                num_decoder_layers=6),
            loss_mask=dict(type='DiceLoss', weight=2.0),
            train_cfg=dict(
                assigner=dict(type='HungarianAssigner',
                              cls_cost=dict(type='ClassificationCost',
                                            weight=1.),
                              reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                              iou_cost=dict(type='IoUCost',
                                            iou_mode='giou',
                                            weight=2.0)),
                sampler=dict(type='PseudoSampler'),
            ),
            **kwargs):
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.canvas_size = canvas_size
        self.pc_range = pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]

        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        self.quality_threshold_things = 0.1
        self.quality_threshold_stuff = quality_threshold_stuff
        self.overlap_threshold_things = overlap_threshold_things
        self.overlap_threshold_stuff = overlap_threshold_stuff
        self.fp16_enabled = False

        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage
        self.num_dec_things = thing_transformer_head['num_decoder_layers']
        self.num_dec_stuff = stuff_transformer_head['num_decoder_layers']
        super(PansegformerHead, self).__init__(*args,
                                            transformer=transformer,
                                            train_cfg=train_cfg,
                                            **kwargs)
        if train_cfg:
            sampler_cfg = train_cfg['sampler_with_mask']
            self.sampler_with_mask = build_sampler(sampler_cfg, context=self)
            assigner_cfg = train_cfg['assigner_with_mask']
            self.assigner_with_mask = build_assigner(assigner_cfg)
            self.assigner_filter = build_assigner(
                dict(
                    type='HungarianAssigner_filter',
                    cls_cost=dict(type='FocalLossCost', weight=2.0),
                    reg_cost=dict(type='BBoxL1Cost',
                                  weight=5.0,
                                  box_format='xywh'),
                    iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0),
                    max_pos=
                    3  # Depends on GPU memory, setting it to 1, model can be trained on 1080Ti
                ), )

        self.loss_mask = build_loss(loss_mask)
        self.things_mask_head = build_transformer(thing_transformer_head)
        self.stuff_mask_head = build_transformer(stuff_transformer_head)
        self.count = 0

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        if not self.as_two_stage:
            self.bev_embedding = nn.Embedding(self.bev_h * self.bev_w, self.embed_dims)

        fc_cls = Linear(self.embed_dims, self.cls_out_channels)
        fc_cls_stuff = Linear(self.embed_dims, 1)
        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, 4))
        reg_branch = nn.Sequential(*reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_pred = (self.transformer.decoder.num_layers + 1) if \
            self.as_two_stage else self.transformer.decoder.num_layers

        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
        else:
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_pred)])
        if not self.as_two_stage:
            self.query_embedding = nn.Embedding(self.num_query,
                                                self.embed_dims * 2)
        self.stuff_query = nn.Embedding(self.num_stuff_classes,
                                        self.embed_dims * 2)
        self.reg_branches2 = _get_clones(reg_branch, self.num_dec_things) # used in mask decoder
        self.cls_thing_branches = _get_clones(fc_cls, self.num_dec_things) # used in mask decoder
        self.cls_stuff_branches = _get_clones(fc_cls_stuff, self.num_dec_stuff) # used in mask deocder

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m.bias, bias_init)
            for m in self.cls_thing_branches:
                nn.init.constant_(m.bias, bias_init)
            for m in self.cls_stuff_branches:
                nn.init.constant_(m.bias, bias_init)
        for m in self.reg_branches:
            constant_init(m[-1], 0, bias=0)
        for m in self.reg_branches2:
            constant_init(m[-1], 0, bias=0)
        nn.init.constant_(self.reg_branches[0][-1].bias.data[2:], -2.0)

        if self.as_two_stage:
            for m in self.reg_branches:
                nn.init.constant_(m[-1].bias.data[2:], 0.0)

    @force_fp32(apply_to=('bev_embed', ))
    def forward(self, bev_embed):
        """Forward function.

        Args:
            bev_embed (tuple[Tensor]): Features from the upstream
                network, each is a 4D-tensor with shape
                (N, C, H, W).
            img_metas (list[dict]): List of image information.

        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, h). \
                Shape [nb_dec, bs, num_query, 4].
            enc_outputs_class (Tensor): The score of each point on encode \
                feature map, has shape (N, h*w, num_class). Only when \
                as_two_stage is True it would be returned, otherwise \
                `None` would be returned.
            enc_outputs_coord (Tensor): The proposal generate from the \
                encode feature map, has shape (N, h*w, 4). Only when \
                as_two_stage is True it would be returned, otherwise \
                `None` would be returned.
        """
        _, bs, _ = bev_embed.shape
        #---对 bev_embed 进行 reshape 和 permute（调整形状），目的是将 4D 的特征图转化为 Transformer 可以接受的格式---
        mlvl_feats = [torch.reshape(bev_embed, (bs, self.bev_h, self.bev_w ,-1)).permute(0, 3, 1, 2)]
        img_masks = mlvl_feats[0].new_zeros((bs, self.bev_h, self.bev_w)) #初始化一个零掩码

        hw_lvl = [feat_lvl.shape[-2:] for feat_lvl in mlvl_feats]#保存每一层特征图的高和宽
        mlvl_masks = []
        mlvl_positional_encodings = []
        for feat in mlvl_feats:
            #--对特征图进行插值生成对应尺寸的掩码，并转为布尔类型--
            mlvl_masks.append(
                F.interpolate(img_masks[None],
                              size=feat.shape[-2:]).to(torch.bool).squeeze(0))
            #--对生成的掩码计算位置编码，用于加入位置信息--
            mlvl_positional_encodings.append(
                self.positional_encoding(mlvl_masks[-1]))

        query_embeds = None
        if not self.as_two_stage:
            query_embeds = self.query_embedding.weight
        
        #---将特征图、掩码、查询嵌入等信息传递给 Transformer 编码器-解码器模型---(Encoder + location decoder)
        (memory, memory_pos, memory_mask, query_pos), hs, init_reference, inter_references, \
        enc_outputs_class, enc_outputs_coord = self.transformer(
            mlvl_feats,
            mlvl_masks,
            query_embeds,
            mlvl_positional_encodings,
            reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
            cls_branches=self.cls_branches if self.as_two_stage else None  # noqa:E501
        )
        #--memory编码器输出的全局特征。(Encoder的输出？)
        #--hs: 解码器阶段的隐藏状态。  (Location Decoder的输出？)
        #--init_reference 和 inter_references: 初始参考点和中间参考点，用于边界框回归。
        #--enc_outputs_class 和 enc_outputs_coord: 编码器的分类和回归结果
        memory = memory.permute(1, 0, 2)
        query = hs[-1].permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        memory_pos = memory_pos.permute(1, 0, 2)

        #-------------Mask decoder---------------
        # we should feed these to mask deocder.
        args_tuple = [memory, memory_mask, memory_pos, query, None, query_pos, hw_lvl]

        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])

            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)

        outs = {
                'bev_embed': None if self.as_two_stage else bev_embed,
                'outputs_classes': outputs_classes,
                'outputs_coords': outputs_coords,
                'enc_outputs_class': enc_outputs_class if self.as_two_stage else None,
                'enc_outputs_coord': enc_outputs_coord.sigmoid() if self.as_two_stage else None,
                'args_tuple': args_tuple,
                'reference': reference,
            }
        
        return outs

    @force_fp32(apply_to=('all_cls_scores_list', 'all_bbox_preds_list',
                          'args_tuple', 'reference'))
    def loss(
        self,
        all_cls_scores,
        all_bbox_preds,
        enc_cls_scores,
        enc_bbox_preds,
        args_tuple,
        reference,
        gt_labels_list,
        gt_bboxes_list,
        gt_masks_list,
        img_metas=None,
        gt_bboxes_ignore=None,
    ):
        """"Loss function.

        Args:
            all_cls_scores (Tensor): Classification score of all
                decoder layers, has shape
                [nb_dec, bs, num_query, cls_out_channels].
            all_bbox_preds (Tensor): Sigmoid regression
                outputs of all decode layers. Each is a 4D-tensor with
                normalized coordinate format (cx, cy, w, h) and shape
                [nb_dec, bs, num_query, 4].
            enc_cls_scores (Tensor): Classification scores of
                points on encode feature map , has shape
                (N, h*w, num_classes). Only be passed when as_two_stage is
                True, otherwise is None.
            enc_bbox_preds (Tensor): Regression results of each points
                on the encode feature map, has shape (N, h*w, 4). Only be
                passed when as_two_stage is True, otherwise is None.
            args_tuple (Tuple) several args
            reference (Tensor) reference from location decoder
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        img_metas[0]['img_shape'] = (self.canvas_size[0], self.canvas_size[1], 3)

        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        ### seprate things and stuff
        gt_things_lables_list = []
        gt_things_bboxes_list = []
        gt_things_masks_list = []
        gt_stuff_labels_list = []
        gt_stuff_masks_list = []
        for i, each in enumerate(gt_labels_list):   
            # MDS: for coco, id<80 (Continuous id) is things. This is not true for other data sets
            things_selected = each < self.num_things_classes

            stuff_selected = things_selected == False

            gt_things_lables_list.append(gt_labels_list[i][things_selected])
            gt_things_bboxes_list.append(gt_bboxes_list[i][things_selected])
            gt_things_masks_list.append(gt_masks_list[i][things_selected])

            gt_stuff_labels_list.append(gt_labels_list[i][stuff_selected])
            gt_stuff_masks_list.append(gt_masks_list[i][stuff_selected])

        num_dec_layers = len(all_cls_scores)
        all_gt_bboxes_list = [
            gt_things_bboxes_list for _ in range(num_dec_layers - 1)
        ]
        all_gt_labels_list = [
            gt_things_lables_list for _ in range(num_dec_layers - 1)
        ]
        # all_gt_masks_list = [gt_masks_list for _ in range(num_dec_layers-1)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers - 1)
        ]
        img_metas_list = [img_metas for _ in range(num_dec_layers - 1)]

        # if the location decoder codntains L layers, we compute the losses of the first L-1 layers
        losses_cls, losses_bbox, losses_iou = multi_apply(
            self.loss_single, all_cls_scores[:-1], all_bbox_preds[:-1],
            all_gt_bboxes_list, all_gt_labels_list, img_metas_list,
            all_gt_bboxes_ignore_list)

        losses_cls_f, losses_bbox_f, losses_iou_f, losses_masks_things_f, losses_masks_stuff_f, loss_mask_things_list_f, loss_mask_stuff_list_f, loss_iou_list_f, loss_bbox_list_f, loss_cls_list_f, loss_cls_stuff_list_f, things_ratio, stuff_ratio = self.loss_single_panoptic(
            all_cls_scores[-1], all_bbox_preds[-1], args_tuple, reference,
            gt_things_bboxes_list, gt_things_lables_list, gt_things_masks_list,
            (gt_stuff_labels_list, gt_stuff_masks_list), img_metas,
            gt_bboxes_ignore)

        loss_dict = dict()
        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            binary_labels_list = [
                torch.zeros_like(gt_things_lables_list[i])
                for i in range(len(img_metas))
            ]
            enc_loss_cls, enc_losses_bbox, enc_losses_iou = \
                self.loss_single(enc_cls_scores, enc_bbox_preds,
                                 gt_things_bboxes_list, binary_labels_list,
                                 img_metas, gt_bboxes_ignore)
            loss_dict['enc_loss_cls'] = enc_loss_cls * things_ratio
            loss_dict['enc_loss_bbox'] = enc_losses_bbox * things_ratio
            loss_dict['enc_loss_iou'] = enc_losses_iou * things_ratio
            # loss_dict['enc_loss_mask'] = enc_losses_mask
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls_f * things_ratio
        loss_dict['loss_bbox'] = losses_bbox_f * things_ratio
        loss_dict['loss_iou'] = losses_iou_f * things_ratio
        loss_dict['loss_mask_things'] = losses_masks_things_f * things_ratio
        loss_dict['loss_mask_stuff'] = losses_masks_stuff_f * stuff_ratio
        # loss from other decoder layers
        num_dec_layer = 0
        for i in range(len(loss_mask_things_list_f)):
            loss_dict[f'd{i}.loss_mask_things_f'] = loss_mask_things_list_f[
                i] * things_ratio
            loss_dict[f'd{i}.loss_iou_f'] = loss_iou_list_f[i] * things_ratio
            loss_dict[f'd{i}.loss_bbox_f'] = loss_bbox_list_f[i] * things_ratio
            loss_dict[f'd{i}.loss_cls_f'] = loss_cls_list_f[i] * things_ratio
        for i in range(len(loss_mask_stuff_list_f)):
            loss_dict[f'd{i}.loss_mask_stuff_f'] = loss_mask_stuff_list_f[
                i] * stuff_ratio
            loss_dict[f'd{i}.loss_cls_stuff_f'] = loss_cls_stuff_list_f[
                i] * stuff_ratio
        for loss_cls_i, loss_bbox_i, loss_iou_i in zip(
                losses_cls,
                losses_bbox,
                losses_iou,
        ):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i * things_ratio
            loss_dict[
                f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i * things_ratio
            loss_dict[f'd{num_dec_layer}.loss_iou'] = loss_iou_i * things_ratio

            num_dec_layer += 1
        # print(loss_dict)
        return loss_dict

    def filter_query(self,
                     cls_scores_list,
                     bbox_preds_list,
                     gt_bboxes_list,
                     gt_labels_list,
                     img_metas,
                     gt_bboxes_ignore_list=None):
        '''
        This function aims to using the cost from the location decoder to filter out low-quality queries.
        '''
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (pos_inds_mask_list, neg_inds_mask_list, labels_list,
         label_weights_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
             self._filter_query_single, cls_scores_list, bbox_preds_list,
             gt_bboxes_list, gt_labels_list, img_metas, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))

        return pos_inds_mask_list, neg_inds_mask_list, labels_list, label_weights_list, bbox_targets_list, \
               bbox_weights_list, num_total_pos, num_total_neg, pos_inds_list, neg_inds_list

    def _filter_query_single(self,
                             cls_score,
                             bbox_pred,
                             gt_bboxes,
                             gt_labels,
                             img_meta,
                             gt_bboxes_ignore=None):
        num_bboxes = bbox_pred.size(0)
        pos_ind_mask, neg_ind_mask, assign_result = self.assigner_filter.assign(
            bbox_pred, cls_score, gt_bboxes, gt_labels, img_meta,
            gt_bboxes_ignore)
        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        # label targets
        labels = gt_bboxes.new_full((num_bboxes, ),
                                    self.num_things_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0
        img_h, img_w, _ = img_meta['img_shape']

        # DETR regress the relative position of boxes (cxcywh) in the image.
        # Thus the learning target should be normalized by the image size, also
        # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
        factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                       img_h]).unsqueeze(0)
        pos_gt_bboxes_normalized = sampling_result.pos_gt_bboxes / factor
        pos_gt_bboxes_targets = bbox_xyxy_to_cxcywh(pos_gt_bboxes_normalized)
        bbox_targets[pos_inds] = pos_gt_bboxes_targets

        return (pos_ind_mask, neg_ind_mask, labels, label_weights,
                bbox_targets, bbox_weights, pos_inds, neg_inds)

    def get_targets_with_mask(self,
                              cls_scores_list,
                              bbox_preds_list,
                              masks_preds_list_thing,
                              gt_bboxes_list,
                              gt_labels_list,
                              gt_masks_list,
                              img_metas,
                              gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            masks_preds_list_thing  (list[Tensor]):
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         mask_targets_list, mask_weights_list, pos_inds_list,
         neg_inds_list) = multi_apply(self._get_target_single_with_mask,
                                      cls_scores_list, bbox_preds_list,
                                      masks_preds_list_thing, gt_bboxes_list,
                                      gt_labels_list, gt_masks_list, img_metas,
                                      gt_bboxes_ignore_list)
        num_total_pos_thing = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg_thing = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, mask_targets_list, mask_weights_list,
                num_total_pos_thing, num_total_neg_thing, pos_inds_list)

    def _get_target_single_with_mask(self,
                                     cls_score,
                                     bbox_pred,
                                     masks_preds_things,
                                     gt_bboxes,
                                     gt_labels,
                                     gt_masks,
                                     img_meta,
                                     gt_bboxes_ignore=None):
        """
        """

        num_bboxes = bbox_pred.size(0)
        # assigner and sampler

        gt_masks = gt_masks.float()

        assign_result = self.assigner_with_mask.assign(bbox_pred, cls_score,
                                                       masks_preds_things,
                                                       gt_bboxes, gt_labels,
                                                       gt_masks, img_meta,
                                                       gt_bboxes_ignore)
        sampling_result = self.sampler_with_mask.sample(
            assign_result, bbox_pred, gt_bboxes, gt_masks)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes, ),
                                    self.num_things_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0
        img_h, img_w, _ = img_meta['img_shape']

        # DETR regress the relative position of boxes (cxcywh) in the image.
        # Thus the learning target should be normalized by the image size, also
        # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
        factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                       img_h]).unsqueeze(0)
        pos_gt_bboxes_normalized = sampling_result.pos_gt_bboxes / factor
        pos_gt_bboxes_targets = bbox_xyxy_to_cxcywh(pos_gt_bboxes_normalized)
        bbox_targets[pos_inds] = pos_gt_bboxes_targets

        mask_weights = masks_preds_things.new_zeros(num_bboxes)
        mask_weights[pos_inds] = 1.0
        pos_gt_masks = sampling_result.pos_gt_masks
        _, w, h = pos_gt_masks.shape
        mask_target = masks_preds_things.new_zeros([num_bboxes, w, h])
        mask_target[pos_inds] = pos_gt_masks

        return (labels, label_weights, bbox_targets, bbox_weights, mask_target,
                mask_weights, pos_inds, neg_inds)

    def get_filter_results_and_loss(self, cls_scores, bbox_preds,
                                    cls_scores_list, bbox_preds_list,
                                    gt_bboxes_list, gt_labels_list, img_metas,
                                    gt_bboxes_ignore_list):


        pos_inds_mask_list, neg_inds_mask_list, labels_list, label_weights_list, bbox_targets_list, \
        bbox_weights_list, num_total_pos_thing, num_total_neg_thing, pos_inds_list, neg_inds_list = self.filter_query(
            cls_scores_list, bbox_preds_list,
            gt_bboxes_list, gt_labels_list,
            img_metas, gt_bboxes_ignore_list)
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos_thing * 1.0 + \
                         num_total_neg_thing * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        loss_cls = self.loss_cls(cls_scores,
                                 labels,
                                 label_weights,
                                 avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes

        num_total_pos_thing = loss_cls.new_tensor([num_total_pos_thing])
        num_total_pos_thing = torch.clamp(reduce_mean(num_total_pos_thing),
                                          min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(img_metas, bbox_preds):
            img_h, img_w, _ = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                                               bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        
        bbox_preds = bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(bboxes,
                                 bboxes_gt,
                                 bbox_weights,
                                 avg_factor=num_total_pos_thing)

        # regression L1 loss
        loss_bbox = self.loss_bbox(bbox_preds,
                                   bbox_targets,
                                   bbox_weights,
                                   avg_factor=num_total_pos_thing)
        return loss_cls, loss_iou, loss_bbox,\
            pos_inds_mask_list, num_total_pos_thing

    def loss_single_panoptic(self,
                             cls_scores,
                             bbox_preds,
                             args_tuple,
                             reference,
                             gt_bboxes_list,
                             gt_labels_list,
                             gt_masks_list,
                             gt_panoptic_list,
                             img_metas,
                             gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.

        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            args_tuple:
            reference:
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        gt_stuff_labels_list, gt_stuff_masks_list = gt_panoptic_list
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        loss_cls, loss_iou, loss_bbox, pos_inds_mask_list, num_total_pos_thing = self.get_filter_results_and_loss(
            cls_scores, bbox_preds, cls_scores_list, bbox_preds_list, gt_bboxes_list, gt_labels_list, img_metas, gt_bboxes_ignore_list)

        memory, memory_mask, memory_pos, query, _, query_pos, hw_lvl = args_tuple

        BS, _, dim_query = query.shape[0], query.shape[1], query.shape[-1]

        len_query = max([len(pos_ind) for pos_ind in pos_inds_mask_list])
        thing_query = torch.zeros([BS, len_query, dim_query],
                                  device=query.device)

        stuff_query, stuff_query_pos = torch.split(self.stuff_query.weight,
                                                   self.embed_dims,
                                                   dim=1)
        stuff_query_pos = stuff_query_pos.unsqueeze(0).expand(BS, -1, -1)
        stuff_query = stuff_query.unsqueeze(0).expand(BS, -1, -1)

        for i in range(BS):
            thing_query[i, :len(pos_inds_mask_list[i])] = query[
                i, pos_inds_mask_list[i]]

        mask_preds_things = []
        mask_preds_stuff = []
        # mask_preds_inter = [[],[],[]]
        mask_preds_inter_things = [[] for _ in range(self.num_dec_things)]
        mask_preds_inter_stuff = [[] for _ in range(self.num_dec_stuff)]
        cls_thing_preds = [[] for _ in range(self.num_dec_things)]
        cls_stuff_preds = [[] for _ in range(self.num_dec_stuff)]
        BS, NQ, L = bbox_preds.shape
        new_bbox_preds = [
            torch.zeros([BS, len_query, L]).to(bbox_preds.device)
            for _ in range(self.num_dec_things)
        ]

        mask_things, mask_inter_things, query_inter_things = self.things_mask_head(
            memory, memory_mask, None, thing_query, None, None, hw_lvl=hw_lvl)

        mask_stuff, mask_inter_stuff, query_inter_stuff = self.stuff_mask_head(
            memory,
            memory_mask,
            None,
            stuff_query,
            None,
            stuff_query_pos,
            hw_lvl=hw_lvl)

        mask_things = mask_things.squeeze(-1)
        mask_inter_things = torch.stack(mask_inter_things, 0).squeeze(-1)

        mask_stuff = mask_stuff.squeeze(-1)
        mask_inter_stuff = torch.stack(mask_inter_stuff, 0).squeeze(-1)

        for i in range(BS):
            tmp_i = mask_things[i][:len(pos_inds_mask_list[i])].reshape(
                -1, *hw_lvl[0])
            mask_preds_things.append(tmp_i)
            pos_ind = pos_inds_mask_list[i]
            reference_i = reference[i:i + 1, pos_ind, :]

            for j in range(self.num_dec_things):
                tmp_i_j = mask_inter_things[j][i][:len(pos_inds_mask_list[i]
                                                       )].reshape(
                                                           -1, *hw_lvl[0])
                mask_preds_inter_things[j].append(tmp_i_j)

                # mask_preds_inter_things[j].append(mask_inter_things[j].reshape(-1, *hw_lvl[0]))
                query_things = query_inter_things[j]
                t1, t2, t3 = query_things.shape
                tmp = self.reg_branches2[j](query_things.reshape(t1 * t2, t3)).reshape(t1, t2, 4)
                if len(pos_ind) == 0:
                    tmp = tmp.sum(
                    ) + reference_i  # for reply bug of pytorch broadcast
                elif reference_i.shape[-1] == 4:
                    tmp += reference_i
                else:
                    assert reference_i.shape[-1] == 2
                    tmp[..., :2] += reference_i

                outputs_coord = tmp.sigmoid()

                new_bbox_preds[j][i][:len(pos_inds_mask_list[i])] = outputs_coord
                cls_thing_preds[j].append(self.cls_thing_branches[j](
                    query_things.reshape(t1 * t2, t3)))

            # stuff
            tmp_i = mask_stuff[i].reshape(-1, *hw_lvl[0])
            mask_preds_stuff.append(tmp_i)
            for j in range(self.num_dec_stuff):
                tmp_i_j = mask_inter_stuff[j][i].reshape(-1, *hw_lvl[0])
                mask_preds_inter_stuff[j].append(tmp_i_j)

                query_stuff = query_inter_stuff[j]
                s1, s2, s3 = query_stuff.shape
                cls_stuff_preds[j].append(self.cls_stuff_branches[j](
                    query_stuff.reshape(s1 * s2, s3)))

        masks_preds_list_thing = [
            mask_preds_things[i] for i in range(num_imgs)
        ]
        mask_preds_things = torch.cat(mask_preds_things, 0)
        mask_preds_inter_things = [
            torch.cat(each, 0) for each in mask_preds_inter_things
        ]
        cls_thing_preds = [torch.cat(each, 0) for each in cls_thing_preds]
        cls_stuff_preds = [torch.cat(each, 0) for each in cls_stuff_preds]
        mask_preds_stuff = torch.cat(mask_preds_stuff, 0)
        mask_preds_inter_stuff = [
            torch.cat(each, 0) for each in mask_preds_inter_stuff
        ]
        cls_scores_list = [
            cls_scores_list[i][pos_inds_mask_list[i]] for i in range(num_imgs)
        ]

        bbox_preds_list = [
            bbox_preds_list[i][pos_inds_mask_list[i]] for i in range(num_imgs)
        ]

        gt_targets = self.get_targets_with_mask(cls_scores_list,
                                                bbox_preds_list,
                                                masks_preds_list_thing,
                                                gt_bboxes_list, gt_labels_list,
                                                gt_masks_list, img_metas,
                                                gt_bboxes_ignore_list)

        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         mask_targets_list, mask_weights_list, _, _,
         pos_inds_list) = gt_targets

        thing_labels = torch.cat(labels_list, 0)
        things_weights = torch.cat(label_weights_list, 0)

        bboxes_taget = torch.cat(bbox_targets_list)
        bboxes_weights = torch.cat(bbox_weights_list)

        factors = []
        for img_meta, bbox_pred in zip(img_metas, bbox_preds_list):
            img_h, img_w, _ = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                                               bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)

        bboxes_gt = bbox_cxcywh_to_xyxy(bboxes_taget) * factors

        mask_things_gt = torch.cat(mask_targets_list, 0).to(torch.float)

        mask_weight_things = torch.cat(mask_weights_list,
                                       0).to(thing_labels.device)

        mask_stuff_gt = []
        mask_weight_stuff = []
        stuff_labels = []
        num_total_pos_stuff = 0
        for i in range(BS):
            num_total_pos_stuff += len(gt_stuff_labels_list[i])  ## all stuff

            select_stuff_index = gt_stuff_labels_list[
                i] - self.num_things_classes
            mask_weight_i_stuff = torch.zeros([self.num_stuff_classes])
            mask_weight_i_stuff[select_stuff_index] = 1
            stuff_masks = torch.zeros(
                (self.num_stuff_classes, *mask_targets_list[i].shape[-2:]),
                device=mask_targets_list[i].device).to(torch.bool)
            stuff_masks[select_stuff_index] = gt_stuff_masks_list[i].to(
                torch.bool)
            mask_stuff_gt.append(stuff_masks)
            select_stuff_index = torch.cat([
                select_stuff_index,
                torch.tensor([self.num_stuff_classes],
                             device=select_stuff_index.device)
            ])

            stuff_labels.append(1 - mask_weight_i_stuff)
            mask_weight_stuff.append(mask_weight_i_stuff)

        mask_weight_stuff = torch.cat(mask_weight_stuff,
                                      0).to(thing_labels.device)
        stuff_labels = torch.cat(stuff_labels, 0).to(thing_labels.device)
        mask_stuff_gt = torch.cat(mask_stuff_gt, 0).to(torch.float)

        num_total_pos_stuff = loss_cls.new_tensor([num_total_pos_stuff])
        num_total_pos_stuff = torch.clamp(reduce_mean(num_total_pos_stuff),
                                          min=1).item()
        if mask_preds_things.shape[0] == 0:
            loss_mask_things = (0 * mask_preds_things).sum()
        else:
            mask_preds = F.interpolate(mask_preds_things.unsqueeze(0),
                                       scale_factor=2.0,
                                       mode='bilinear').squeeze(0)
            mask_targets_things = F.interpolate(mask_things_gt.unsqueeze(0),
                                                size=mask_preds.shape[-2:],
                                                mode='bilinear').squeeze(0)
            loss_mask_things = self.loss_mask(mask_preds,
                                              mask_targets_things,
                                              mask_weight_things,
                                              avg_factor=num_total_pos_thing)
        if mask_preds_stuff.shape[0] == 0:
            loss_mask_stuff = (0 * mask_preds_stuff).sum()
        else:
            mask_preds = F.interpolate(mask_preds_stuff.unsqueeze(0),
                                       scale_factor=2.0,
                                       mode='bilinear').squeeze(0)
            mask_targets_stuff = F.interpolate(mask_stuff_gt.unsqueeze(0),
                                               size=mask_preds.shape[-2:],
                                               mode='bilinear').squeeze(0)

            loss_mask_stuff = self.loss_mask(mask_preds,
                                             mask_targets_stuff,
                                             mask_weight_stuff,
                                             avg_factor=num_total_pos_stuff)

        loss_mask_things_list = []
        loss_mask_stuff_list = []
        loss_iou_list = []
        loss_bbox_list = []
        for j in range(len(mask_preds_inter_things)):
            mask_preds_this_level = mask_preds_inter_things[j]
            if mask_preds_this_level.shape[0] == 0:
                loss_mask_j = (0 * mask_preds_this_level).sum()
            else:
                mask_preds_this_level = F.interpolate(
                    mask_preds_this_level.unsqueeze(0),
                    scale_factor=2.0,
                    mode='bilinear').squeeze(0)
                loss_mask_j = self.loss_mask(mask_preds_this_level,
                                             mask_targets_things,
                                             mask_weight_things,
                                             avg_factor=num_total_pos_thing)
            loss_mask_things_list.append(loss_mask_j)
            bbox_preds_this_level = new_bbox_preds[j].reshape(-1, 4)
            bboxes_this_level = bbox_cxcywh_to_xyxy(
                bbox_preds_this_level) * factors
            # We let this loss be 0. We didn't predict bbox in our mask decoder. Predicting bbox in the mask decoder is basically useless
            loss_iou_j = self.loss_iou(bboxes_this_level,
                                       bboxes_gt,
                                       bboxes_weights,
                                       avg_factor=num_total_pos_thing) * 0
            if bboxes_taget.shape[0] != 0:
                loss_bbox_j = self.loss_bbox(
                    bbox_preds_this_level,
                    bboxes_taget,
                    bboxes_weights,
                    avg_factor=num_total_pos_thing) * 0
            else:
                loss_bbox_j = bbox_preds_this_level.sum() * 0
            loss_iou_list.append(loss_iou_j)
            loss_bbox_list.append(loss_bbox_j)
        for j in range(len(mask_preds_inter_stuff)):
            mask_preds_this_level = mask_preds_inter_stuff[j]
            if mask_preds_this_level.shape[0] == 0:
                loss_mask_j = (0 * mask_preds_this_level).sum()
            else:
                mask_preds_this_level = F.interpolate(
                    mask_preds_this_level.unsqueeze(0),
                    scale_factor=2.0,
                    mode='bilinear').squeeze(0)
                loss_mask_j = self.loss_mask(mask_preds_this_level,
                                             mask_targets_stuff,
                                             mask_weight_stuff,
                                             avg_factor=num_total_pos_stuff)
            loss_mask_stuff_list.append(loss_mask_j)

        loss_cls_thing_list = []
        loss_cls_stuff_list = []
        thing_labels = thing_labels.reshape(-1)
        for j in range(len(mask_preds_inter_things)):
            # We let this loss be 0. When using "query-filter", only partial thing queries are feed to the mask decoder. This will cause imbalance when supervising these queries.
            cls_scores = cls_thing_preds[j]

            if cls_scores.shape[0] == 0:
                loss_cls_thing_j = cls_scores.sum() * 0
            else:
                loss_cls_thing_j = self.loss_cls(
                    cls_scores,
                    thing_labels,
                    things_weights,
                    avg_factor=num_total_pos_thing) * 2 * 0
            loss_cls_thing_list.append(loss_cls_thing_j)

        for j in range(len(mask_preds_inter_stuff)):
            cls_scores = cls_stuff_preds[j]
            if cls_scores.shape[0] == 0:
                loss_cls_stuff_j = cls_stuff_preds[j].sum() * 0
            else:
                loss_cls_stuff_j = self.loss_cls(
                    cls_stuff_preds[j],
                    stuff_labels.to(torch.long),
                    avg_factor=num_total_pos_stuff) * 2
            loss_cls_stuff_list.append(loss_cls_stuff_j)

        ## dynamic adjusting the weights
        things_ratio, stuff_ratio = num_total_pos_thing / (
            num_total_pos_stuff + num_total_pos_thing), num_total_pos_stuff / (
                num_total_pos_stuff + num_total_pos_thing)

        return loss_cls, loss_bbox, loss_iou, loss_mask_things, loss_mask_stuff, loss_mask_things_list, loss_mask_stuff_list, loss_iou_list, loss_bbox_list, loss_cls_thing_list, loss_cls_stuff_list, things_ratio, stuff_ratio
    
    def forward_test(self,
                    pts_feats=None,
                    gt_lane_labels=None,
                    gt_lane_masks=None,
                    img_metas=None,
                    rescale=False):
        bbox_list = [dict() for i in range(len(img_metas))]

        pred_seg_dict = self(pts_feats)
        results = self.get_bboxes(pred_seg_dict['outputs_classes'],
                                           pred_seg_dict['outputs_coords'],
                                           pred_seg_dict['enc_outputs_class'],
                                           pred_seg_dict['enc_outputs_coord'],
                                           pred_seg_dict['args_tuple'],
                                           pred_seg_dict['reference'],
                                           img_metas,
                                           rescale=rescale)

        with torch.no_grad():
            drivable_pred = results[0]['drivable']
            drivable_gt = gt_lane_masks[0][0, -1]
            drivable_iou, drivable_intersection, drivable_union = IOU(drivable_pred.view(1, -1), drivable_gt.view(1, -1))

            lane_pred = results[0]['lane']
            lanes_pred = (results[0]['lane'].sum(0) > 0).int()
            lanes_gt = (gt_lane_masks[0][0][:-1].sum(0) > 0).int()
            lanes_iou, lanes_intersection, lanes_union = IOU(lanes_pred.view(1, -1), lanes_gt.view(1, -1))

            divider_gt = (gt_lane_masks[0][0][gt_lane_labels[0][0] == 0].sum(0) > 0).int()
            crossing_gt = (gt_lane_masks[0][0][gt_lane_labels[0][0] == 1].sum(0) > 0).int()
            contour_gt = (gt_lane_masks[0][0][gt_lane_labels[0][0] == 2].sum(0) > 0).int()
            divider_iou, divider_intersection, divider_union = IOU(lane_pred[0].view(1, -1), divider_gt.view(1, -1))
            crossing_iou, crossing_intersection, crossing_union = IOU(lane_pred[1].view(1, -1), crossing_gt.view(1, -1))
            contour_iou, contour_intersection, contour_union = IOU(lane_pred[2].view(1, -1), contour_gt.view(1, -1))


            ret_iou = {'drivable_intersection': drivable_intersection,
                       'drivable_union': drivable_union,
                       'lanes_intersection': lanes_intersection,
                       'lanes_union': lanes_union,
                       'divider_intersection': divider_intersection,
                       'divider_union': divider_union,
                       'crossing_intersection': crossing_intersection,
                       'crossing_union': crossing_union,
                       'contour_intersection': contour_intersection,
                       'contour_union': contour_union,
                       'drivable_iou': drivable_iou,
                       'lanes_iou': lanes_iou,
                       'divider_iou': divider_iou,
                       'crossing_iou': crossing_iou,
                       'contour_iou': contour_iou}
        for result_dict, pts_bbox in zip(bbox_list, results):
            result_dict['pts_bbox'] = pts_bbox
            result_dict['ret_iou'] = ret_iou
            result_dict['args_tuple'] = pred_seg_dict['args_tuple']
        return bbox_list


    @auto_fp16(apply_to=("bev_feat", "prev_bev"))
    def forward_train(self,
                          bev_feat=None,
                          img_metas=None,
                          gt_lane_labels=None,
                          gt_lane_bboxes=None,
                          gt_lane_masks=None,
                         ):
        """
        Forward pass of the segmentation model during training.

        Args:
            bev_feat (torch.Tensor): Bird's eye view feature maps. Shape [batch_size, channels, height, width].
            img_metas (list[dict]): List of image meta information dictionaries.
            gt_lane_labels (list[torch.Tensor]): Ground-truth lane class labels. Shape [batch_size, num_lanes, max_lanes].
            gt_lane_bboxes (list[torch.Tensor]): Ground-truth lane bounding boxes. Shape [batch_size, num_lanes, 4].
            gt_lane_masks (list[torch.Tensor]): Ground-truth lane masks. Shape [batch_size, num_lanes, height, width].
            prev_bev (torch.Tensor): Previous bird's eye view feature map. Shape [batch_size, channels, height, width].

        Returns:
            tuple:
                - losses_seg (torch.Tensor): Total segmentation loss.
                - pred_seg_dict (dict): Dictionary of predicted segmentation outputs.
        """
        #---------开始Mapformer/Panoptic,输入只需要BEVfeature-------
        pred_seg_dict = self(bev_feat) 
        loss_inputs = [
            pred_seg_dict['outputs_classes'],
            pred_seg_dict['outputs_coords'],
            pred_seg_dict['enc_outputs_class'],
            pred_seg_dict['enc_outputs_coord'],
            pred_seg_dict['args_tuple'],
            pred_seg_dict['reference'],
            gt_lane_labels,
            gt_lane_bboxes,
            gt_lane_masks
        ]
        #--------阻止计算losses_seg-------
        # losses_seg = self.loss(*loss_inputs, img_metas=img_metas)
        losses_seg = {}
        return losses_seg, pred_seg_dict

    def _get_bboxes_single(self,
                           cls_score,
                           bbox_pred,
                           img_shape,
                           scale_factor,
                           rescale=False):
        """
        """
        assert len(cls_score) == len(bbox_pred)
        max_per_img = self.test_cfg.get('max_per_img', self.num_query)

        # exclude background
        if self.loss_cls.use_sigmoid:
            cls_score = cls_score.sigmoid()
            scores, indexes = cls_score.view(-1).topk(max_per_img)
            det_labels = indexes % self.num_things_classes
            bbox_index = indexes // self.num_things_classes
            bbox_pred = bbox_pred[bbox_index]
        else:
            scores, det_labels = F.softmax(cls_score, dim=-1)[..., :-1].max(-1)
            scores, bbox_index = scores.topk(max_per_img)
            bbox_pred = bbox_pred[bbox_index]
            det_labels = det_labels[bbox_index]

        det_bboxes = bbox_cxcywh_to_xyxy(bbox_pred)
        det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
        det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
        det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])
        if rescale:
            det_bboxes /= det_bboxes.new_tensor(scale_factor)
        det_bboxes = torch.cat((det_bboxes, scores.unsqueeze(1)), -1)

        return bbox_index, det_bboxes, det_labels

    @force_fp32(apply_to=('all_cls_scores_list', 'all_bbox_preds_list',
                          'args_tuple'))
    def get_bboxes(
        self,
        all_cls_scores,
        all_bbox_preds,
        enc_cls_scores,
        enc_bbox_preds,
        args_tuple,
        reference,
        img_metas,
        rescale=False,
    ):
        """
        """
        cls_scores = all_cls_scores[-1]
        bbox_preds = all_bbox_preds[-1]
        memory, memory_mask, memory_pos, query, _, query_pos, hw_lvl = args_tuple

        seg_list = []
        stuff_score_list = []
        panoptic_list = []
        bbox_list = []
        labels_list = []
        drivable_list = []
        lane_list = []
        lane_score_list = []
        score_list = []
        for img_id in range(len(img_metas)):
            cls_score = cls_scores[img_id]
            bbox_pred = bbox_preds[img_id]
            # img_shape = img_metas[img_id]['img_shape']
            # ori_shape = img_metas[img_id]['ori_shape']
            # scale_factor = img_metas[img_id]['scale_factor']
            img_shape = (self.canvas_size[0], self.canvas_size[1], 3)
            ori_shape = (self.canvas_size[0], self.canvas_size[1], 3)
            scale_factor = 1

            index, bbox, labels = self._get_bboxes_single(
                cls_score, bbox_pred, img_shape, scale_factor, rescale)

            i = img_id
            thing_query = query[i:i + 1, index, :]
            thing_query_pos = query_pos[i:i + 1, index, :]
            joint_query = torch.cat([
                thing_query, self.stuff_query.weight[None, :, :self.embed_dims]
            ], 1)

            stuff_query_pos = self.stuff_query.weight[None, :,
                                                      self.embed_dims:]

            mask_things, mask_inter_things, query_inter_things = self.things_mask_head(
                memory[i:i + 1],
                memory_mask[i:i + 1],
                None,
                joint_query[:, :-self.num_stuff_classes],
                None,
                None,
                hw_lvl=hw_lvl)
            mask_stuff, mask_inter_stuff, query_inter_stuff = self.stuff_mask_head(
                memory[i:i + 1],
                memory_mask[i:i + 1],
                None,
                joint_query[:, -self.num_stuff_classes:],
                None,
                stuff_query_pos,
                hw_lvl=hw_lvl)

            attn_map = torch.cat([mask_things, mask_stuff], 1)
            attn_map = attn_map.squeeze(-1)  # BS, NQ, N_head,LEN

            stuff_query = query_inter_stuff[-1]
            scores_stuff = self.cls_stuff_branches[-1](
                stuff_query).sigmoid().reshape(-1)

            mask_pred = attn_map.reshape(-1, *hw_lvl[0])

            mask_pred = F.interpolate(mask_pred.unsqueeze(0),
                                      size=ori_shape[:2],
                                      mode='bilinear').squeeze(0)

            masks_all = mask_pred
            score_list.append(masks_all)
            drivable_list.append(masks_all[-1] > 0.5)
            masks_all = masks_all[:-self.num_stuff_classes]
            seg_all = masks_all > 0.5
            sum_seg_all = seg_all.sum((1, 2)).float() + 1
            # scores_all = torch.cat([bbox[:, -1], scores_stuff], 0)
            # bboxes_all = torch.cat([bbox, torch.zeros([self.num_stuff_classes, 5], device=labels.device)], 0)
            # labels_all = torch.cat([labels, torch.arange(self.num_things_classes, self.num_things_classes+self.num_stuff_classes).to(labels.device)], 0)
            scores_all = bbox[:, -1]
            bboxes_all = bbox
            labels_all = labels

            ## mask wise merging
            seg_scores = (masks_all * seg_all.float()).sum(
                (1, 2)) / sum_seg_all
            scores_all *= (seg_scores**2)

            scores_all, index = torch.sort(scores_all, descending=True)

            masks_all = masks_all[index]
            labels_all = labels_all[index]
            bboxes_all = bboxes_all[index]
            seg_all = seg_all[index]

            bboxes_all[:, -1] = scores_all

            # MDS: select things for instance segmeantion
            things_selected = labels_all < self.num_things_classes
            stuff_selected = labels_all >= self.num_things_classes
            bbox_th = bboxes_all[things_selected][:100]
            labels_th = labels_all[things_selected][:100]
            seg_th = seg_all[things_selected][:100]
            labels_st = labels_all[stuff_selected]
            scores_st = scores_all[stuff_selected]
            masks_st = masks_all[stuff_selected]
            
            stuff_score_list.append(scores_st)

            results = torch.zeros((2, *mask_pred.shape[-2:]),
                                  device=mask_pred.device).to(torch.long)
            id_unique = 1
            lane = torch.zeros((self.num_things_classes, *mask_pred.shape[-2:]), device=mask_pred.device).to(torch.long)
            lane_score =  torch.zeros((self.num_things_classes, *mask_pred.shape[-2:]), device=mask_pred.device).to(mask_pred.dtype)
            for i, scores in enumerate(scores_all):
                # MDS: things and sutff have different threholds may perform a little bit better
                if labels_all[i] < self.num_things_classes and scores < self.quality_threshold_things:
                    continue
                elif labels_all[i] >= self.num_things_classes and scores < self.quality_threshold_stuff:
                    continue
                _mask = masks_all[i] > 0.5
                mask_area = _mask.sum().item()
                intersect = _mask & (results[0] > 0)
                intersect_area = intersect.sum().item()
                if labels_all[i] < self.num_things_classes:
                    if mask_area == 0 or (intersect_area * 1.0 / mask_area
                                          ) > self.overlap_threshold_things:
                        continue
                else:
                    if mask_area == 0 or (intersect_area * 1.0 / mask_area
                                          ) > self.overlap_threshold_stuff:
                        continue
                if intersect_area > 0:
                    _mask = _mask & (results[0] == 0)
                results[0, _mask] = labels_all[i]
                if labels_all[i] < self.num_things_classes:
                    lane[labels_all[i], _mask] = 1
                    lane_score[labels_all[i], _mask] = masks_all[i][_mask]
                    results[1, _mask] = id_unique
                    id_unique += 1

            file_name = img_metas[img_id]['pts_filename'].split('/')[-1].split('.')[0]
            panoptic_list.append(
                (results.permute(1, 2, 0).cpu().numpy(), file_name, ori_shape))

            bbox_list.append(bbox_th)
            labels_list.append(labels_th)
            seg_list.append(seg_th)
            lane_list.append(lane)
            lane_score_list.append(lane_score)
        results = []
        for i in range(len(img_metas)):
            results.append({
                'bbox': bbox_list[i],
                'segm': seg_list[i],
                'labels': labels_list[i],
                'panoptic': panoptic_list[i],
                'drivable': drivable_list[i],
                'score_list': score_list[i],
                'lane': lane_list[i],
                'lane_score': lane_score_list[i],
                'stuff_score_list' : stuff_score_list[i],
            })
        return results
