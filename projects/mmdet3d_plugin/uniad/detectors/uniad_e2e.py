#---------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
#---------------------------------------------------------------------------------#

import torch
from mmcv.runner import auto_fp16
from mmdet.models import DETECTORS
import copy
import os
from ..dense_heads.seg_head_plugin import IOU
from .uniad_track import UniADTrack
from mmdet.models.builder import build_head

@DETECTORS.register_module()
class UniAD(UniADTrack):
    """
    UniAD: Unifying Detection, Tracking, Segmentation, Motion Forecasting, Occupancy Prediction and Planning for Autonomous Driving
    """
    def __init__(
        self,
        seg_head=None,
        motion_head=None,
        occ_head=None,
        planning_head=None,
        freeze_uniad=None,
        IMU_head=None,
        task_loss_weight=dict(
            track=1.0,
            map=1.0,
            motion=1.0,
            occ=1.0,
            planning=1.0,
            IMU_predict=1.0,
        ),
        **kwargs,  #捕获多余的关键字参数，用于传递给父类 UniADTrack 的构造函数
    ):
        #-----这个就是调用父类UniADTrack的__init__，并把**kwargs传进去-----
        super(UniAD, self).__init__(**kwargs)

        #-----初始化任务头-----
        if seg_head:
            self.seg_head = build_head(seg_head)
            if freeze_uniad:
                self.freeze_module(self.seg_head)

        if occ_head:
            self.occ_head = build_head(occ_head)
            if freeze_uniad:
                self.freeze_module(self.occ_head)

        if motion_head:
            self.motion_head = build_head(motion_head)
            if freeze_uniad:
                self.freeze_module(self.motion_head)

        if planning_head:
            self.planning_head = build_head(planning_head)
            if freeze_uniad:
                self.freeze_module(self.planning_head)
        
        #-----------------------------
        if IMU_head:
            self.IMU_head = build_head(IMU_head)


        self.task_loss_weight = task_loss_weight
        assert set(task_loss_weight.keys()) == \
               {'track', 'occ', 'motion', 'map', 'planning', 'IMU_predict'}

    #--------对于传入的模块执行参数frozen操作----------
    def freeze_module(self, module):
        """Freeze the parameters of a given module."""
        for param in module.parameters():
            param.requires_grad = False
        module.eval()  # Set the module to evaluation mode (especially useful for BN layers)

    #---@property 装饰器将 with_planning_head 定义为一个只读属性，用户可以像访问普通属性一样调用它---
    #---属性检查模型中是否存在 planning_head---
    @property
    def with_planning_head(self):
        return hasattr(self, 'planning_head') and self.planning_head is not None
    
    @property
    def with_occ_head(self):
        return hasattr(self, 'occ_head') and self.occ_head is not None

    @property
    def with_motion_head(self):
        return hasattr(self, 'motion_head') and self.motion_head is not None

    @property
    def with_seg_head(self):
        return hasattr(self, 'seg_head') and self.seg_head is not None

    #--------------------------
    @property
    def with_IMU_head(self):
        return hasattr(self, 'IMU_head') and self.IMU_head is not None
   

    def forward_dummy(self, img):
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)
        

    # Add the subtask loss to the whole model loss
    @auto_fp16(apply_to=('img', 'points'))
    def forward_train(self,
                      img=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_inds=None,
                      l2g_t=None,
                      l2g_r_mat=None,
                      timestamp=None,
                      gt_lane_labels=None,
                      gt_lane_bboxes=None,
                      gt_lane_masks=None,
                      gt_fut_traj=None,
                      gt_fut_traj_mask=None,
                      gt_past_traj=None,
                      gt_past_traj_mask=None,
                      gt_sdc_bbox=None,
                      gt_sdc_label=None,
                      gt_sdc_fut_traj=None,
                      gt_sdc_fut_traj_mask=None,
                      
                      # Occ_gt
                      gt_segmentation=None,
                      gt_instance=None, 
                      gt_occ_img_is_valid=None,
                      
                      #planning
                      sdc_planning=None,       #"SDC" 可能代表 "Self-Driving Car"
                      sdc_planning_mask=None,
                      command=None,
                      
                      # fut gt for planning
                      gt_future_boxes=None,

                      #data for IMU predict
                      current_frame_e2g_r = None,
                      previous_frame_e2g_r = None,
                      gt_future_frame_e2g_r = None,
                      **kwargs,  # [1, 9]
                      ):
        """Forward training function for the model that includes multiple tasks, such as tracking, segmentation, motion prediction, occupancy prediction, and planning.

            Args:
            img (torch.Tensor, optional): Tensor containing images of each sample with shape (N, C, H, W). Defaults to None.
            img_metas (list[dict], optional): List of dictionaries containing meta information for each sample. Defaults to None.
            gt_bboxes_3d (list[:obj:BaseInstance3DBoxes], optional): List of ground truth 3D bounding boxes for each sample. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): List of tensors containing ground truth labels for 3D bounding boxes. Defaults to None.
            gt_inds (list[torch.Tensor], optional): List of tensors containing indices of ground truth objects. Defaults to None.
            l2g_t (list[torch.Tensor], optional): List of tensors containing translation vectors from local to global coordinates. Defaults to None.
            l2g_r_mat (list[torch.Tensor], optional): List of tensors containing rotation matrices from local to global coordinates. Defaults to None.
            timestamp (list[float], optional): List of timestamps for each sample. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): List of tensors containing ground truth 2D bounding boxes in images to be ignored. Defaults to None.
            gt_lane_labels (list[torch.Tensor], optional): List of tensors containing ground truth lane labels. Defaults to None.
            gt_lane_bboxes (list[torch.Tensor], optional): List of tensors containing ground truth lane bounding boxes. Defaults to None.
            gt_lane_masks (list[torch.Tensor], optional): List of tensors containing ground truth lane masks. Defaults to None.
            gt_fut_traj (list[torch.Tensor], optional): List of tensors containing ground truth future trajectories. Defaults to None.
            gt_fut_traj_mask (list[torch.Tensor], optional): List of tensors containing ground truth future trajectory masks. Defaults to None.
            gt_past_traj (list[torch.Tensor], optional): List of tensors containing ground truth past trajectories. Defaults to None.
            gt_past_traj_mask (list[torch.Tensor], optional): List of tensors containing ground truth past trajectory masks. Defaults to None.
            gt_sdc_bbox (list[torch.Tensor], optional): List of tensors containing ground truth self-driving car bounding boxes. Defaults to None.
            gt_sdc_label (list[torch.Tensor], optional): List of tensors containing ground truth self-driving car labels. Defaults to None.
            gt_sdc_fut_traj (list[torch.Tensor], optional): List of tensors containing ground truth self-driving car future trajectories. Defaults to None.
            gt_sdc_fut_traj_mask (list[torch.Tensor], optional): List of tensors containing ground truth self-driving car future trajectory masks. Defaults to None.
            gt_segmentation (list[torch.Tensor], optional): List of tensors containing ground truth segmentation masks. Defaults to
            gt_instance (list[torch.Tensor], optional): List of tensors containing ground truth instance segmentation masks. Defaults to None.
            gt_occ_img_is_valid (list[torch.Tensor], optional): List of tensors containing binary flags indicating whether an image is valid for occupancy prediction. Defaults to None.
            sdc_planning (list[torch.Tensor], optional): List of tensors containing self-driving car planning information. Defaults to None.
            sdc_planning_mask (list[torch.Tensor], optional): List of tensors containing self-driving car planning masks. Defaults to None.
            command (list[torch.Tensor], optional): List of tensors containing high-level command information for planning. Defaults to None.
            gt_future_boxes (list[torch.Tensor], optional): List of tensors containing ground truth future bounding boxes for planning. Defaults to None.
            gt_future_labels (list[torch.Tensor], optional): List of tensors containing ground truth future labels for planning. Defaults to None.
            
            Returns:
                dict: Dictionary containing losses of different tasks, such as tracking, segmentation, motion prediction, occupancy prediction, and planning. Each key in the dictionary 
                    is prefixed with the corresponding task name, e.g., 'track', 'map', 'motion', 'occ', and 'planning'. The values are the calculated losses for each task.
        """
        #------------去掉IMU数据中间嵌套的那层列表-------------
        current_frame_e2g_r = current_frame_e2g_r[0]
        previous_frame_e2g_r = previous_frame_e2g_r[0]
        gt_future_frame_e2g_r = gt_future_frame_e2g_r[0]

        losses = dict()
        len_queue = img.size(1)
        
        #-----开始track模块的前向传播计算:包含img_backbone/neck,BEV_encoder,Trackformer-----
        losses_track, outs_track = self.forward_track_train(img, gt_bboxes_3d, gt_labels_3d, gt_past_traj, gt_past_traj_mask, gt_inds, gt_sdc_bbox, gt_sdc_label,
                                                        l2g_t, l2g_r_mat, img_metas, timestamp)
        #-----为损失字典中的每个损失项添加前缀，并根据任务的损失权重对其进行加权-----
        # losses_track = self.loss_weighted_and_prefixed(losses_track, prefix='track')
        losses.update(losses_track)
        
        # Upsample bev for tiny version（如果使用的是bevformer_tiny版本）
        outs_track = self.upsample_bev_if_tiny(outs_track)
        bev_embed = outs_track["bev_embed"]
        bev_pos  = outs_track["bev_pos"]

        #---提取最新帧的图像元数据---
        img_metas = [each[len_queue-1] for each in img_metas]

        #-----开始segmentation的前向传播计算，即Mapformer-----
        outs_seg = dict()
        if self.with_seg_head:          
            losses_seg, outs_seg = self.seg_head.forward_train(bev_embed, img_metas,
                                                          gt_lane_labels, gt_lane_bboxes, gt_lane_masks)         
            # losses_seg = self.loss_weighted_and_prefixed(losses_seg, prefix='map')
            losses.update(losses_seg)

        #-----开始路径预测模块的前向传播计算，即Motionformer-----
        outs_motion = dict()
        # Forward Motion Head
        if self.with_motion_head:
            ret_dict_motion = self.motion_head.forward_train(bev_embed,
                                                        gt_bboxes_3d, gt_labels_3d, 
                                                        gt_fut_traj, gt_fut_traj_mask, 
                                                        gt_sdc_fut_traj, gt_sdc_fut_traj_mask, 
                                                        outs_track=outs_track, outs_seg=outs_seg
                                                    )
            losses_motion = ret_dict_motion["losses"]
            outs_motion = ret_dict_motion["outs_motion"]
            outs_motion['bev_pos'] = bev_pos
            # losses_motion = self.loss_weighted_and_prefixed(losses_motion, prefix='motion')
            losses.update(losses_motion)

        #-----开始占用网络预测模块的前向传播计算，即Occformer-----
        # Forward Occ Head
        if self.with_occ_head:
            if outs_motion['track_query'].shape[1] == 0:
                # TODO: rm hard code
                outs_motion['track_query'] = torch.zeros((1, 1, 256)).to(bev_embed)
                outs_motion['track_query_pos'] = torch.zeros((1,1, 256)).to(bev_embed)
                outs_motion['traj_query'] = torch.zeros((3, 1, 1, 6, 256)).to(bev_embed)
                outs_motion['all_matched_idxes'] = [[-1]]
            losses_occ = self.occ_head.forward_train(
                            bev_embed, 
                            outs_motion, 
                            gt_inds_list=gt_inds,
                            gt_segmentation=gt_segmentation,
                            gt_instance=gt_instance,
                            gt_img_is_valid=gt_occ_img_is_valid,
                        )
            # losses_occ = self.loss_weighted_and_prefixed(losses_occ, prefix='occ')
            losses.update(losses_occ)
        
        #-----开始Plan预测模块的前向传播计算，即Planner-----
        # Forward Plan Head
        if self.with_planning_head:
            outs_planning = self.planning_head.forward_train(bev_embed, outs_motion, sdc_planning, sdc_planning_mask, command, gt_future_boxes)
            losses_planning = outs_planning['losses']
            # losses_planning = self.loss_weighted_and_prefixed(losses_planning, prefix='planning')
            losses.update(losses_planning)
        
        #-----开始IMU预测部分的训练, 即IMU_head/IMUformer------
        if self.with_IMU_head:
            outs_IMU = self.IMU_head.forward_train(bev_embed, outs_planning['outs_motion']['sdc_traj'], current_frame_e2g_r, previous_frame_e2g_r, gt_future_frame_e2g_r)
            losses_IMU = outs_IMU['losses']
            losses_IMU = self.loss_weighted_and_prefixed(losses_IMU, prefix='IMU_predict')
            losses.update(losses_IMU)

        #----处理损失函数值中可能出现的NaN值，将其替换为0，以确保进一步的计算不会受到NaN值的影响----
        for k,v in losses.items():
            losses[k] = torch.nan_to_num(v)
        return losses
    
    #-----------为损失字典中的每个损失项添加前缀，并根据任务的损失权重对其进行加权---------
    def loss_weighted_and_prefixed(self, loss_dict, prefix=''):
        loss_factor = self.task_loss_weight[prefix] #获取损失权重
        loss_dict = {f"{prefix}.{k}" : v*loss_factor for k, v in loss_dict.items()}
        return loss_dict

    def forward_test(self,
                     img=None,
                     img_metas=None,
                     l2g_t=None,
                     l2g_r_mat=None,
                     timestamp=None,
                     gt_lane_labels=None,
                     gt_lane_masks=None,
                     rescale=False,
                     # planning gt(for evaluation only)
                     sdc_planning=None,
                     sdc_planning_mask=None,
                     command=None,
 
                     # Occ_gt (for evaluation only)
                     gt_segmentation=None,
                     gt_instance=None, 
                     gt_occ_img_is_valid=None,
                     **kwargs
                    ):
        """Test function
        """
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img

        if img_metas[0][0]['scene_token'] != self.prev_frame_info['scene_token']:
            # the first sample of each scene is truncated
            self.prev_frame_info['prev_bev'] = None
        # update idx
        self.prev_frame_info['scene_token'] = img_metas[0][0]['scene_token']

        # do not use temporal information
        if not self.video_test_mode:
            self.prev_frame_info['prev_bev'] = None

        # Get the delta of ego position and angle between two timestamps.
        tmp_pos = copy.deepcopy(img_metas[0][0]['can_bus'][:3])
        tmp_angle = copy.deepcopy(img_metas[0][0]['can_bus'][-1])
        # first frame
        if self.prev_frame_info['scene_token'] is None:
            img_metas[0][0]['can_bus'][:3] = 0
            img_metas[0][0]['can_bus'][-1] = 0
        # following frames
        else:
            img_metas[0][0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
            img_metas[0][0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
        self.prev_frame_info['prev_pos'] = tmp_pos
        self.prev_frame_info['prev_angle'] = tmp_angle

        img = img[0]
        img_metas = img_metas[0]
        timestamp = timestamp[0] if timestamp is not None else None

        result = [dict() for i in range(len(img_metas))]
        result_track = self.simple_test_track(img, l2g_t, l2g_r_mat, img_metas, timestamp)

        # Upsample bev for tiny model        
        result_track[0] = self.upsample_bev_if_tiny(result_track[0])
        
        bev_embed = result_track[0]["bev_embed"]

        if self.with_seg_head:
            result_seg =  self.seg_head.forward_test(bev_embed, gt_lane_labels, gt_lane_masks, img_metas, rescale)

        if self.with_motion_head:
            result_motion, outs_motion = self.motion_head.forward_test(bev_embed, outs_track=result_track[0], outs_seg=result_seg[0])
            outs_motion['bev_pos'] = result_track[0]['bev_pos']

        outs_occ = dict()
        if self.with_occ_head:
            occ_no_query = outs_motion['track_query'].shape[1] == 0
            outs_occ = self.occ_head.forward_test(
                bev_embed, 
                outs_motion,
                no_query = occ_no_query,
                gt_segmentation=gt_segmentation,
                gt_instance=gt_instance,
                gt_img_is_valid=gt_occ_img_is_valid,
            )
            result[0]['occ'] = outs_occ
        
        if self.with_planning_head:
            planning_gt=dict(
                segmentation=gt_segmentation,
                sdc_planning=sdc_planning,
                sdc_planning_mask=sdc_planning_mask,
                command=command
            )
            result_planning = self.planning_head.forward_test(bev_embed, outs_motion, outs_occ, command)
            result[0]['planning'] = dict(
                planning_gt=planning_gt,
                result_planning=result_planning,
            )

        pop_track_list = ['prev_bev', 'bev_pos', 'bev_embed', 'track_query_embeddings', 'sdc_embedding']
        result_track[0] = pop_elem_in_result(result_track[0], pop_track_list)

        if self.with_seg_head:
            result_seg[0] = pop_elem_in_result(result_seg[0], pop_list=['pts_bbox', 'args_tuple'])
        if self.with_motion_head:
            result_motion[0] = pop_elem_in_result(result_motion[0])
        if self.with_occ_head:
            result[0]['occ'] = pop_elem_in_result(result[0]['occ'],  \
                pop_list=['seg_out_mask', 'flow_out', 'future_states_occ', 'pred_ins_masks', 'pred_raw_occ', 'pred_ins_logits', 'pred_ins_sigmoid'])
        
        for i, res in enumerate(result):
            res['token'] = img_metas[i]['sample_idx']
            res.update(result_track[i])
            if self.with_motion_head:
                res.update(result_motion[i])
            if self.with_seg_head:
                res.update(result_seg[i])

        return result


def pop_elem_in_result(task_result:dict, pop_list:list=None):
    all_keys = list(task_result.keys())
    for k in all_keys:
        if k.endswith('query') or k.endswith('query_pos') or k.endswith('embedding'):
            task_result.pop(k)
    
    if pop_list is not None:
        for pop_k in pop_list:
            task_result.pop(pop_k, None)
    return task_result
