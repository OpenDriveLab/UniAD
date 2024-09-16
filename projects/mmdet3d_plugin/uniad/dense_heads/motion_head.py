#---------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
#---------------------------------------------------------------------------------#

import torch
import copy
from mmdet.models import HEADS
from mmcv.runner import force_fp32, auto_fp16
from projects.mmdet3d_plugin.models.utils.functional import (
    bivariate_gaussian_activation,
    norm_points,
    pos2posemb2d,
    anchor_coordinate_transform
)
from .motion_head_plugin.motion_utils import nonlinear_smoother
from .motion_head_plugin.base_motion_head import BaseMotionHead


@HEADS.register_module()
class MotionHead(BaseMotionHead):
    """
    MotionHead module for a neural network, which predicts motion trajectories and is used in an autonomous driving task.

    Args:
        *args: Variable length argument list.
        predict_steps (int): The number of steps to predict motion trajectories.
        transformerlayers (dict): A dictionary defining the configuration of transformer layers.
        bbox_coder: An instance of a bbox coder to be used for encoding/decoding boxes.
        num_cls_fcs (int): The number of fully-connected layers in the classification branch.
        bev_h (int): The height of the bird's-eye-view map.
        bev_w (int): The width of the bird's-eye-view map.
        embed_dims (int): The number of dimensions to use for the query and key vectors in transformer layers.
        num_anchor (int): The number of anchor points.
        det_layer_num (int): The number of layers in the transformer model.
        group_id_list (list): A list of group IDs to use for grouping the classes.
        pc_range: The range of the point cloud.
        use_nonlinear_optimizer (bool): A boolean indicating whether to use a non-linear optimizer for training.
        anchor_info_path (str): The path to the file containing the anchor information.
        vehicle_id_list(list[int]): class id of vehicle class, used for filtering out non-vehicle objects
    """
    def __init__(self,
                 *args,
                 predict_steps=12,
                 transformerlayers=None,
                 bbox_coder=None,
                 num_cls_fcs=2,
                 bev_h=30,
                 bev_w=30,
                 embed_dims=256,
                 num_anchor=6,
                 det_layer_num=6,
                 group_id_list=[],
                 pc_range=None,
                 use_nonlinear_optimizer=False,
                 anchor_info_path=None,
                 loss_traj=dict(),
                 num_classes=0,
                 vehicle_id_list=[0, 1, 2, 3, 4, 6, 7],
                 **kwargs):
        super(MotionHead, self).__init__()
        
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.num_cls_fcs = num_cls_fcs - 1
        self.num_reg_fcs = num_cls_fcs - 1
        self.embed_dims = embed_dims        
        self.num_anchor = num_anchor
        self.num_anchor_group = len(group_id_list)
        
        # we merge the classes into groups for anchor assignment
        self.cls2group = [0 for i in range(num_classes)]
        for i, grouped_ids in enumerate(group_id_list):
            for gid in grouped_ids:
                self.cls2group[gid] = i
        self.cls2group = torch.tensor(self.cls2group)
        self.pc_range = pc_range
        self.predict_steps = predict_steps
        self.vehicle_id_list = vehicle_id_list
        
        self.use_nonlinear_optimizer = use_nonlinear_optimizer
        self._load_anchors(anchor_info_path)
        self._build_loss(loss_traj)
        self._build_layers(transformerlayers, det_layer_num)
        self._init_layers()

    def forward_train(self,
                      bev_embed,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      gt_fut_traj=None,
                      gt_fut_traj_mask=None,
                      gt_sdc_fut_traj=None, 
                      gt_sdc_fut_traj_mask=None, 
                      outs_track={},
                      outs_seg={}
                  ):
        """Forward function
        Args:
            bev_embed (Tensor): BEV feature map with the shape of [B, C, H, W].
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth \
                bboxes of each sample.
            gt_labels_3d (list[torch.Tensor]): Labels of each sample.
            img_metas (list[dict]): Meta information of each sample.
            gt_fut_traj (list[torch.Tensor]): Ground truth future trajectory of each sample.
            gt_fut_traj_mask (list[torch.Tensor]): Ground truth future trajectory mask of each sample.
            gt_sdc_fut_traj (list[torch.Tensor]): Ground truth future trajectory of each sample.
            gt_sdc_fut_traj_mask (list[torch.Tensor]): Ground truth future trajectory mask of each sample.
            outs_track (dict): Outputs of track head.
            outs_seg (dict): Outputs of seg head.
            future_states (list[torch.Tensor]): Ground truth future states of each sample.
        Returns:
            dict: Losses of each branch.
        """
        track_query = outs_track['track_query_embeddings'][None, None, ...] # num_dec, B, A_track, D
        all_matched_idxes = [outs_track['track_query_matched_idxes']] #BxN
        track_boxes = outs_track['track_bbox_results']
        
        # cat sdc query/gt to the last
        sdc_match_index = torch.zeros((1,), dtype=all_matched_idxes[0].dtype, device=all_matched_idxes[0].device)
        sdc_match_index[0] = gt_fut_traj[0].shape[0]
        all_matched_idxes = [torch.cat([all_matched_idxes[0], sdc_match_index], dim=0)]
        gt_fut_traj[0] = torch.cat([gt_fut_traj[0], gt_sdc_fut_traj[0]], dim=0)
        gt_fut_traj_mask[0] = torch.cat([gt_fut_traj_mask[0], gt_sdc_fut_traj_mask[0]], dim=0)
        track_query = torch.cat([track_query, outs_track['sdc_embedding'][None, None, None, :]], dim=2)
        sdc_track_boxes = outs_track['sdc_track_bbox_results']
        track_boxes[0][0].tensor = torch.cat([track_boxes[0][0].tensor, sdc_track_boxes[0][0].tensor], dim=0)
        track_boxes[0][1] = torch.cat([track_boxes[0][1], sdc_track_boxes[0][1]], dim=0)
        track_boxes[0][2] = torch.cat([track_boxes[0][2], sdc_track_boxes[0][2]], dim=0)
        track_boxes[0][3] = torch.cat([track_boxes[0][3], sdc_track_boxes[0][3]], dim=0)
        
        memory, memory_mask, memory_pos, lane_query, _, lane_query_pos, hw_lvl = outs_seg['args_tuple']

        #-------------Motionformer运算-------------
        outs_motion = self(bev_embed, track_query, lane_query, lane_query_pos, track_boxes)

        #------------阻止计算losses_motion-----------
        # loss_inputs = [gt_bboxes_3d, gt_fut_traj, gt_fut_traj_mask, outs_motion, all_matched_idxes, track_boxes]
        # losses = self.loss(*loss_inputs)
        losses = {}

        def filter_vehicle_query(outs_motion, all_matched_idxes, gt_labels_3d, vehicle_id_list):
            query_label = gt_labels_3d[0][-1][all_matched_idxes[0]]
            # select vehicle query according to vehicle_id_list
            vehicle_mask = torch.zeros_like(query_label)
            for veh_id in vehicle_id_list:
                vehicle_mask |=  query_label == veh_id
            outs_motion['traj_query'] = outs_motion['traj_query'][:, :, vehicle_mask>0]
            outs_motion['track_query'] = outs_motion['track_query'][:, vehicle_mask>0]
            outs_motion['track_query_pos'] = outs_motion['track_query_pos'][:, vehicle_mask>0]
            all_matched_idxes[0] = all_matched_idxes[0][vehicle_mask>0]
            return outs_motion, all_matched_idxes

        all_matched_idxes[0] = all_matched_idxes[0][:-1]
        outs_motion['sdc_traj_query'] = outs_motion['traj_query'][:, :, -1]         # [3, 1, 6, 256]     [n_dec, b, n_mode, d]
        outs_motion['sdc_track_query'] = outs_motion['track_query'][:, -1]          # [1, 256]           [b, d]
        outs_motion['sdc_track_query_pos'] = outs_motion['track_query_pos'][:, -1]  # [1, 256]           [b, d]
        outs_motion['traj_query'] = outs_motion['traj_query'][:, :, :-1]            # [3, 1, 3, 6, 256]  [n_dec, b, nq, n_mode, d]
        outs_motion['track_query'] = outs_motion['track_query'][:, :-1]             # [1, 3, 256]        [b, nq, d]   
        outs_motion['track_query_pos'] = outs_motion['track_query_pos'][:, :-1]     # [1, 3, 256]        [b, nq, d]  

        
        outs_motion, all_matched_idxes = filter_vehicle_query(outs_motion, all_matched_idxes, gt_labels_3d, self.vehicle_id_list)
        outs_motion['all_matched_idxes'] = all_matched_idxes

        ret_dict = dict(losses=losses, outs_motion=outs_motion, track_boxes=track_boxes)
        return ret_dict

    def forward_test(self, bev_embed, outs_track={}, outs_seg={}):
        """Test function"""
        track_query = outs_track['track_query_embeddings'][None, None, ...]
        track_boxes = outs_track['track_bbox_results']
        
        track_query = torch.cat([track_query, outs_track['sdc_embedding'][None, None, None, :]], dim=2)
        sdc_track_boxes = outs_track['sdc_track_bbox_results']

        track_boxes[0][0].tensor = torch.cat([track_boxes[0][0].tensor, sdc_track_boxes[0][0].tensor], dim=0)
        track_boxes[0][1] = torch.cat([track_boxes[0][1], sdc_track_boxes[0][1]], dim=0)
        track_boxes[0][2] = torch.cat([track_boxes[0][2], sdc_track_boxes[0][2]], dim=0)
        track_boxes[0][3] = torch.cat([track_boxes[0][3], sdc_track_boxes[0][3]], dim=0)      
        memory, memory_mask, memory_pos, lane_query, _, lane_query_pos, hw_lvl = outs_seg['args_tuple']
        outs_motion = self(bev_embed, track_query, lane_query, lane_query_pos, track_boxes)
        traj_results = self.get_trajs(outs_motion, track_boxes)
        bboxes, scores, labels, bbox_index, mask = track_boxes[0]
        outs_motion['track_scores'] = scores[None, :]
        labels[-1] = 0
        def filter_vehicle_query(outs_motion, labels, vehicle_id_list):
            if len(labels) < 1:  # No other obj query except sdc query.
                return None

            # select vehicle query according to vehicle_id_list
            vehicle_mask = torch.zeros_like(labels)
            for veh_id in vehicle_id_list:
                vehicle_mask |=  labels == veh_id
            outs_motion['traj_query'] = outs_motion['traj_query'][:, :, vehicle_mask>0]
            outs_motion['track_query'] = outs_motion['track_query'][:, vehicle_mask>0]
            outs_motion['track_query_pos'] = outs_motion['track_query_pos'][:, vehicle_mask>0]
            outs_motion['track_scores'] = outs_motion['track_scores'][:, vehicle_mask>0]
            return outs_motion
        
        outs_motion = filter_vehicle_query(outs_motion, labels, self.vehicle_id_list)
        
        # filter sdc query
        outs_motion['sdc_traj_query'] = outs_motion['traj_query'][:, :, -1]
        outs_motion['sdc_track_query'] = outs_motion['track_query'][:, -1]
        outs_motion['sdc_track_query_pos'] = outs_motion['track_query_pos'][:, -1]
        outs_motion['traj_query'] = outs_motion['traj_query'][:, :, :-1]
        outs_motion['track_query'] = outs_motion['track_query'][:, :-1]
        outs_motion['track_query_pos'] = outs_motion['track_query_pos'][:, :-1]
        outs_motion['track_scores'] = outs_motion['track_scores'][:, :-1]

        return traj_results, outs_motion

    @auto_fp16(apply_to=('bev_embed', 'track_query', 'lane_query', 'lane_query_pos', 'lane_query_embed', 'prev_bev'))
    def forward(self, 
                bev_embed, 
                track_query, 
                lane_query, 
                lane_query_pos, 
                track_bbox_results):
        """
        Applies forward pass on the model for motion prediction using bird's eye view (BEV) embedding, track query, lane query, and track bounding box results.

        Args:
        bev_embed (torch.Tensor): A tensor of shape (h*w, B, D) representing the bird's eye view embedding.
        track_query (torch.Tensor): A tensor of shape (B, num_dec, A_track, D) representing the track query.
        lane_query (torch.Tensor): A tensor of shape (N, M_thing, D) representing the lane query.
        lane_query_pos (torch.Tensor): A tensor of shape (N, M_thing, D) representing the position of the lane query.
        track_bbox_results (List[torch.Tensor]): A list of tensors containing the tracking bounding box results for each image in the batch.

        Returns:
        dict: A dictionary containing the following keys and values:
        - 'all_traj_scores': A tensor of shape (num_levels, B, A_track, num_points) with trajectory scores for each level.
        - 'all_traj_preds': A tensor of shape (num_levels, B, A_track, num_points, num_future_steps, 2) with predicted trajectories for each level.
        - 'valid_traj_masks': A tensor of shape (B, A_track) indicating the validity of trajectory masks.
        - 'traj_query': A tensor containing intermediate states of the trajectory queries.
        - 'track_query': A tensor containing the input track queries.
        - 'track_query_pos': A tensor containing the positional embeddings of the track queries.
        """
        
        dtype = track_query.dtype
        device = track_query.device
        num_groups = self.kmeans_anchors.shape[0]

        # extract the last frame of the track query
        track_query = track_query[:, -1]
        
        # encode the center point of the track query
        reference_points_track = self._extract_tracking_centers(
            track_bbox_results, self.pc_range)
        track_query_pos = self.boxes_query_embedding_layer(pos2posemb2d(reference_points_track.to(device)))  # B, A, D
        
        # construct the learnable query positional embedding
        # split and stack according to groups
        learnable_query_pos = self.learnable_motion_query_embedding.weight.to(dtype)  # latent anchor (P*G, D)
        learnable_query_pos = torch.stack(torch.split(learnable_query_pos, self.num_anchor, dim=0))

        # construct the agent level/scene-level query positional embedding 
        # (num_groups, num_anchor, 12, 2)
        # to incorporate the information of different groups and coordinates, and embed the headding and location information
        agent_level_anchors = self.kmeans_anchors.to(dtype).to(device).view(num_groups, self.num_anchor, self.predict_steps, 2).detach()
        scene_level_ego_anchors = anchor_coordinate_transform(agent_level_anchors, track_bbox_results, with_translation_transform=True)  # B, A, G, P ,12 ,2
        scene_level_offset_anchors = anchor_coordinate_transform(agent_level_anchors, track_bbox_results, with_translation_transform=False)  # B, A, G, P ,12 ,2

        agent_level_norm = norm_points(agent_level_anchors, self.pc_range)
        scene_level_ego_norm = norm_points(scene_level_ego_anchors, self.pc_range)
        scene_level_offset_norm = norm_points(scene_level_offset_anchors, self.pc_range)

        # we only use the last point of the anchor
        agent_level_embedding = self.agent_level_embedding_layer(
            pos2posemb2d(agent_level_norm[..., -1, :]))  # G, P, D
        scene_level_ego_embedding = self.scene_level_ego_embedding_layer(
            pos2posemb2d(scene_level_ego_norm[..., -1, :]))  # B, A, G, P , D
        scene_level_offset_embedding = self.scene_level_offset_embedding_layer(
            pos2posemb2d(scene_level_offset_norm[..., -1, :]))  # B, A, G, P , D

        batch_size, num_agents = scene_level_ego_embedding.shape[:2]
        agent_level_embedding = agent_level_embedding[None,None, ...].expand(batch_size, num_agents, -1, -1, -1)
        learnable_embed = learnable_query_pos[None, None, ...].expand(batch_size, num_agents, -1, -1, -1)

        
        # save for latter, anchors
        # B, A, G, P ,12 ,2 -> B, A, P ,12 ,2
        scene_level_offset_anchors = self.group_mode_query_pos(track_bbox_results, scene_level_offset_anchors)  

        # select class embedding
        # B, A, G, P , D-> B, A, P , D
        agent_level_embedding = self.group_mode_query_pos(
            track_bbox_results, agent_level_embedding)  
        scene_level_ego_embedding = self.group_mode_query_pos(
            track_bbox_results, scene_level_ego_embedding)  # B, A, G, P , D-> B, A, P , D
        
        # B, A, G, P , D -> B, A, P , D
        scene_level_offset_embedding = self.group_mode_query_pos(
            track_bbox_results, scene_level_offset_embedding)  
        learnable_embed = self.group_mode_query_pos(
            track_bbox_results, learnable_embed)  

        init_reference = scene_level_offset_anchors.detach()

        outputs_traj_scores = []
        outputs_trajs = []
        #------------------------------Motionformer计算---------------------------
        inter_states, inter_references = self.motionformer(
            track_query,  # B, A_track, D
            lane_query,  # B, M, D
            track_query_pos=track_query_pos,
            lane_query_pos=lane_query_pos,
            track_bbox_results=track_bbox_results,
            bev_embed=bev_embed,
            reference_trajs=init_reference,
            traj_reg_branches=self.traj_reg_branches,
            traj_cls_branches=self.traj_cls_branches,
            # anchor embeddings 
            agent_level_embedding=agent_level_embedding,
            scene_level_ego_embedding=scene_level_ego_embedding,
            scene_level_offset_embedding=scene_level_offset_embedding,
            learnable_embed=learnable_embed,
            # anchor positional embeddings layers
            agent_level_embedding_layer=self.agent_level_embedding_layer,
            scene_level_ego_embedding_layer=self.scene_level_ego_embedding_layer,
            scene_level_offset_embedding_layer=self.scene_level_offset_embedding_layer,
            spatial_shapes=torch.tensor(
                [[self.bev_h, self.bev_w]], device=device),
            level_start_index=torch.tensor([0], device=device))

        for lvl in range(inter_states.shape[0]):
            outputs_class = self.traj_cls_branches[lvl](inter_states[lvl])
            tmp = self.traj_reg_branches[lvl](inter_states[lvl])
            tmp = self.unflatten_traj(tmp)
            
            # we use cumsum trick here to get the trajectory 
            tmp[..., :2] = torch.cumsum(tmp[..., :2], dim=3)

            outputs_class = self.log_softmax(outputs_class.squeeze(3))
            outputs_traj_scores.append(outputs_class)

            for bs in range(tmp.shape[0]):
                tmp[bs] = bivariate_gaussian_activation(tmp[bs])
            outputs_trajs.append(tmp)
        outputs_traj_scores = torch.stack(outputs_traj_scores)
        outputs_trajs = torch.stack(outputs_trajs)

        B, A_track, D = track_query.shape
        valid_traj_masks = track_query.new_ones((B, A_track)) > 0
        outs = {
            'all_traj_scores': outputs_traj_scores,
            'all_traj_preds': outputs_trajs,
            'valid_traj_masks': valid_traj_masks,
            'traj_query': inter_states,
            'track_query': track_query,
            'track_query_pos': track_query_pos,
        }

        return outs

    def group_mode_query_pos(self, bbox_results, mode_query_pos):
        """
        Group mode query positions based on the input bounding box results.
        
        Args:
            bbox_results (List[Tuple[torch.Tensor]]): A list of tuples containing the bounding box results for each image in the batch.
            mode_query_pos (torch.Tensor): A tensor of shape (B, A, G, P, D) representing the mode query positions.
        
        Returns:
            torch.Tensor: A tensor of shape (B, A, P, D) representing the classified mode query positions.
        """
        batch_size = len(bbox_results)
        agent_num = mode_query_pos.shape[1]
        batched_mode_query_pos = []
        self.cls2group = self.cls2group.to(mode_query_pos.device)
        # TODO: vectorize this
        # group the embeddings based on the class
        for i in range(batch_size):
            bboxes, scores, labels, bbox_index, mask = bbox_results[i]
            label = labels.to(mode_query_pos.device)
            grouped_label = self.cls2group[label]
            grouped_mode_query_pos = []
            for j in range(agent_num):
                grouped_mode_query_pos.append(
                    mode_query_pos[i, j, grouped_label[j]])
            batched_mode_query_pos.append(torch.stack(grouped_mode_query_pos))
        return torch.stack(batched_mode_query_pos)

    @force_fp32(apply_to=('preds_dicts_motion'))
    def loss(self,
             gt_bboxes_3d,
             gt_fut_traj,
             gt_fut_traj_mask,
             preds_dicts_motion,
             all_matched_idxes,
             track_bbox_results):
        """
        Computes the loss function for the given ground truth and prediction dictionaries.
        
        Args:
            gt_bboxes_3d (List[torch.Tensor]): A list of tensors representing ground truth 3D bounding boxes for each image.
            gt_fut_traj (torch.Tensor): A tensor representing the ground truth future trajectories.
            gt_fut_traj_mask (torch.Tensor): A tensor representing the ground truth future trajectory masks.
            preds_dicts_motion (Dict[str, torch.Tensor]): A dictionary containing motion-related prediction tensors.
            all_matched_idxes (List[torch.Tensor]): A list of tensors containing the matched ground truth indices for each image in the batch.
            track_bbox_results (List[Tuple[torch.Tensor]]): A list of tuples containing the tracking bounding box results for each image in the batch.

        Returns:
            dict[str, torch.Tensor]: A dictionary of loss components.
        """

        # motion related predictions
        all_traj_scores = preds_dicts_motion['all_traj_scores']
        all_traj_preds = preds_dicts_motion['all_traj_preds']

        num_dec_layers = len(all_traj_scores)

        all_gt_fut_traj = [gt_fut_traj for _ in range(num_dec_layers)]
        all_gt_fut_traj_mask = [
            gt_fut_traj_mask for _ in range(num_dec_layers)]

        losses_traj = []
        gt_fut_traj_all, gt_fut_traj_mask_all = self.compute_matched_gt_traj(
            all_gt_fut_traj[0], all_gt_fut_traj_mask[0], all_matched_idxes, track_bbox_results, gt_bboxes_3d)
        for i in range(num_dec_layers):
            loss_traj, l_class, l_reg, l_mindae, l_minfde, l_mr = self.compute_loss_traj(all_traj_scores[i], all_traj_preds[i],
                                                                                         gt_fut_traj_all, gt_fut_traj_mask_all, all_matched_idxes)
            losses_traj.append(
                (loss_traj, l_class, l_reg, l_mindae, l_minfde, l_mr))

        loss_dict = dict()
        loss_dict['loss_traj'] = losses_traj[-1][0]
        loss_dict['l_class'] = losses_traj[-1][1]
        loss_dict['l_reg'] = losses_traj[-1][2]
        loss_dict['min_ade'] = losses_traj[-1][3]
        loss_dict['min_fde'] = losses_traj[-1][4]
        loss_dict['mr'] = losses_traj[-1][5]

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_traj_i in losses_traj[:-1]:
            loss_dict[f'd{num_dec_layer}.loss_traj'] = loss_traj_i[0]
            loss_dict[f'd{num_dec_layer}.l_class'] = loss_traj_i[1]
            loss_dict[f'd{num_dec_layer}.l_reg'] = loss_traj_i[2]
            loss_dict[f'd{num_dec_layer}.min_ade'] = loss_traj_i[3]
            loss_dict[f'd{num_dec_layer}.min_fde'] = loss_traj_i[4]
            loss_dict[f'd{num_dec_layer}.mr'] = loss_traj_i[5]
            num_dec_layer += 1

        return loss_dict

    def compute_matched_gt_traj(self,
                                gt_fut_traj,
                                gt_fut_traj_mask,
                                all_matched_idxes,
                                track_bbox_results,
                                gt_bboxes_3d):
        """
        Computes the matched ground truth trajectories for a batch of images based on matched indexes.

        Args:
        gt_fut_traj (torch.Tensor): Ground truth future trajectories of shape (num_imgs, num_objects, num_future_steps, 2).
        gt_fut_traj_mask (torch.Tensor): Ground truth future trajectory masks of shape (num_imgs, num_objects, num_future_steps, 2).
        all_matched_idxes (List[torch.Tensor]): A list of tensors containing the matched indexes for each image in the batch.
        track_bbox_results (List[torch.Tensor]): A list of tensors containing the tracking bounding box results for each image in the batch.
        gt_bboxes_3d (List[torch.Tensor]): A list of tensors containing the ground truth 3D bounding boxes for each image in the batch.

        Returns:
        torch.Tensor: A concatenated tensor of the matched ground truth future trajectories.
        torch.Tensor: A concatenated tensor of the matched ground truth future trajectory masks.
        """
        num_imgs = len(all_matched_idxes)
        gt_fut_traj_all = []
        gt_fut_traj_mask_all = []
        for i in range(num_imgs):
            matched_gt_idx = all_matched_idxes[i]
            valid_traj_masks = matched_gt_idx >= 0
            matched_gt_fut_traj = gt_fut_traj[i][matched_gt_idx][valid_traj_masks]
            matched_gt_fut_traj_mask = gt_fut_traj_mask[i][matched_gt_idx][valid_traj_masks]
            if self.use_nonlinear_optimizer:
                # TODO: sdc query is not supported non-linear optimizer
                bboxes = track_bbox_results[i][0].tensor[valid_traj_masks]
                matched_gt_bboxes_3d = gt_bboxes_3d[i][-1].tensor[matched_gt_idx[:-1]
                                                                  ][valid_traj_masks[:-1]]
                sdc_gt_fut_traj = matched_gt_fut_traj[-1:]
                sdc_gt_fut_traj_mask = matched_gt_fut_traj_mask[-1:]
                matched_gt_fut_traj = matched_gt_fut_traj[:-1]
                matched_gt_fut_traj_mask = matched_gt_fut_traj_mask[:-1]
                bboxes = bboxes[:-1]
                matched_gt_fut_traj, matched_gt_fut_traj_mask = nonlinear_smoother(
                    matched_gt_bboxes_3d, matched_gt_fut_traj, matched_gt_fut_traj_mask, bboxes)
                matched_gt_fut_traj = torch.cat(
                    [matched_gt_fut_traj, sdc_gt_fut_traj], dim=0)
                matched_gt_fut_traj_mask = torch.cat(
                    [matched_gt_fut_traj_mask, sdc_gt_fut_traj_mask], dim=0)
            matched_gt_fut_traj_mask = torch.all(
                matched_gt_fut_traj_mask > 0, dim=-1)
            gt_fut_traj_all.append(matched_gt_fut_traj)
            gt_fut_traj_mask_all.append(matched_gt_fut_traj_mask)
        gt_fut_traj_all = torch.cat(gt_fut_traj_all, dim=0)
        gt_fut_traj_mask_all = torch.cat(gt_fut_traj_mask_all, dim=0)
        return gt_fut_traj_all, gt_fut_traj_mask_all

    def compute_loss_traj(self,
                          traj_scores,
                          traj_preds,
                          gt_fut_traj_all,
                          gt_fut_traj_mask_all,
                          all_matched_idxes):
        """
        Computes the trajectory loss given the predicted trajectories, ground truth trajectories, and other relevant information.
        
        Args:
            traj_scores (torch.Tensor): A tensor representing the trajectory scores.
            traj_preds (torch.Tensor): A tensor representing the predicted trajectories.
            gt_fut_traj_all (torch.Tensor): A tensor representing the ground truth future trajectories.
            gt_fut_traj_mask_all (torch.Tensor): A tensor representing the ground truth future trajectory masks.
            all_matched_idxes (List[torch.Tensor]): A list of tensors containing the matched ground truth indices for each image in the batch.
        
        Returns:
            tuple: A tuple containing the total trajectory loss, classification loss, regression loss, minimum average displacement error, minimum final displacement error, and miss rate.
        """
        num_imgs = traj_scores.size(0)
        traj_prob_all = []
        traj_preds_all = []
        for i in range(num_imgs):
            matched_gt_idx = all_matched_idxes[i]
            valid_traj_masks = matched_gt_idx >= 0
            # select valid and matched
            batch_traj_prob = traj_scores[i, valid_traj_masks, :]
            # (n_objs, n_modes, step, 5)
            batch_traj_preds = traj_preds[i, valid_traj_masks, ...]
            traj_prob_all.append(batch_traj_prob)
            traj_preds_all.append(batch_traj_preds)
        traj_prob_all = torch.cat(traj_prob_all, dim=0)
        traj_preds_all = torch.cat(traj_preds_all, dim=0)
        traj_loss, l_class, l_reg, l_minade, l_minfde, l_mr = self.loss_traj(
            traj_prob_all, traj_preds_all, gt_fut_traj_all, gt_fut_traj_mask_all)
        return traj_loss, l_class, l_reg, l_minade, l_minfde, l_mr

    @force_fp32(apply_to=('preds_dicts'))
    def get_trajs(self, preds_dicts, bbox_results):
        """
        Generates trajectories from the prediction results, bounding box results.
        
        Args:
            preds_dicts (tuple[list[dict]]): A tuple containing lists of dictionaries with prediction results.
            bbox_results (List[Tuple[torch.Tensor]]): A list of tuples containing the bounding box results for each image in the batch.
        
        Returns:
            List[dict]: A list of dictionaries containing decoded bounding boxes, scores, and labels after non-maximum suppression.
        """
        num_samples = len(bbox_results)
        num_layers = preds_dicts['all_traj_preds'].shape[0]
        ret_list = []
        for i in range(num_samples):
            preds = dict()
            for j in range(num_layers):
                subfix = '_' + str(j) if j < (num_layers - 1) else ''
                traj = preds_dicts['all_traj_preds'][j, i]
                traj_scores = preds_dicts['all_traj_scores'][j, i]

                traj_scores, traj = traj_scores.cpu(), traj.cpu()
                preds['traj' + subfix] = traj
                preds['traj_scores' + subfix] = traj_scores
            ret_list.append(preds)
        return ret_list
