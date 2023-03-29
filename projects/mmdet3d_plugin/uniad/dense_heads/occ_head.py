#---------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
#---------------------------------------------------------------------------------#

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import HEADS, build_loss
from mmcv.runner import BaseModule
from einops import rearrange
from mmdet.core import reduce_mean
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
import copy
from .occ_head_plugin import MLP, BevFeatureSlicer, SimpleConv2d, CVT_Decoder, Bottleneck, UpsamplingAdd, \
                             e2e_predict_instance_segmentation_and_trajectories

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

@HEADS.register_module()
class OccHead(BaseModule):
    def __init__(self, 
                 # General
                 receptive_field=3,
                 n_future=4,
                 spatial_extent=(50, 50),
                 ignore_index=255,

                 # BEV
                 grid_conf = None,

                 bev_input_size=(200, 200),
                 bev_size=(200, 200),
                 bev_emb_dim=256,
                 bev_proj_dim=64,
                 bev_proj_nlayers=1,

                 # Query
                 query_dim=256,
                 query_hidden_dim=256,
                 query_mlp_layers=3,
                 mode_fuser_version=1,
                 detach_query_pos=True,
                 temporal_mlp_layer=2,

                 # Transformer

                 transformer_decoder=None,

                 with_mask_attn=False,
                 attn_mask_thresh=0.5,
                 resue_temporal_embed_for_mask_attn=True,
                 
                 # Loss
                 sample_ignore_mode='all_valid',
                 aux_loss_weight=1.,

                 loss_mask=None,
                 loss_dice=None,

                 # Cfgs
                 train_cfg=None,
                 test_cfg=None,
                 vis_cfg=None,  # Used in testing
                 init_cfg=None,
                 with_conv_init=False,

                 # Eval
                 pan_eval=False,
                 test_seg_thresh:float=0.5,
                 test_with_track_score=False,
                 pred_ins_score_thres=0.55, # vpq
                 pred_seg_score_thres=0.3, # iou
                 ins_mask_alpha=2.0, # vpq
                 seg_mask_alpha=1.0, # iou
                 ):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
            'behavior, init_cfg is not allowed to be set'
        super().__init__(init_cfg)
        self.receptive_field = receptive_field  # NOTE: Used by prepare_future_labels in E2EPredTransformer
        self.n_future = n_future
        self.spatial_extent = spatial_extent
        self.ignore_index  = ignore_index

        bevformer_bev_conf = {
            'xbound': [-51.2, 51.2, 0.512],
            'ybound': [-51.2, 51.2, 0.512],
            'zbound': [-10.0, 10.0, 20.0],
        }
        self.bev_sampler =  BevFeatureSlicer(bevformer_bev_conf, grid_conf)
        
        self.bev_input_size = bev_input_size
        self.bev_size = bev_size
        assert self.bev_input_size == self.bev_size

        # Proj bev_emb to smaller channels
        self.bev_proj_dim = bev_proj_dim

        if bev_proj_nlayers == 0:
            self.bev_light_proj = nn.Sequential()
        else:
            self.bev_light_proj = SimpleConv2d(
                in_channels=bev_emb_dim,
                conv_channels=bev_emb_dim,
                out_channels=self.bev_proj_dim,
                num_conv=bev_proj_nlayers,
            )

        # Downscale bev_feat -> /4
        self.base_downscale = nn.Sequential(
            Bottleneck(in_channels=self.bev_proj_dim, downsample=True),
            Bottleneck(in_channels=self.bev_proj_dim, downsample=True)
        )

        # Future blocks with transformer
        self.n_future_blocks = self.n_future + 1

        # - transformer
        self.with_mask_attn = with_mask_attn
        self.attn_mask_thresh = attn_mask_thresh
        
        self.resue_temporal_embed_for_mask_attn = resue_temporal_embed_for_mask_attn
        
        # assert transformer_decoder.num_layers == self.n_future_blocks
        self.num_trans_layers = transformer_decoder.num_layers
        assert self.num_trans_layers % self.n_future_blocks == 0

        self.num_heads = transformer_decoder.transformerlayers.\
            attn_cfgs.num_heads
        self.transformer_decoder = build_transformer_layer_sequence(
            transformer_decoder)

        # - temporal-mlps
        query_out_dim = self.bev_proj_dim
        assert query_dim == query_hidden_dim == query_out_dim == self.bev_proj_dim

        temporal_mlp = MLP(query_dim, query_hidden_dim, query_out_dim, num_layers=temporal_mlp_layer)
        self.temporal_mlps = _get_clones(temporal_mlp, self.n_future_blocks)
            
        # - downscale-convs
        downscale_conv = Bottleneck(in_channels=self.bev_proj_dim, downsample=True)
        self.downscale_convs = _get_clones(downscale_conv, self.n_future_blocks)
        

        # - upsampleAdds
        upsample_add = UpsamplingAdd(in_channels=self.bev_proj_dim, out_channels=self.bev_proj_dim)
        self.upsample_adds = _get_clones(upsample_add, self.n_future_blocks)

        # Decoder
        self.dense_decoder = CVT_Decoder(
            dim=self.bev_proj_dim,
            blocks=[self.bev_proj_dim, self.bev_proj_dim],
        )

        # Query

        # * ------------------------------------------------------------------- *
        # * This part be will cleaned later.
        self.mode_fuser_version = mode_fuser_version
        if mode_fuser_version == 1:
            self.mode_fuser = nn.Sequential(
                    nn.Linear(query_dim, query_out_dim),
                    nn.LayerNorm(query_out_dim),
                    nn.ReLU(inplace=True)
                )
        elif mode_fuser_version == 2:
            # predict a softmax distribution, then sum up
            self.mode_fuser = nn.Sequential(
                    nn.Linear(query_dim, query_out_dim),
                    # nn.LayerNorm(query_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(query_out_dim, 1)  # predicted softmax
                )
        elif mode_fuser_version == 3:
            motion_n_dec = 3
            motion_n_mode = 6
            self.simple_fuser = nn.Sequential(
                nn.Linear(query_dim, query_out_dim),
                nn.LayerNorm(query_out_dim),
                nn.ReLU(inplace=True),
                nn.Linear(query_out_dim, query_out_dim)
            )
            self.query_weights = nn.Linear(
                query_dim, motion_n_dec * motion_n_mode
            )
        # * ------------------------------------------------------------------- *

        self.multi_query_fuser =  nn.Sequential(
                nn.Linear(query_dim * 3, query_dim * 2),
                nn.LayerNorm(query_dim * 2),
                nn.ReLU(inplace=True),
                nn.Linear(query_dim * 2, query_out_dim),
            )

        self.sdc_index = -9
        self.detach_query_pos = detach_query_pos

        self.query_to_occ_feat = MLP(
            query_dim, query_hidden_dim, query_out_dim, num_layers=query_mlp_layers
        )

        if self.with_mask_attn:
            self.temporal_mlp_for_mask = copy.deepcopy(self.query_to_occ_feat)
        
        # Loss
        # For matching
        self.sample_ignore_mode = sample_ignore_mode
        assert self.sample_ignore_mode in ['all_valid', 'past_valid', 'none']

        self.aux_loss_weight = aux_loss_weight

        self.loss_dice = build_loss(loss_dice)

        self.loss_mask = build_loss(loss_mask)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # Vis in test
        self.vis_cfg = vis_cfg
        
        self.test_ind = 0   # Recommended to use only one gpu for correct visualize
        self.pan_eval = pan_eval
        self.test_seg_thresh  = test_seg_thresh

        self.test_with_track_score = test_with_track_score
        
        self.pred_ins_score_thres = pred_ins_score_thres
        self.pred_seg_score_thres = pred_seg_score_thres
        self.ins_mask_alpha = ins_mask_alpha
        self.seg_mask_alpha = seg_mask_alpha
        
        self.with_conv_init = with_conv_init
        self.init_weights()

    def init_weights(self):
        for p in self.transformer_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)
        if self.with_conv_init:
            self.init_cfg = dict(type='Kaiming', layer='Conv2d')

    def get_attn_mask(self, state, ins_query):
        """
        Compute the attention mask for the given state and instance query.

        This function calculates the attention mask for a given state and instance query,
        and then upsamples the mask prediction to match the desired output size.

        Args:
            state (torch.Tensor): A tensor representing the input state. Shape: (b, c, h, w).
            ins_query (torch.Tensor): A tensor representing the instance query. Shape: (b, q, c).

        Returns:
            tuple: A tuple containing the following tensors:
                - attn_mask (torch.Tensor): The computed attention mask. Shape: (b, num_heads, (h * w), q).
                - upsampled_mask_pred (torch.Tensor): The upsampled mask prediction. Shape: (b, q, h, w).
                - ins_embed (torch.Tensor): The instance embeddings. Shape: (b, q, c).
        """
        assert self.with_mask_attn is True
        # state: b, c, h, w
        # ins_query: b, q, c
        ins_embed = self.temporal_mlp_for_mask(
            ins_query 
        )
        mask_pred = torch.einsum("bqc,bchw->bqhw", ins_embed, state)
        attn_mask = mask_pred.sigmoid() < self.attn_mask_thresh
        attn_mask = rearrange(attn_mask, 'b q h w -> b (h w) q').unsqueeze(1).repeat(
            1, self.num_heads, 1, 1).flatten(0, 1)
        attn_mask = attn_mask.detach()
        
        # if a mask is all True(all background), then set it all False.
        attn_mask[torch.where(
            attn_mask.sum(-1) == attn_mask.shape[-1])] = False

        upsampled_mask_pred = F.interpolate(
            mask_pred,
            self.bev_size,
            mode='bilinear',
            align_corners=False
        )  # Supervised by gt

        return attn_mask, upsampled_mask_pred, ins_embed

    # * Move this into forward
    def get_future_states(self, x, ins_query):
        """
        Computes future states using a sequence of transformer layers, downscaling, and attention mechanisms.

        Args:
            x (Tensor): Input tensor with shape (h*w, b, d) where h and w are height and width, b is the batch size, and d is the feature dimension.
            ins_query (Tensor): Instance query tensor.

        Returns:
            dict: A dictionary containing 'future_states', 'temporal_query', 'mask_preds', and 'temporal_embed_for_mask_attn' tensors.
        """
        base_state = rearrange(x, '(h w) b d -> b d h w', h=self.bev_size[0])

        base_state = self.bev_sampler(base_state)
        base_state = self.bev_light_proj(base_state)
        base_state = self.base_downscale(base_state)
        base_ins_query = ins_query

        last_state = base_state
        last_ins_query = base_ins_query
        future_states = []
        mask_preds = []
        temporal_query = []
        temporal_embed_for_mask_attn = []
        n_trans_layer_each_block = self.num_trans_layers // self.n_future_blocks
        assert n_trans_layer_each_block >= 1
        
        for i in range(self.n_future_blocks):
            # Downscale
            cur_state = self.downscale_convs[i](last_state)  # /4 -> /8

            # Attention
            # temporal_aware ins_query
            cur_ins_query = self.temporal_mlps[i](last_ins_query)  # [b, q, d]
            temporal_query.append(cur_ins_query)

            # [Optional] generate attn mask 
            if self.with_mask_attn:
                attn_mask, mask_pred, cur_ins_emb_for_mask_attn = self.get_attn_mask(cur_state, cur_ins_query)
                attn_masks = [None, attn_mask] 

                mask_preds.append(mask_pred)  # /1
                temporal_embed_for_mask_attn.append(cur_ins_emb_for_mask_attn)
            else:
                attn_masks = [None, None]

            cur_state = rearrange(cur_state, 'b c h w -> (h w) b c')
            cur_ins_query = rearrange(cur_ins_query, 'b q c -> q b c')
            

            for j in range(n_trans_layer_each_block):
                trans_layer_ind = i * n_trans_layer_each_block + j
                trans_layer = self.transformer_decoder.layers[trans_layer_ind]
                cur_state = trans_layer(
                    query=cur_state,  # [h'*w', b, c]
                    key=cur_ins_query,  # [nq, b, c]
                    value=cur_ins_query,  # [nq, b, c]
                    query_pos=None,  
                    key_pos=None,
                    attn_masks=attn_masks,
                    query_key_padding_mask=None,
                    key_padding_mask=None
                )  # out size: [h'*w', b, c]

            cur_state = rearrange(cur_state, '(h w) b c -> b c h w', h=self.bev_size[0]//8)
            
            # Upscale to /4
            cur_state = self.upsample_adds[i](cur_state, last_state)

            # Out
            future_states.append(cur_state)  # [b, d, h/4, w/4]
            last_state = cur_state

        first_stage = dict()
        future_states = torch.stack(future_states, dim=1)  # [b, t, d, h/4, w/4]
        first_stage['future_states'] = future_states

        temporal_query = torch.stack(temporal_query, dim=1)  # [b, t, q, d]
        first_stage['temporal_query'] = temporal_query
    
        if self.with_mask_attn:
            first_stage['mask_preds'] = torch.stack(mask_preds, dim=2)  # [b, q, t, h, w]
            first_stage['temporal_embed_for_mask_attn'] = torch.stack(temporal_embed_for_mask_attn, dim=1)  # [b, t, q, d]

        return first_stage

    def forward(self, x, ins_query):
        output = {}
        first_stage_output = self.get_future_states(x, ins_query)
        
        if 'mask_preds' in first_stage_output:
            assert self.with_mask_attn
            output['mask_preds'] = first_stage_output['mask_preds']

        future_states = first_stage_output['future_states']
        temporal_embed_for_mask_attn = first_stage_output.get('temporal_embed_for_mask_attn', None)

        # Decode future states to larger resolution
        future_states = self.dense_decoder(future_states)['decoded_feature']


        assert ins_query.size(0) == 1, f"{ins_query.size()}"
        
        ins_query = temporal_embed_for_mask_attn

        ins_occ_query = self.query_to_occ_feat(ins_query)    # [b, q, query_out_dim] or [b, t, q, query_out_dim]
        
        # Generate final outputs
        assert ins_occ_query.dim() == 4
        ins_occ_masks = torch.einsum("btqc,btchw->bqthw", ins_occ_query, future_states)
        
        output['pred_ins_masks'] = ins_occ_masks # [b, q, t, h, w]
        output['pred_raw_occ'] = ins_occ_masks.sigmoid().max(1)[0] # [b, t, h, w], per-pixel probs
        output['future_states_occ'] = future_states
        return output

    def merge_queries(self, outs_dict, detach_query_pos=True):
        """
        Merges instance and track queries, and applies mode fusion based on the mode_fuser_version attribute.

        Args:
            outs_dict (dict): A dictionary containing 'traj_query', 'track_query', and 'track_query_pos' keys.
            detach_query_pos (bool, optional): If True, detaches the track_query_pos tensor from the graph. Default is True.

        Returns:
            Tensor: A merged and mode-fused query tensor.
        """
        ins_query = outs_dict.get('traj_query', None)       # [n_dec, b, nq, n_modes, dim]
        track_query = outs_dict['track_query']              # [b, nq, d]
        track_query_pos = outs_dict['track_query_pos']      # [b, nq, d]

        if detach_query_pos:
            track_query_pos = track_query_pos.detach()

        if self.mode_fuser_version == 1:
            ins_query = ins_query[-1]
            ins_query = self.mode_fuser(ins_query).max(2)[0]
        elif self.mode_fuser_version == 2:
            ins_query = ins_query[-1]
            query_weight = self.mode_fuser(ins_query)
            query_weight = query_weight.softmax(-2)
            ins_query = (ins_query * query_weight).sum(2)
        elif self.mode_fuser_version == 3:
            ins_query = self.simple_fuser(ins_query)
            query_weight = self.query_weights(track_query).softmax(-1).unsqueeze(-1) 
            ins_query = rearrange(ins_query, 'd b q m c -> b q (d m) c')
            ins_query = (ins_query * query_weight).sum(2)
            assert ins_query.size() == track_query.size(), f"{ins_query.size()}, {track_query.size()}"
        ins_query = self.multi_query_fuser(torch.cat([ins_query, track_query, track_query_pos], dim=-1))
        
        return ins_query

    # With matched queries [a small part of all queries] and matched_gt results
    def forward_train(
                    self,
                    bev_feat,
                    occ_gt,
                    outs_dict,
                    gt_inds_list=None,
                ):
        # Generate warpped gt and related inputs
        warpped_labels = self.get_occflow_labels(occ_gt)
        
        all_matched_gt_ids = outs_dict['all_matched_idxes']  # list of tensor, length bs

        ins_query = self.merge_queries(outs_dict, self.detach_query_pos)

        # Forward the occ-flow model
        pred_dict = self(bev_feat, ins_query=ins_query)
        
        # Get pred and gt
        ins_seg_preds_batch    = pred_dict['pred_ins_masks']    # [b, q, t, h, w]
        ins_seg_targets_batch  = warpped_labels['instance'] # [1, 5, 200, 200] [b, t, h, w] # ins targets of a batch

        mask_preds_batch = pred_dict.get('mask_preds', None)  # [b, q, t, h, w]  # Supervise attention mask of mask2former
        
        # img_valid flag, for filtering out invalid samples in sequence when calculating loss
        img_is_valid = warpped_labels['img_is_valid']  # [1, 7]
        assert img_is_valid.size(1) == self.receptive_field + self.n_future,  \
                f"Img_is_valid can only be 7 as for loss calculation and evaluation!!! Don't change it"
        frame_valid_mask = img_is_valid.bool()
        past_valid_mask  = frame_valid_mask[:, :self.receptive_field]
        future_frame_mask = frame_valid_mask[:, (self.receptive_field-1):]  # [1, 5]  including current frame

        # only supervise when all 3 past frames are valid
        past_valid = past_valid_mask.all(dim=1)
        future_frame_mask[~past_valid] = False
        
        # Calculate loss in the batch
        loss_dict = dict()
        loss_dice = ins_seg_preds_batch.new_zeros(1)[0].float()
        loss_mask = ins_seg_preds_batch.new_zeros(1)[0].float()
        if self.with_mask_attn:
            loss_aux_dice = ins_seg_preds_batch.new_zeros(1)[0].float()
            loss_aux_mask = ins_seg_preds_batch.new_zeros(1)[0].float()

        bs = ins_query.size(0)
        assert bs == 1
        for ind in range(bs):
            # Each gt_bboxes contains 3 frames, we only use the last one
            cur_gt_inds   = gt_inds_list[ind][-1]

            cur_matched_gt = all_matched_gt_ids[ind]  # [n_gt] or [n_gt, 2]
            cur_pos_inds = None  # already ordered (track/motion)
            
            if torch.is_tensor(cur_matched_gt) and cur_matched_gt.dim() == 2:  # [n_gt, 2] (det)
                cur_pos_inds = cur_matched_gt[:, 0]    # [n_gt], pos_inds from all queries
                cur_matched_gt = cur_matched_gt[:, 1]  # [n_gt]
            
            # Re-order gt according to matched_gt_inds
            cur_gt_inds   = cur_gt_inds[cur_matched_gt]
            
            # Deal matched_gt: -1, its actually background(unmatched)
            cur_gt_inds[cur_matched_gt == -1] = -1  # Bugfixed
            cur_gt_inds[cur_matched_gt == -2] = -2  

            frame_mask = future_frame_mask[ind]  # [t]

            # Prediction
            ins_seg_preds = ins_seg_preds_batch[ind]   # [q(n_gt for matched), t, h, w]
            ins_seg_targets = ins_seg_targets_batch[ind]  # [t, h, w]
            if self.with_mask_attn:
                mask_preds = mask_preds_batch[ind]
            
            if cur_pos_inds is not None:
                # From det, need to order the predictions with assigned-gt
                ins_seg_preds = ins_seg_preds[cur_pos_inds]
                if self.with_mask_attn:
                    mask_preds = mask_preds[cur_pos_inds]

            # Assigned-gt
            ins_seg_targets_ordered = []
            for ins_id in cur_gt_inds:
                # -9 for sdc query
                # -1 for unmatched query
                # If ins_seg_targets is all 255, ignore (directly append occ-and-flow gt to list)
                # 255 for special object --> change to -20 (same as in occflow_label.py)
                # -2 for no_query situation
                if (ins_seg_targets == self.ignore_index).all().item() is True:
                    ins_tgt = ins_seg_targets.long()
                elif ins_id.item() in [-1, -2] :  # false positive query (unmatched)
                    ins_tgt = torch.ones_like(ins_seg_targets).long() * self.ignore_index
                else:
                    SPECIAL_INDEX = -20
                    if ins_id.item() == self.ignore_index:
                        ins_id = torch.ones_like(ins_id) * SPECIAL_INDEX
                    ins_tgt = (ins_seg_targets == ins_id).long()  # [t, h, w], 0 or 1
                
                ins_seg_targets_ordered.append(ins_tgt)
            
            ins_seg_targets_ordered = torch.stack(ins_seg_targets_ordered, dim=0)  # [n_gt, t, h, w]
            
            # Sanity check
            t, h, w = ins_seg_preds.shape[-3:]
            assert t == 1+self.n_future, f"{ins_seg_preds.size()}"
            assert ins_seg_preds.size() == ins_seg_targets_ordered.size(),   \
                            f"{ins_seg_preds.size()}, {ins_seg_targets_ordered.size()}"
            
            num_total_pos = ins_seg_preds.size(0)  # Check this line 

            # loss for a sample in batch
            num_total_pos = ins_seg_preds.new_tensor([num_total_pos])
            num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()
            
            cur_dice_loss = self.loss_dice(
                ins_seg_preds, ins_seg_targets_ordered, avg_factor=num_total_pos, frame_mask=frame_mask)

            cur_mask_loss = self.loss_mask(
                ins_seg_preds, ins_seg_targets_ordered, frame_mask=frame_mask
            )

            if self.with_mask_attn:
                cur_aux_dice_loss = self.loss_dice(
                    mask_preds, ins_seg_targets_ordered, avg_factor=num_total_pos, frame_mask=frame_mask
                )
                cur_aux_mask_loss = self.loss_mask(
                    mask_preds, ins_seg_targets_ordered, frame_mask=frame_mask
                )

                # aux_loss_weight
                cur_aux_dice_loss = self.aux_loss_weight * cur_aux_dice_loss
                cur_aux_mask_loss = self.aux_loss_weight * cur_aux_mask_loss


            loss_dice += cur_dice_loss
            loss_mask += cur_mask_loss
            if self.with_mask_attn:
                loss_aux_dice += cur_aux_dice_loss
                loss_aux_mask += cur_aux_mask_loss

        loss_dict['loss_dice'] = loss_dice / bs
        loss_dict['loss_mask'] = loss_mask / bs
        
        if self.with_mask_attn:
            loss_dict['loss_aux_dice'] = loss_aux_dice / bs
            loss_dict['loss_aux_mask'] = loss_aux_mask / bs

        return loss_dict, pred_dict

    def forward_test(
                    self,
                    bev_feat,
                    occ_gt,
                    outs_dict,
                    no_query=False,
                ):
        warpped_labels = self.get_occflow_labels(occ_gt)

        out_dict = dict()
        out_dict['seg_gt']  = warpped_labels['segmentation'][:, :1+self.n_future]  # [1, 5, 1, 200, 200]
        out_dict['ins_seg_gt'] = self.get_ins_seg_gt(warpped_labels['instance'][:, :1+self.n_future])  # [1, 5, 200, 200]
        if no_query:
            # output all zero results
            out_dict['seg_out'] = torch.zeros_like(out_dict['seg_gt']).long()  # [1, 5, 1, 200, 200]
            out_dict['ins_seg_out'] = torch.zeros_like(out_dict['ins_seg_gt']).long()  # [1, 5, 200, 200]

            # fake query for future_states
            fake_query = torch.zeros((1, 1, 256)).to(bev_feat)
            fake_future_states = self(bev_feat, ins_query=fake_query)
            out_dict['future_states_occ'] = fake_future_states['future_states_occ']
            out_dict['pred_raw_occ']  = torch.zeros_like(out_dict['ins_seg_gt']).float()  # [1, 5, 200, 200]

            return out_dict



        ins_query = self.merge_queries(outs_dict, self.detach_query_pos)

        # track_scores = outs_dict['track_scores'].to(ins_query.device)
        track_scores = outs_dict.get('track_scores', None)
        if track_scores is not None:
            track_scores = track_scores.to(ins_query)

        pred_dict = self(bev_feat, ins_query=ins_query)

        out_dict['future_states_occ'] = pred_dict['future_states_occ']
        out_dict['pred_ins_masks'] = pred_dict['pred_ins_masks']
        out_dict['pred_raw_occ']   = pred_dict['pred_raw_occ']

        pred_ins_masks = pred_dict['pred_ins_masks'][:,:,:1+self.n_future]  # [b, q, t, h, w]

        pred_ins_masks_sigmoid = pred_ins_masks.sigmoid()  # [b, q, t, h, w]

        # ----------------- Consider remove this completely -----------------
        if self.test_with_track_score:
            assert track_scores is not None, "Must use track score"
            # For ins_seg
            pred_ins_score_mask = track_scores[0] > self.pred_ins_score_thres
            track_score_weighted = track_scores[:, pred_ins_score_mask, None, None, None] ** self.ins_mask_alpha
            pred_ins_masks = pred_ins_masks_sigmoid[:, pred_ins_score_mask, ...] * track_score_weighted           # [b, q, t, h, w]
            # For sem_seg
            pred_seg_score_mask = track_scores[0] > self.pred_seg_score_thres
            track_score_weighted = track_scores[:, pred_seg_score_mask, None, None, None] ** self.seg_mask_alpha
            
            pred_seg_masks = (pred_ins_masks_sigmoid[:, pred_seg_score_mask, ...] * track_score_weighted).max(1)[0]   # [b, t, h, w]
        else:
            pred_ins_masks = pred_ins_masks_sigmoid
            pred_seg_masks = pred_ins_masks.max(1)[0]

        test_seg_thresh = self.test_seg_thresh
        out_dict['topk_query_ins_segs'] = pred_ins_masks
        out_dict['seg_out'] = (pred_seg_masks > test_seg_thresh).long().unsqueeze(2)  # [b, t, 1, h, w]
        # -------------------------------------------------------------------

        if self.test_with_track_score:
            out_dict['seg_out_mask'] = (pred_ins_masks_sigmoid.max(1)[0] > test_seg_thresh).long().unsqueeze(2)
        else:
            out_dict['seg_out_mask'] = out_dict['seg_out']

        if self.pan_eval:
            # ins_pred
            pred_consistent_instance_seg =  \
                e2e_predict_instance_segmentation_and_trajectories(out_dict)  # bg is 0, fg starts with 1, consecutive
            
            out_dict['ins_seg_out'] = pred_consistent_instance_seg  # [1, 5, 200, 200]

        return out_dict

    def get_ins_seg_gt(self, warpped_instance_label):
        """
        Convert a non-consecutive instance segmentation ground truth label to a consecutive one.

        Args:
            warpped_instance_label (Tensor): A non-consecutive instance segmentation ground truth label.

        Returns:
            Tensor: A consecutive instance segmentation ground truth label.
        """
        ins_gt_old = warpped_instance_label  # Not consecutive, 0 for bg, otherwise ins_ind(start from 1)
        ins_gt_new = torch.zeros_like(ins_gt_old).to(ins_gt_old)  # Make it consecutive
        ins_inds_unique = torch.unique(ins_gt_old)
        new_id = 1
        for uni_id in ins_inds_unique:
            if uni_id.item() in [0, self.ignore_index]:  # ignore background_id
                continue
            ins_gt_new[ins_gt_old == uni_id] = new_id
            new_id += 1
        return ins_gt_new  # Consecutive

    def get_occflow_labels(self, batch):
        """
        Extracts and processes occlusion flow labels from a batch of input data.

        Args:
            batch (dict): A batch of input data containing 'segmentation', 'instance', and 'img_is_valid' keys.

        Returns:
            dict: A dictionary containing processed 'segmentation', 'instance', and 'img_is_valid' labels.
        """
        seg_labels = batch['segmentation']
        ins_labels = batch['instance']
        img_is_valid = batch['img_is_valid']
        if not self.training:
            seg_labels = seg_labels[0]
            ins_labels = ins_labels[0]
            img_is_valid = img_is_valid[0]

        labels = dict()
        labels['segmentation'] = seg_labels[:, :self.n_future+1].long().unsqueeze(2)
        labels['instance'] = ins_labels[:, :self.n_future+1].long()
        labels['img_is_valid'] = img_is_valid[:, :self.receptive_field + self.n_future]
        return labels
