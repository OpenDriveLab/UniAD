_base_ = ["../datasets/custom_nus-3d.py", "../_base_/default_runtime.py"]
#
plugin = True
plugin_dir = "projects/mmdet3d_plugin/"
# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]
patch_size = [102.4, 102.4]
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# For nuScenes we usually do 10-class detection
class_names = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]
vehicle_id_list = [0, 1, 2, 3, 4, 6, 7]
group_id_list = [[0,1,2,3,4], [6,7], [8], [5,9]]
input_modality = dict(
    use_lidar=False, use_camera=True, use_radar=False, use_map=False, use_external=True
)
_dim_ = 256
_pos_dim_ = _dim_ // 2
_ffn_dim_ = _dim_ * 2
_num_levels_ = 4
bev_h_ = 200
bev_w_ = 200
_feed_dim_ = _ffn_dim_
_dim_half_ = _pos_dim_
canvas_size = (bev_h_, bev_w_)
queue_length = 5  # each sequence contains `queue_length` frames.

### traj prediction args ###
predict_steps = 12
predict_modes = 6
fut_steps = 4
past_steps = 4
use_nonlinear_optimizer = False
use_future_states = False
use_sdc_query = True
select_methods = 'before_mem_bank'

occflow_grid_conf = {
    'xbound': [-50.0, 50.0, 0.5],
    'ybound': [-50.0, 50.0, 0.5],
    'zbound': [-10.0, 10.0, 20.0],
}

# Other settings
fix_track_coor_bug = True

model = dict(
    type="E2EPredTransformer",
    gt_iou_threshold=0.3,
    select_methods=select_methods,
    use_sdc_query=use_sdc_query,
    queue_length=queue_length,
    use_grid_mask=True,
    video_test_mode=True,
    num_query=900,
    num_classes=10,
    vehicle_id_list=vehicle_id_list,
    pc_range=point_cloud_range,
    prev_bev_mode="prev",
    img_backbone=dict(
        type='VoVNet',
        spec_name='V-99-eSE',
        norm_eval=True,
        frozen_stages=4,
        input_ch=3,
        out_features=('stage3', 'stage4', 'stage5')),
    img_neck=dict(
        type='FPN',
        in_channels=[512, 768, 1024],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=4,
        relu_before_extra_convs=True),
  fix_feats=False,  # set fix feats to true can fix the backbone
    score_thresh=0.4,
    filter_score_thresh=0.35,
    qim_args=dict(
        qim_type="QIMBase",
        merger_dropout=0,
        update_query_pos=True,
        fp_ratio=0.3,
        random_drop=0.1,
    ),  # hyper-param for query dropping mentioned in MOTR
    mem_cfg=dict(
        memory_bank_type="MemoryBank",
        memory_bank_score_thresh=0.0,
        memory_bank_len=4,
    ),
    loss_cfg=dict(
        type="ClipMatcher",
        num_classes=10,
        weight_dict=None,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
        assigner=dict(
            type="HungarianAssigner3DTrack",
            cls_cost=dict(type="FocalLossCost", weight=2.0),
            reg_cost=dict(type="BBox3DL1Cost", weight=0.25),
            pc_range=point_cloud_range,
        ),
        loss_cls=dict(
            type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0
        ),
        loss_bbox=dict(type="L1Loss", loss_weight=0.25),
        loss_iou=dict(type="IoU3DLoss", loss_weight=0),
    ),  # loss cfg for tracking
    pts_bbox_head=dict(
        type="BEVFormerTrackHead",
        bev_h=bev_h_,
        bev_w=bev_w_,
        num_query=900,
        num_classes=10,
        in_channels=_dim_,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        past_steps=past_steps,
        fut_steps=fut_steps,
        transformer=dict(
            type="PerceptionTransformer",
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=True,
            embed_dims=_dim_,
            encoder=dict(
                type="BEVFormerEncoder",
                num_layers=6,
                pc_range=point_cloud_range,
                num_points_in_pillar=4,
                return_intermediate=False,
                transformerlayers=dict(
                    type="BEVFormerLayer",
                    attn_cfgs=[
                        dict(
                            type="TemporalSelfAttention", embed_dims=_dim_, num_levels=1
                        ),
                        dict(
                            type="SpatialCrossAttention",
                            pc_range=point_cloud_range,
                            deformable_attention=dict(
                                type="MSDeformableAttention3D",
                                embed_dims=_dim_,
                                num_points=8,
                                num_levels=_num_levels_,
                            ),
                            embed_dims=_dim_,
                        ),
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=(
                        "self_attn",
                        "norm",
                        "cross_attn",
                        "norm",
                        "ffn",
                        "norm",
                    ),
                ),
            ),
            decoder=dict(
                type="DetectionTransformerDecoder",
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type="DetrTransformerDecoderLayer",
                    attn_cfgs=[
                        dict(
                            type="MultiheadAttention",
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1,
                        ),
                        dict(
                            type="CustomMSDeformableAttention",
                            embed_dims=_dim_,
                            num_levels=1,
                        ),
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=(
                        "self_attn",
                        "norm",
                        "cross_attn",
                        "norm",
                        "ffn",
                        "norm",
                    ),
                ),
            ),
            bev_feat_no_grad=False, # False for with_grad, True for no_grad
        ),
        bbox_coder=dict(
            type="NMSFreeCoder",
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=10,
        ),
        positional_encoding=dict(
            type="LearnedPositionalEncoding",
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
        ),
        loss_cls=dict(
            type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0
        ),
        loss_bbox=dict(type="L1Loss", loss_weight=0.25),
        loss_iou=dict(type="GIoULoss", loss_weight=0.0),
    ),
    seg_head=dict(
        type='PansegformerHead',
        bev_h=bev_h_,
        bev_w=bev_w_,
        canvas_size=canvas_size,
        pc_range=point_cloud_range,
        num_query=300,
        num_classes=4,  # 3+1
        num_things_classes=3,
        num_stuff_classes=1,
        in_channels=2048,
        sync_cls_avg_factor=True,
        as_two_stage=False,
        with_box_refine=True,
        transformer=dict(
            type='Deformable_Transformer',
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention',
                        embed_dims=_dim_,
                        num_levels=_num_levels_,
                         ),
                    feedforward_channels=_feed_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='DeformableDetrTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='MultiScaleDeformableAttention',
                            embed_dims=_dim_,
                            num_levels=_num_levels_,
                        )
                    ],
                    feedforward_channels=_feed_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')
                ),
            ),
        ),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=_dim_half_,
            normalize=True,
            offset=-0.5),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0),
        loss_mask=dict(type='DiceLoss', loss_weight=2.0),
        thing_transformer_head=dict(type='MaskHead',d_model=_dim_,nhead=8,num_decoder_layers=4),
        stuff_transformer_head=dict(type='MaskHead',d_model=_dim_,nhead=8,num_decoder_layers=6,self_attn=True),
        train_cfg=dict(
            assigner=dict(
                type='HungarianAssigner',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0),
                ),
            assigner_with_mask=dict(
                type='HungarianAssigner_multi_info',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0),
                mask_cost=dict(type='DiceCost', weight=2.0),
                ),
            sampler =dict(type='PseudoSampler'),
            sampler_with_mask =dict(type='PseudoSampler_segformer'),
        ),
    ),
    occ_head=dict(
        type='OccFlowHeadFormer',

        grid_conf=occflow_grid_conf,
        ignore_index=255,

        convert_ego2lidar=True,

        bev_grid_sample=True,
        bev_proj_dim=256,
        bev_proj_nlayers=4,

        # Transformer
        attn_cur_frame=True,

        # From DETR, TODO: From Mask2Former
        transformer_decoder=dict(
            type='DetrTransformerDecoder',
            return_intermediate=True,
            num_layers=5,
            transformerlayers=dict(
                type='DetrTransformerDecoderLayer',
                attn_cfgs=dict(
                    type='MultiheadAttention',
                    embed_dims=256,
                    num_heads=8,
                    dropout=0.1),
                feedforward_channels=2048,
                ffn_dropout=0.1,
                operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                    'ffn', 'norm')),
        ),
        # Dense_decoder
        with_decoder=True,
        decoder_mode='cvt',  # ['fiery', 'cvt']


        # Query
        query_only_last_layer=True,
        query_dim=256,  # Motion Query dim: 768
        query_hidden_dim=256,
        query_mlp_layers=3,
        with_multi_query_fuser=True,
        bev_input_size=(bev_h_, bev_w_),

        sample_ignore_mode='past_valid',

        loss_mask=dict(
            type='FieryBinarySegmentationLoss',
            use_top_k=True,
            top_k_ratio=0.25,
            future_discount=0.95,
            loss_weight=5.0,
            ignore_index=255,
        ),
        loss_dice=dict(
            type='MyDiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            ignore_index=255,
            loss_weight=1.0),

        with_flow=False,
        
        pan_eval=True,
        # test_topk=30,
        test_seg_thresh=0.1,
        test_with_track_score=True,
        pred_ins_score_thres=-1, # vpq
        pred_seg_score_thres=-1, # iou
        ins_mask_alpha=2.0, # vpq
        seg_mask_alpha=2.0, # iou
        
    ),
    motion_head=dict(
        type='MotionHead',
        bev_h=bev_h_,
        bev_w=bev_w_,
        num_query=300,
        num_classes=10,
        predict_steps=predict_steps,
        predict_modes=predict_modes,
        embed_dims=_dim_,
        sync_cls_avg_factor=True,
        loss_traj=dict(type='TrajLoss', 
            use_variance=True, 
            cls_loss_weight=0.5, 	
            nll_loss_weight=0.5, 	
            loss_weight_minade=0., 	
            loss_weight_minfde=0.25),
        num_cls_fcs=3,
        use_last_query=True,
        use_kmeans_anchor=True,
        rotate_kmeans_anchors_with_yaw=True,
        use_kmeans_anchor_as_init=True,
        pc_range=point_cloud_range,
        group_id_list=group_id_list,
        num_anchor=6,
        use_future_states=False,
        anchor_info_path='/mnt/petrelfs/share_data/yangjiazhi/to_keyu/anchor_infos_mode6.pkl',
        transformerlayers=dict(
            type='MotionTransformerDecoder',
            pc_range=point_cloud_range,
            embed_dims=_dim_,
            num_layers=3,
            return_intermediate=True,
            with_map=True,
            transformerlayers=dict(
                type='TrajTransformerAttentionLayer',
                batch_first=True,
                attn_cfgs=[
                    # dict(
                    #     type='CustomModeMultiheadAttention',
                    #     embed_dims=_dim_,
                    #     num_heads=8,
                    #     dropout=0.1),
                    dict(
                        type='TrajDeformableAttention',
                        num_steps=predict_steps,
                        embed_dims=_dim_,
                        num_levels=1,
                        num_heads=8,
                        num_points=4,
                        sample_index=-1),
                ],

                feedforward_channels=_ffn_dim_,
                ffn_dropout=0.1,
                operation_order=('cross_attn', 'norm', 'ffn', 'norm')),
        ),
    ),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            grid_size=[512, 512, 1],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_size_factor=4,
            assigner=dict(
                type="HungarianAssigner3D",
                cls_cost=dict(type="FocalLossCost", weight=2.0),
                reg_cost=dict(type="BBox3DL1Cost", weight=0.25),
                iou_cost=dict(
                    type="IoUCost", weight=0.0
                ),  # Fake cost. This is just to make it compatible with DETR head.
                pc_range=point_cloud_range,
            ),
        )
    ),
)

# For Yihan: Uncomment here
# dataset_type = "NuScenesE2EDataset"
# version_subfix = ""
# data_root = "data/nuscenes/"
# info_root = "/mnt/nas37/yihan01.hu/nusc_infos/"
# file_client_args = dict(backend="disk")
# ann_file_train=info_root + f"nuscenes_infos_temporal_train{version_subfix}.pkl"
# ann_file_val=info_root + f"nuscenes_infos_temporal_val{version_subfix}.pkl"
# ann_file_test=info_root + f"nuscenes_infos_temporal_val{version_subfix}.pkl"


# For Jiazhi: Uncomment here

dataset_type = "NuScenesE2EDataset"
data_root = "/mnt/lustre/chenli1/data/nuscenes/"
info_root = "/mnt/lustre/chenli1/data/infos/"
# file_client_args = dict(backend="disk")
file_client_args = dict(
    backend='petrel',
    path_mapping=dict({
        # './data/nuscenes/': 's3://nus_bevf/',
        # 'data/nuscenes/': 's3://nus_bevf/',
        '/mnt/lustre/chenli1/data/nuscenes/': 's3://nus_bevf/',
    }))

ann_file_train=info_root + f"nuscenes_infos_temporal_train.pkl"
ann_file_val=info_root + f"nuscenes_infos_temporal_val.pkl"
ann_file_test=info_root + f"nuscenes_infos_temporal_val.pkl"


train_pipeline = [
    dict(type="LoadMultiViewImageFromFilesInCeph", to_float32=True,file_client_args=file_client_args),
    dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(
        type="LoadAnnotations3D_E2E",
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False,

        with_future_anns=True,  # occ_flow gt
        with_ins_inds_3d=True,  # ins_inds 
        ins_inds_add_1=True,    # ins_inds start from 1
    ),

    dict(type='GenerateOccFlowLabels', grid_conf=occflow_grid_conf, ignore_index=255, only_vehicle=True, 
                                    filter_invisible=False),  # NOTE: Currently vis_token is not in pkl 

    dict(type="ObjectRangeFilterTrack", point_cloud_range=point_cloud_range),
    dict(type="ObjectNameFilterTrack", classes=class_names),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="PadMultiViewImage", size_divisor=32),
    dict(type="DefaultFormatBundle3D", class_names=class_names),
    dict(
        type="CustomCollect3D",
        keys=[
            "gt_bboxes_3d",
            "gt_labels_3d",
            "gt_inds",
            "img",
            "timestamp",
            "l2g_r_mat",
            "l2g_t",
            "gt_fut_traj",
            "gt_fut_traj_mask",
            "gt_past_traj",
            "gt_past_traj_mask",
            "gt_sdc_bbox",
            "gt_sdc_label",
            "gt_sdc_fut_traj",
            "gt_sdc_fut_traj_mask",
            "gt_lane_labels",
            "gt_lane_bboxes",
            "gt_lane_masks",
             # Occ gt
            "gt_segmentation",
            "gt_instance", 
            "gt_centerness", 
            "gt_offset", 
            "gt_flow",
            "gt_backward_flow",
            # "gt_occ_future_egomotions",
            "gt_occ_has_invalid_frame",
            "gt_occ_img_is_valid",
        ],
    ),
]
test_pipeline = [
    # dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type='LoadMultiViewImageFromFilesInCeph', to_float32=True,
            file_client_args=file_client_args),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="PadMultiViewImage", size_divisor=32),
    dict(type='LoadAnnotations3D_E2E', 
         with_bbox_3d=False,  # NOTE: NO need for gt_bboxes and gt_labels for testing, only need occ_gt for evaluation
         with_label_3d=False, 
         with_attr_label=False,

         with_future_anns=True,  # occ_flow gt
         with_ins_inds_3d=False,  # No need to use matching in test
         ins_inds_add_1=True,    # ins_inds start from 1
         ),
    dict(type='GenerateOccFlowLabels', grid_conf=occflow_grid_conf, ignore_index=255, only_vehicle=True, 
                                       filter_invisible=False),  # NOTE: Currently vis_token is not in pkl
    dict(
        type="MultiScaleFlipAug3D",
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type="DefaultFormatBundle3D", class_names=class_names, with_label=False
            ),
            dict(
                type="CustomCollect3D", keys=[
                                            "img",
                                            "timestamp",
                                            "l2g_r_mat",
                                            "l2g_t",
                                            "gt_lane_labels",
                                            "gt_lane_bboxes",
                                            "gt_lane_masks",
                                            "gt_segmentation",
                                            "gt_instance", 
                                            "gt_centerness", 
                                            "gt_offset", 
                                            "gt_flow",
                                            "gt_backward_flow",
                                            # "gt_occ_future_egomotions",
                                            "gt_occ_has_invalid_frame",
                                            "gt_occ_img_is_valid",
                                        ]
            ),
        ],
    ),
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,

        # is_debug=True,
        # len_debug=0,
        
        file_client_args=file_client_args,
        data_root=data_root,
        ann_file=ann_file_train,
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        patch_size=patch_size,
        canvas_size=canvas_size,
        bev_size=(bev_h_, bev_w_),
        queue_length=queue_length,
        predict_steps=predict_steps,
        past_steps=past_steps,
        fut_steps=fut_steps,
        use_nonlinear_optimizer=use_nonlinear_optimizer,

        fix_track_coor_bug=fix_track_coor_bug,

        occ_receptive_field=3,
        occ_n_future=4,
        occ_filter_invalid_sample=False,
        # occ_convert_lidar2ego=True,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d="LiDAR",
    ),
    val=dict(
        type=dataset_type,
        file_client_args=file_client_args,
        data_root=data_root,
        ann_file=ann_file_val,
        pipeline=test_pipeline,
        patch_size=patch_size,
        canvas_size=canvas_size,
        bev_size=(bev_h_, bev_w_),
        predict_steps=predict_steps,
        past_steps=past_steps,
        fut_steps=fut_steps,
        use_nonlinear_optimizer=use_nonlinear_optimizer,
        classes=class_names,
        modality=input_modality,
        samples_per_gpu=1,
        eval_mod=['det', 'track',],

        fix_track_coor_bug=fix_track_coor_bug,

        occ_receptive_field=3,
        occ_n_future=4,
        occ_filter_invalid_sample=False,
        # occ_convert_lidar2ego=True,
    ),
    test=dict(
        type=dataset_type,
        file_client_args=file_client_args,
        data_root=data_root,
        test_mode=True,
        ann_file=ann_file_test,
        pipeline=test_pipeline,
        patch_size=patch_size,
        canvas_size=canvas_size,
        bev_size=(bev_h_, bev_w_),
        predict_steps=predict_steps,
        past_steps=past_steps,
        fut_steps=fut_steps,
        use_nonlinear_optimizer=use_nonlinear_optimizer,
        classes=class_names,
        modality=input_modality,
        eval_mod=['det', 'map', 'track', 'motion'],
        fix_track_coor_bug=fix_track_coor_bug,

    ),
    shuffler_sampler=dict(type="DistributedGroupSampler"),
    nonshuffler_sampler=dict(type="DistributedSampler"),
)
optimizer = dict(
    type="AdamW",
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            "img_backbone": dict(lr_mult=0.1),
        }
    ),
    weight_decay=0.01,
)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy="CosineAnnealing",
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)
total_epochs = 6
evaluation = dict(interval=1, pipeline=test_pipeline)
runner = dict(type="EpochBasedRunner", max_epochs=total_epochs)
log_config = dict(
    interval=25, hooks=[dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")]
)
checkpoint_config = dict(interval=1)
# load_from =  '/mnt/nas37/yihan01.hu/models/e2e/bevformer_exp/det_query_e2e_repo/dev-li-track-anchor_query_cumsum_05-05_0_025_deform_occflow128_ep2_v2/latest.pth'
# resume_from = '/mnt/petrelfs/yangjiazhi/E2EFormer_2/projects/work_dirs/e2eformer/yjz_e2eformer_qim_occ_former_motion_use_future/latest.pth'
load_from = "/mnt/lustre/chenli1/bevformer_vovnet_epoch_24.pth"
# resume_from = '/mnt/petrelfs/likeyu/work_dirs/e2emodel/track_map_v299_200x200_5frame_12ep/latest.pth'
# work_dir = "/mnt/nas37/yihan01.hu/models/e2e/test"
find_unused_parameters = True
# fp16 = dict(loss_scale=512.)