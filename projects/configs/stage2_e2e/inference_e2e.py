_base_ = ["./base_e2e.py"]

img_norm_cfg = dict(mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
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

inference_pipeline = [
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="PadMultiViewImage", size_divisor=32),
    # dict(
    #     type="MultiScaleFlipAug3D",
    #     img_scale=(1600, 900),
    #     pts_scale_ratio=1,
    #     flip=False,
    #     transforms=[
    #         dict(
    #             type="DefaultFormatBundle3D",
    #             class_names=class_names,
    #             with_label=False,
    #             with_gt=False,
    #         ),
    #         dict(
    #             type="CustomCollect3D",
    #             keys=[
    #                 "img",
    #                 "timestamp",
    #                 "l2g_r_mat",
    #                 "l2g_t",
    #             ],
    #         ),
    #     ],
    # ),
]
