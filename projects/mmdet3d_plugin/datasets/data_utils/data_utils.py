import math
import numpy as np
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from nuscenes.utils.data_classes import Box as NuScenesBox
import pyquaternion


def output_to_nusc_box(detection):
    """Convert the output to the box class in the nuScenes.
    Args:
        detection (dict): Detection results.
            - boxes_3d (:obj:`BaseInstance3DBoxes`): Detection bbox.
            - scores_3d (torch.Tensor): Detection scores.
            - labels_3d (torch.Tensor): Predicted box labels.
    Returns:
        list[:obj:`NuScenesBox`]: List of standard NuScenesBoxes.
    """
    box3d = detection['boxes_3d']
    scores = detection['scores_3d'].numpy()
    labels = detection['labels_3d'].numpy()
    if 'track_ids' in detection:
        ids = detection['track_ids'].numpy()
    else:
        ids = np.ones_like(labels)

    box_gravity_center = box3d.gravity_center.numpy()
    box_dims = box3d.dims.numpy()
    box_yaw = box3d.yaw.numpy()
    # TODO: check whether this is necessary
    # with dir_offset & dir_limit in the head
    box_yaw = -box_yaw - np.pi / 2

    box_list = []
    for i in range(len(box3d)):
        quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
        velocity = (*box3d.tensor[i, 7:9], 0.0)
        # velo_val = np.linalg.norm(box3d[i, 7:9])
        # velo_ori = box3d[i, 6]
        # velocity = (
        # velo_val * np.cos(velo_ori), velo_val * np.sin(velo_ori), 0.0)
        box = NuScenesBox(
            box_gravity_center[i],
            box_dims[i],
            quat,
            label=labels[i],
            score=scores[i],
            velocity=velocity)
        box.token = ids[i]
        box_list.append(box)
    return box_list


def output_to_nusc_box_det(detection):
    """Convert the output to the box class in the nuScenes.

    Args:
        detection (dict): Detection results.

            - boxes_3d (:obj:`BaseInstance3DBoxes`): Detection bbox.
            - scores_3d (torch.Tensor): Detection scores.
            - labels_3d (torch.Tensor): Predicted box labels.

    Returns:
        list[:obj:`NuScenesBox`]: List of standard NuScenesBoxes.
    """
    if 'boxes_3d_det' in detection:
        box3d = detection['boxes_3d_det']
        scores = detection['scores_3d_det'].numpy()
        labels = detection['labels_3d_det'].numpy()
    else:
        box3d = detection['boxes_3d']
        scores = detection['scores_3d'].numpy()
        labels = detection['labels_3d'].numpy()

    box_gravity_center = box3d.gravity_center.numpy()
    box_dims = box3d.dims.numpy()
    box_yaw = box3d.yaw.numpy()
    # TODO: check whether this is necessary
    # with dir_offset & dir_limit in the head
    box_yaw = -box_yaw - np.pi / 2

    box_list = []
    for i in range(len(box3d)):
        quat = Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
        velocity = (*box3d.tensor[i, 7:9], 0.0)
        box = NuScenesBox(
            box_gravity_center[i],
            box_dims[i],
            quat,
            label=labels[i],
            score=scores[i],
            velocity=velocity)
        box_list.append(box)
    return box_list


def lidar_nusc_box_to_global(info,
                             boxes,
                             classes,
                             eval_configs,
                             eval_version='detection_cvpr_2019'):
    """Convert the box from ego to global coordinate.
    Args:
        info (dict): Info for a specific sample data, including the
            calibration information.
        boxes (list[:obj:`NuScenesBox`]): List of predicted NuScenesBoxes.
        classes (list[str]): Mapped classes in the evaluation.
        eval_configs (object): Evaluation configuration object.
        eval_version (str, optional): Evaluation version.
            Default: 'detection_cvpr_2019'
    Returns:
        list: List of standard NuScenesBoxes in the global
            coordinate.
    """
    box_list = []
    keep_idx = []
    for i, box in enumerate(boxes):
        # Move box to ego vehicle coord system
        box.rotate(Quaternion(info['lidar2ego_rotation']))
        box.translate(np.array(info['lidar2ego_translation']))
        # filter det in ego.
        cls_range_map = eval_configs.class_range
        radius = np.linalg.norm(box.center[:2], 2)
        det_range = cls_range_map[classes[box.label]]
        if radius > det_range:
            continue
        # Move box to global coord system
        box.rotate(Quaternion(info['ego2global_rotation']))
        box.translate(np.array(info['ego2global_translation']))
        box_list.append(box)
        keep_idx.append(i)
    return box_list, keep_idx


def obtain_map_info(nusc,
                    nusc_maps,
                    sample,
                    patch_size=(102.4, 102.4),
                    canvas_size=(256, 256),
                    layer_names=['lane_divider', 'road_divider'],
                    thickness=10):
    """
    Export 2d annotation from the info file and raw data.
    """
    l2e_r = sample['lidar2ego_rotation']
    l2e_t = sample['lidar2ego_translation']
    e2g_r = sample['ego2global_rotation']
    e2g_t = sample['ego2global_translation']
    l2e_r_mat = Quaternion(l2e_r).rotation_matrix
    e2g_r_mat = Quaternion(e2g_r).rotation_matrix

    scene = nusc.get('scene', sample['scene_token'])
    log = nusc.get('log', scene['log_token'])
    nusc_map = nusc_maps[log['location']]
    if layer_names is None:
        layer_names = nusc_map.non_geometric_layers

    l2g_r_mat = (l2e_r_mat.T @ e2g_r_mat.T).T
    l2g_t = l2e_t @ e2g_r_mat.T + e2g_t
    patch_box = (l2g_t[0], l2g_t[1], patch_size[0], patch_size[1])
    patch_angle = math.degrees(Quaternion(matrix=l2g_r_mat).yaw_pitch_roll[0])

    map_mask = nusc_map.get_map_mask(
        patch_box, patch_angle, layer_names, canvas_size=canvas_size)
    map_mask = map_mask[-2] | map_mask[-1]
    map_mask = map_mask[np.newaxis, :]
    map_mask = map_mask.transpose((2, 1, 0)).squeeze(2)  # (H, W, C)

    erode = nusc_map.get_map_mask(patch_box, patch_angle, [
                                  'drivable_area'], canvas_size=canvas_size)
    erode = erode.transpose((2, 1, 0)).squeeze(2)

    map_mask = np.concatenate([erode[None], map_mask[None]], axis=0)
    return map_mask
