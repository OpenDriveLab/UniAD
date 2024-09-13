import mmcv
import numpy as np
import os
from collections import OrderedDict
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from nuscenes.prediction import PredictHelper
from os import path as osp
from pyquaternion import Quaternion
from shapely.geometry import MultiPoint, box
from typing import List, Tuple, Union

from mmdet3d.core.bbox.box_np_ops import points_cam2img
from mmdet3d.datasets import NuScenesDataset

nus_categories = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                  'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                  'barrier')

nus_attributes = ('cycle.with_rider', 'cycle.without_rider',
                  'pedestrian.moving', 'pedestrian.standing',
                  'pedestrian.sitting_lying_down', 'vehicle.moving',
                  'vehicle.parked', 'vehicle.stopped', 'None')

#-----------------------从对应的数据集生成蕴含其相关信息的.pkl文件，就是所谓的data info文件-----------
def create_nuscenes_infos(root_path,
                          out_path,
                          can_bus_root_path,
                          info_prefix,
                          version='v1.0-trainval',
                          max_sweeps=10):
    """Create info file of nuscene dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        version (str): Version of the data.
            Default: 'v1.0-trainval'
        max_sweeps (int): Max number of sweeps.
            Default: 10
    """
    from nuscenes.nuscenes import NuScenes
    from nuscenes.can_bus.can_bus_api import NuScenesCanBus
    print(version, root_path)
    #--------通过 NuScenes 类实例化一个对象 nusc，加载指定版本的数据集------
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    nusc_can_bus = NuScenesCanBus(dataroot=can_bus_root_path)

    #-------splits 模块包含预定义的数据集场景分割，这些分割定义了哪些场景用于训练、验证和测试-----
    from nuscenes.utils import splits
    available_vers = ['v1.0-trainval', 'v1.0-test', 'v1.0-mini']
    #-------根据版本选择训练和验证场景-------
    assert version in available_vers
    if version == 'v1.0-trainval':
        train_scenes = splits.train  #使用 splits.train 作为训练场景
        val_scenes = splits.val      #使用 splits.val 作为验证场景
    elif version == 'v1.0-test':
        train_scenes = splits.test
        val_scenes = []
    elif version == 'v1.0-mini':
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    else:
        raise ValueError('unknown')
    #注意：splits.py是一个函数，函数里面定义了train, val, test这几个列表，内容是场景scene-ID
    #这里可以直接用splits.train，splits.val，splits.test来访问/表示这几个列表


    # filter existing scenes.
    available_scenes = get_available_scenes(nusc)
    available_scene_names = [s['name'] for s in available_scenes]
    train_scenes = list(filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    train_scenes = set([
        available_scenes[available_scene_names.index(s)]['token']
        for s in train_scenes
    ])
    val_scenes = set([
        available_scenes[available_scene_names.index(s)]['token']
        for s in val_scenes
    ])

    test = 'test' in version
    if test:
        print('test scene: {}'.format(len(train_scenes)))
    else:
        print('train scene: {}, val scene: {}'.format(
            len(train_scenes), len(val_scenes)))

    train_nusc_infos, val_nusc_infos = _fill_trainval_infos(
        nusc, nusc_can_bus, train_scenes, val_scenes, test, max_sweeps=max_sweeps)

    metadata = dict(version=version)
    if test:
        print('test sample: {}'.format(len(train_nusc_infos)))
        data = dict(infos=train_nusc_infos, metadata=metadata)
        info_path = osp.join(out_path,
                             '{}_infos_temporal_test.pkl'.format(info_prefix))
        mmcv.dump(data, info_path)
    else:
        print('train sample: {}, val sample: {}'.format(
            len(train_nusc_infos), len(val_nusc_infos)))
        data = dict(infos=train_nusc_infos, metadata=metadata)
        info_path = osp.join(out_path,
                             '{}_infos_temporal_train.pkl'.format(info_prefix))
        mmcv.dump(data, info_path)
        data['infos'] = val_nusc_infos
        info_val_path = osp.join(out_path,
                                 '{}_infos_temporal_val.pkl'.format(info_prefix))
        mmcv.dump(data, info_val_path)


def get_available_scenes(nusc):
    """Get available scenes from the input nuscenes class.

    Given the raw data, get the information of available scenes for
    further info generation.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.

    Returns:
        available_scenes (list[dict]): List of basic information for the
            available scenes.
    """
    available_scenes = []
    print('total scene num: {}'.format(len(nusc.scene)))
    for scene in nusc.scene:
        scene_token = scene['token']
        scene_rec = nusc.get('scene', scene_token)
        sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
        sd_rec = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec['token'])
            lidar_path = str(lidar_path)
            if os.getcwd() in lidar_path:
                # path from lyftdataset is absolute path
                lidar_path = lidar_path.split(f'{os.getcwd()}/')[-1]
                # relative path
            if not mmcv.is_filepath(lidar_path):
                scene_not_exist = True
                break
            else:
                break
        if scene_not_exist:
            continue
        available_scenes.append(scene)
    print('exist scene num: {}'.format(len(available_scenes)))
    return available_scenes


def _get_can_bus_info(nusc, nusc_can_bus, sample):
    scene_name = nusc.get('scene', sample['scene_token'])['name']
    sample_timestamp = sample['timestamp']
    try:
        pose_list = nusc_can_bus.get_messages(scene_name, 'pose')
    except:
        return np.zeros(18)  # server scenes do not have can bus information.
    can_bus = []
    # during each scene, the first timestamp of can_bus may be large than the first sample's timestamp
    last_pose = pose_list[0]
    for i, pose in enumerate(pose_list):
        if pose['utime'] > sample_timestamp:
            break
        last_pose = pose
    _ = last_pose.pop('utime')  # useless
    pos = last_pose.pop('pos')
    rotation = last_pose.pop('orientation')
    can_bus.extend(pos)
    can_bus.extend(rotation)
    for key in last_pose.keys():
        can_bus.extend(pose[key])  # 16 elements
    can_bus.extend([0., 0.])
    return np.array(can_bus)

def _get_future_traj_info(nusc, sample, predict_steps=16):
    sample_token = sample['token']
    ann_tokens = np.array(sample['anns'])
    sd_rec = nusc.get('sample', sample_token)
    fut_traj_all = []
    fut_traj_valid_mask_all = []
    _, boxes, _ = nusc.get_sample_data(sd_rec['data']['LIDAR_TOP'], selected_anntokens=ann_tokens)
    predict_helper = PredictHelper(nusc)
    for i, ann_token in enumerate(ann_tokens):
        box = boxes[i]
        instance_token = nusc.get('sample_annotation', ann_token)['instance_token']
        fut_traj_local = predict_helper.get_future_for_agent(instance_token,
                                                             sample_token,
                                                             seconds=predict_steps//2,
                                                             in_agent_frame=True)

        fut_traj = np.zeros((predict_steps, 2))
        fut_traj_valid_mask = np.zeros((predict_steps, 2))
        if fut_traj_local.shape[0] > 0:
            # trans = box.center
            # trans = np.array([0, 0, 0])
            # rot = Quaternion(matrix=box.rotation_matrix)
            # fut_traj_scence_centric = convert_local_coords_to_global(fut_traj_local, trans, rot)  
            fut_traj_scence_centric = fut_traj_local
            fut_traj[:fut_traj_scence_centric.shape[0], :] = fut_traj_scence_centric
            fut_traj_valid_mask[:fut_traj_scence_centric.shape[0], :] = 1
        fut_traj_all.append(fut_traj)
        fut_traj_valid_mask_all.append(fut_traj_valid_mask)
    if len(ann_tokens) > 0:
        fut_traj_all = np.stack(fut_traj_all, axis=0)
        fut_traj_valid_mask_all = np.stack(fut_traj_valid_mask_all, axis=0)
    else:
        fut_traj_all = np.zeros((0, predict_steps, 2))
        fut_traj_valid_mask_all = np.zeros((0, predict_steps, 2))
    return fut_traj_all, fut_traj_valid_mask_all

def _fill_trainval_infos(nusc,
                         nusc_can_bus,
                         train_scenes,
                         val_scenes,
                         test=False,
                         max_sweeps=10):
    """Generate the train/val infos from the raw data.

    Args:
        nusc (:obj:`NuScenes`): Dataset class in the nuScenes dataset.
        train_scenes (list[str]): Basic information of training scenes.
        val_scenes (list[str]): Basic information of validation scenes.
        test (bool): Whether use the test mode. In the test mode, no
            annotations can be accessed. Default: False.
        max_sweeps (int): Max number of sweeps. Default: 10.

    Returns:
        tuple[list[dict]]: Information of training set and validation set
            that will be saved to the info file.
    """
    train_nusc_infos = []
    val_nusc_infos = []
    frame_idx = 0
    for sample in mmcv.track_iter_progress(nusc.sample):
        lidar_token = sample['data']['LIDAR_TOP']
        sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        cs_record = nusc.get('calibrated_sensor',
                             sd_rec['calibrated_sensor_token'])
        pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
        lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)

        mmcv.check_file_exist(lidar_path)
        can_bus = _get_can_bus_info(nusc, nusc_can_bus, sample)
        ##
        info = {
            'lidar_path': lidar_path,
            'token': sample['token'],
            'prev': sample['prev'],
            'next': sample['next'],
            'can_bus': can_bus,
            'frame_idx': frame_idx,  # temporal related info
            'sweeps': [],
            'cams': dict(),
            'scene_token': sample['scene_token'],  # temporal related info
            'lidar2ego_translation': cs_record['translation'],
            'lidar2ego_rotation': cs_record['rotation'],
            'ego2global_translation': pose_record['translation'],
            'ego2global_rotation': pose_record['rotation'],
            'timestamp': sample['timestamp'],
        }

        if sample['next'] == '':
            frame_idx = 0
        else:
            frame_idx += 1

        l2e_r = info['lidar2ego_rotation']
        l2e_t = info['lidar2ego_translation']
        e2g_r = info['ego2global_rotation']
        e2g_t = info['ego2global_translation']
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        # obtain 6 image's information per frame
        camera_types = [
            'CAM_FRONT',
            'CAM_FRONT_RIGHT',
            'CAM_FRONT_LEFT',
            'CAM_BACK',
            'CAM_BACK_LEFT',
            'CAM_BACK_RIGHT',
        ]
        for cam in camera_types:
            cam_token = sample['data'][cam]
            cam_path, _, cam_intrinsic = nusc.get_sample_data(cam_token)
            cam_info = obtain_sensor2top(nusc, cam_token, l2e_t, l2e_r_mat,
                                         e2g_t, e2g_r_mat, cam)
            cam_info.update(cam_intrinsic=cam_intrinsic)
            info['cams'].update({cam: cam_info})

        # obtain sweeps for a single key-frame
        sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        sweeps = []
        while len(sweeps) < max_sweeps:
            if not sd_rec['prev'] == '':
                sweep = obtain_sensor2top(nusc, sd_rec['prev'], l2e_t,
                                          l2e_r_mat, e2g_t, e2g_r_mat, 'lidar')
                sweeps.append(sweep)
                sd_rec = nusc.get('sample_data', sd_rec['prev'])
            else:
                break
        info['sweeps'] = sweeps
        # obtain annotation
        if not test:
            annotations = [
                nusc.get('sample_annotation', token)
                for token in sample['anns']
            ]
            locs = np.array([b.center for b in boxes]).reshape(-1, 3)
            dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
            rots = np.array([b.orientation.yaw_pitch_roll[0]
                             for b in boxes]).reshape(-1, 1)
            velocity = np.array(
                [nusc.box_velocity(token)[:2] for token in sample['anns']])
            valid_flag = np.array(
                [(anno['num_lidar_pts'] + anno['num_radar_pts']) > 0
                 for anno in annotations],
                dtype=bool).reshape(-1)
            instance_inds = [nusc.getind('instance', ann['instance_token'])
                             for ann in annotations]
            future_traj_all, future_traj_valid_mask_all = _get_future_traj_info(nusc, sample)
            instance_tokens = [ann['instance_token'] for ann in annotations]  # dtype('<U[length_of_str]')

            # TODO: Add traj in next dataset_version
            # future_traj_all, future_traj_valid_mask_all = _get_future_traj_info(nusc, sample)
            # convert velo from global to lidar
            for i in range(len(boxes)):
                velo = np.array([*velocity[i], 0.0])
                velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(
                    l2e_r_mat).T
                velocity[i] = velo[:2]

            names = [b.name for b in boxes]
            for i in range(len(names)):
                if names[i] in NuScenesDataset.NameMapping:
                    names[i] = NuScenesDataset.NameMapping[names[i]]
            names = np.array(names)
            # instance_inds = [nusc.getind('instance', ann['instance_token']) for ann in annotations]
            # we need to convert rot to SECOND format.
            gt_boxes = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1)
            assert len(gt_boxes) == len(
                annotations), f'{len(gt_boxes)}, {len(annotations)}'
            info['gt_boxes'] = gt_boxes
            info['gt_names'] = names
            info['gt_velocity'] = velocity.reshape(-1, 2)
            info['num_lidar_pts'] = np.array(
                [a['num_lidar_pts'] for a in annotations])
            info['num_radar_pts'] = np.array(
                [a['num_radar_pts'] for a in annotations])
            info['valid_flag'] = valid_flag
            info['gt_inds'] = np.array(instance_inds)
            info['gt_ins_tokens'] = np.array(instance_tokens)
            info['fut_traj'] = future_traj_all
            info['fut_traj_valid_mask'] = future_traj_valid_mask_all

            # add visibility_tokens
            visibility_tokens = [int(anno['visibility_token'])
                                 for anno in annotations]
            info['visibility_tokens'] = np.array(visibility_tokens)

        if sample['scene_token'] in train_scenes:
            train_nusc_infos.append(info)
        else:
            val_nusc_infos.append(info)

    return train_nusc_infos, val_nusc_infos


def obtain_sensor2top(nusc,
                      sensor_token,
                      l2e_t,
                      l2e_r_mat,
                      e2g_t,
                      e2g_r_mat,
                      sensor_type='lidar'):
    """Obtain the info with RT matric from general sensor to Top LiDAR.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.
        sensor_token (str): Sample data token corresponding to the
            specific sensor type.
        l2e_t (np.ndarray): Translation from lidar to ego in shape (1, 3).
        l2e_r_mat (np.ndarray): Rotation matrix from lidar to ego
            in shape (3, 3).
        e2g_t (np.ndarray): Translation from ego to global in shape (1, 3).
        e2g_r_mat (np.ndarray): Rotation matrix from ego to global
            in shape (3, 3).
        sensor_type (str): Sensor to calibrate. Default: 'lidar'.

    Returns:
        sweep (dict): Sweep information after transformation.
    """
    sd_rec = nusc.get('sample_data', sensor_token)
    cs_record = nusc.get('calibrated_sensor',
                         sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    data_path = str(nusc.get_sample_data_path(sd_rec['token']))
    if os.getcwd() in data_path:  # path from lyftdataset is absolute path
        data_path = data_path.split(f'{os.getcwd()}/')[-1]  # relative path
    sweep = {
        'data_path': data_path,
        'type': sensor_type,
        'sample_data_token': sd_rec['token'],
        'sensor2ego_translation': cs_record['translation'],
        'sensor2ego_rotation': cs_record['rotation'],
        'ego2global_translation': pose_record['translation'],
        'ego2global_rotation': pose_record['rotation'],
        'timestamp': sd_rec['timestamp']
    }

    l2e_r_s = sweep['sensor2ego_rotation']
    l2e_t_s = sweep['sensor2ego_translation']
    e2g_r_s = sweep['ego2global_rotation']
    e2g_t_s = sweep['ego2global_translation']

    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego'->lidar
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                  ) + l2e_t @ np.linalg.inv(l2e_r_mat).T
    sweep['sensor2lidar_rotation'] = R.T  # points @ R.T + T
    sweep['sensor2lidar_translation'] = T
    return sweep


def export_2d_annotation(root_path, info_path, version, mono3d=True):
    """Export 2d annotation from the info file and raw data.

    Args:
        root_path (str): Root path of the raw data.
        info_path (str): Path of the info file.
        version (str): Dataset version.
        mono3d (bool): Whether to export mono3d annotation. Default: True.
    """
    # get bbox annotations for camera
    camera_types = [
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_FRONT_LEFT',
        'CAM_BACK',
        'CAM_BACK_LEFT',
        'CAM_BACK_RIGHT',
    ]
    nusc_infos = mmcv.load(info_path)['infos']
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    # info_2d_list = []
    cat2Ids = [
        dict(id=nus_categories.index(cat_name), name=cat_name)
        for cat_name in nus_categories
    ]
    coco_ann_id = 0
    coco_2d_dict = dict(annotations=[], images=[], categories=cat2Ids)
    for info in mmcv.track_iter_progress(nusc_infos):
        for cam in camera_types:
            cam_info = info['cams'][cam]
            coco_infos = get_2d_boxes(
                nusc,
                cam_info['sample_data_token'],
                visibilities=['', '1', '2', '3', '4'],
                mono3d=mono3d)
            (height, width, _) = mmcv.imread(cam_info['data_path']).shape
            coco_2d_dict['images'].append(
                dict(
                    file_name=cam_info['data_path'].split('data/nuscenes/')
                    [-1],
                    id=cam_info['sample_data_token'],
                    token=info['token'],
                    cam2ego_rotation=cam_info['sensor2ego_rotation'],
                    cam2ego_translation=cam_info['sensor2ego_translation'],
                    ego2global_rotation=info['ego2global_rotation'],
                    ego2global_translation=info['ego2global_translation'],
                    cam_intrinsic=cam_info['cam_intrinsic'],
                    width=width,
                    height=height))
            for coco_info in coco_infos:
                if coco_info is None:
                    continue
                # add an empty key for coco format
                coco_info['segmentation'] = []
                coco_info['id'] = coco_ann_id
                coco_2d_dict['annotations'].append(coco_info)
                coco_ann_id += 1
    if mono3d:
        json_prefix = f'{info_path[:-4]}_mono3d'
    else:
        json_prefix = f'{info_path[:-4]}'
    mmcv.dump(coco_2d_dict, f'{json_prefix}.coco.json')


def get_2d_boxes(nusc,
                 sample_data_token: str,
                 visibilities: List[str],
                 mono3d=True):
    """Get the 2D annotation records for a given `sample_data_token`.

    Args:
        sample_data_token (str): Sample data token belonging to a camera \
            keyframe.
        visibilities (list[str]): Visibility filter.
        mono3d (bool): Whether to get boxes with mono3d annotation.

    Return:
        list[dict]: List of 2D annotation record that belongs to the input
            `sample_data_token`.
    """

    # Get the sample data and the sample corresponding to that sample data.
    sd_rec = nusc.get('sample_data', sample_data_token)

    assert sd_rec[
        'sensor_modality'] == 'camera', 'Error: get_2d_boxes only works' \
        ' for camera sample_data!'
    if not sd_rec['is_key_frame']:
        raise ValueError(
            'The 2D re-projections are available only for keyframes.')

    s_rec = nusc.get('sample', sd_rec['sample_token'])

    # Get the calibrated sensor and ego pose
    # record to get the transformation matrices.
    cs_rec = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    pose_rec = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    camera_intrinsic = np.array(cs_rec['camera_intrinsic'])

    # Get all the annotation with the specified visibilties.
    ann_recs = [
        nusc.get('sample_annotation', token) for token in s_rec['anns']
    ]
    ann_recs = [
        ann_rec for ann_rec in ann_recs
        if (ann_rec['visibility_token'] in visibilities)
    ]

    repro_recs = []

    for ann_rec in ann_recs:
        # Augment sample_annotation with token information.
        ann_rec['sample_annotation_token'] = ann_rec['token']
        ann_rec['sample_data_token'] = sample_data_token

        # Get the box in global coordinates.
        box = nusc.get_box(ann_rec['token'])

        # Move them to the ego-pose frame.
        box.translate(-np.array(pose_rec['translation']))
        box.rotate(Quaternion(pose_rec['rotation']).inverse)

        # Move them to the calibrated sensor frame.
        box.translate(-np.array(cs_rec['translation']))
        box.rotate(Quaternion(cs_rec['rotation']).inverse)

        # Filter out the corners that are not in front of the calibrated
        # sensor.
        corners_3d = box.corners()
        in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
        corners_3d = corners_3d[:, in_front]

        # Project 3d box to 2d.
        corner_coords = view_points(corners_3d, camera_intrinsic,
                                    True).T[:, :2].tolist()

        # Keep only corners that fall within the image.
        final_coords = post_process_coords(corner_coords)

        # Skip if the convex hull of the re-projected corners
        # does not intersect the image canvas.
        if final_coords is None:
            continue
        else:
            min_x, min_y, max_x, max_y = final_coords

        # Generate dictionary record to be included in the .json file.
        repro_rec = generate_record(ann_rec, min_x, min_y, max_x, max_y,
                                    sample_data_token, sd_rec['filename'])

        # If mono3d=True, add 3D annotations in camera coordinates
        if mono3d and (repro_rec is not None):
            loc = box.center.tolist()

            dim = box.wlh
            dim[[0, 1, 2]] = dim[[1, 2, 0]]  # convert wlh to our lhw
            dim = dim.tolist()

            rot = box.orientation.yaw_pitch_roll[0]
            rot = [-rot]  # convert the rot to our cam coordinate

            global_velo2d = nusc.box_velocity(box.token)[:2]
            global_velo3d = np.array([*global_velo2d, 0.0])
            e2g_r_mat = Quaternion(pose_rec['rotation']).rotation_matrix
            c2e_r_mat = Quaternion(cs_rec['rotation']).rotation_matrix
            cam_velo3d = global_velo3d @ np.linalg.inv(
                e2g_r_mat).T @ np.linalg.inv(c2e_r_mat).T
            velo = cam_velo3d[0::2].tolist()

            repro_rec['bbox_cam3d'] = loc + dim + rot
            repro_rec['velo_cam3d'] = velo

            center3d = np.array(loc).reshape([1, 3])
            center2d = points_cam2img(
                center3d, camera_intrinsic, with_depth=True)
            repro_rec['center2d'] = center2d.squeeze().tolist()
            # normalized center2D + depth
            # if samples with depth < 0 will be removed
            if repro_rec['center2d'][2] <= 0:
                continue

            ann_token = nusc.get('sample_annotation',
                                 box.token)['attribute_tokens']
            if len(ann_token) == 0:
                attr_name = 'None'
            else:
                attr_name = nusc.get('attribute', ann_token[0])['name']
            attr_id = nus_attributes.index(attr_name)
            repro_rec['attribute_name'] = attr_name
            repro_rec['attribute_id'] = attr_id

        repro_recs.append(repro_rec)

    return repro_recs


def post_process_coords(
    corner_coords: List, imsize: Tuple[int, int] = (1600, 900)
) -> Union[Tuple[float, float, float, float], None]:
    """Get the intersection of the convex hull of the reprojected bbox corners
    and the image canvas, return None if no intersection.

    Args:
        corner_coords (list[int]): Corner coordinates of reprojected
            bounding box.
        imsize (tuple[int]): Size of the image canvas.

    Return:
        tuple [float]: Intersection of the convex hull of the 2D box
            corners and the image canvas.
    """
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = box(0, 0, imsize[0], imsize[1])

    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        intersection_coords = np.array(
            [coord for coord in img_intersection.exterior.coords])

        min_x = min(intersection_coords[:, 0])
        min_y = min(intersection_coords[:, 1])
        max_x = max(intersection_coords[:, 0])
        max_y = max(intersection_coords[:, 1])

        return min_x, min_y, max_x, max_y
    else:
        return None


def generate_record(ann_rec: dict, x1: float, y1: float, x2: float, y2: float,
                    sample_data_token: str, filename: str) -> OrderedDict:
    """Generate one 2D annotation record given various informations on top of
    the 2D bounding box coordinates.

    Args:
        ann_rec (dict): Original 3d annotation record.
        x1 (float): Minimum value of the x coordinate.
        y1 (float): Minimum value of the y coordinate.
        x2 (float): Maximum value of the x coordinate.
        y2 (float): Maximum value of the y coordinate.
        sample_data_token (str): Sample data token.
        filename (str):The corresponding image file where the annotation
            is present.

    Returns:
        dict: A sample 2D annotation record.
            - file_name (str): flie name
            - image_id (str): sample data token
            - area (float): 2d box area
            - category_name (str): category name
            - category_id (int): category id
            - bbox (list[float]): left x, top y, dx, dy of 2d box
            - iscrowd (int): whether the area is crowd
    """
    repro_rec = OrderedDict()
    repro_rec['sample_data_token'] = sample_data_token
    coco_rec = dict()

    relevant_keys = [
        'attribute_tokens',
        'category_name',
        'instance_token',
        'next',
        'num_lidar_pts',
        'num_radar_pts',
        'prev',
        'sample_annotation_token',
        'sample_data_token',
        'visibility_token',
    ]

    for key, value in ann_rec.items():
        if key in relevant_keys:
            repro_rec[key] = value

    repro_rec['bbox_corners'] = [x1, y1, x2, y2]
    repro_rec['filename'] = filename

    coco_rec['file_name'] = filename
    coco_rec['image_id'] = sample_data_token
    coco_rec['area'] = (y2 - y1) * (x2 - x1)

    if repro_rec['category_name'] not in NuScenesDataset.NameMapping:
        return None
    cat_name = NuScenesDataset.NameMapping[repro_rec['category_name']]
    coco_rec['category_name'] = cat_name
    coco_rec['category_id'] = nus_categories.index(cat_name)
    coco_rec['bbox'] = [x1, y1, x2 - x1, y2 - y1]
    coco_rec['iscrowd'] = 0

    return coco_rec
    