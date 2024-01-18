from dataclasses import dataclass
from typing import Dict, Optional
import uuid
from mmdet3d.datasets.pipelines import Compose
from mmdet.datasets.builder import PIPELINES
from mmdet3d.models import build_model
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmcv.runner import load_checkpoint
from mmcv import Config
import numpy as np
from pyquaternion import Quaternion
import torch

from projects.mmdet3d_plugin.uniad.detectors.uniad_e2e import UniAD
from projects.mmdet3d_plugin.datasets.nuscenes_e2e_dataset import (
    quaternion_yaw,
)
from tools.data_converter.uniad_nuscenes_converter import _get_can_bus_info

# NOTE: this is what they do to the can bus signals
# in preproc of the dataset
### pos 3 | m in global frame
### orientation 4 | quaternion expressed in the ego-vehicle frame
### accel 3 | m/s^2 in ego-vehicle frame
### rotation_rate 3 | rad/s in ego-vehicle frame
### vel 3 | m/s in ego-vehicle frame
### zeros 2

# in preproc of the dataloader
### rotation = Quaternion(input_dict["ego2global_rotation"]) # = nuscenes pose_record["rotation"]
### translation = input_dict["ego2global_translation"] # = nuscenes pose_record["translation"]
### can_bus = input_dict["can_bus"]
### can_bus[:3] = translation
### can_bus[3:7] = rotation
### patch_angle = quaternion_yaw(rotation) / np.pi * 180
### if patch_angle < 0:
###     patch_angle += 360
### can_bus[-2] = patch_angle / 180 * np.pi
### can_bus[-1] = patch_angle

# this means that the can bus signals are:
# 0-3 translation in global frame (to ego-frame)
# 3-7 rotation in global frame (to ego-frame)
# 7-10 acceleration in ego-frame
# 10-13 rotation rate in ego-frame
# 13-16 velocity in ego-frame
# 16 patch angle in degrees
# 17 patch angle in radians


@dataclass
class UniADInferenceInput:
    imgs: np.ndarray
    """shape: (n-cams (6), 3, h (900), w (1600)) | images without any preprocessing. should be in RGB order"""
    pose: np.ndarray
    """shape: (3, 4) | lidar pose in global frame"""
    lidar2img: np.ndarray
    """shape: (n-cams (6), 4, 4) | lidar2img transformation matrix, i.e., lidar2cam @ camera2img"""
    timestamp: float
    """timestamp of the current frame in seconds"""
    can_bus_signals: np.ndarray
    """shape: (18,) | see above for details"""
    command: int
    """0: right, 1: left, 2: straight"""


@dataclass
class UniADInferenceOutput:
    trajectory: np.ndarray
    """shape: (n-future (6), 2) | predicted trajectory in the ego-frame @ 2Hz"""
    aux_outputs: Optional[Dict] = None
    """aux outputs such as objects, tracks, segmentation and motion forecast"""


class UniADRunner:
    def __init__(self, config_path: str, checkpoint_path: str, device: torch.device):
        config = Config.fromfile(config_path)
        self.config = config

        self.model: UniAD = build_model(
            config.model, train_cfg=None, test_cfg=config.get("test_cfg")
        )

        self.model.eval()
        # load the checkpoint
        if checkpoint_path is not None:
            _ = load_checkpoint(self.model, checkpoint_path, map_location="cpu")
        # do more stuff here maybe?
        self.model = self.model.to(device)
        self.device = device
        self.preproc_pipeline = Compose(config.inference_pipeline)
        self.reset()

    def reset(self):
        # making a new scene token for each new scene. these are used in the model.
        self.scene_token = uuid.uuid4()

    def preproc(self, input: UniADInferenceInput):
        # TODO: make torch version of the preproc pipeline instead of using mmcv version'
        raise NotImplementedError

    def forward_inference(
        self, input: UniADInferenceInput, command: int = 2
    ) -> UniADInferenceOutput:
        """Run inference without all the preprocessed dataset stuff.

        This functions is a modified version of the forward_test function in UniAD.

        Args:
            inputs: UniADInferenceInput
            command: int, 0: right, 1: left, 2: straight

        """
        # input to preproc shoudl be dict(img=imgs) where imgs: n x h x w x c in bgr format
        # permute rgb -> bgr
        imgs = input.imgs[:, ::-1, :, :]
        # flip nchw to nhwc
        imgs = np.moveaxis(imgs, 1, -1)
        preproc_input = dict(img=imgs)
        # run it through the inference pipeline (which is same as eval pipeline except not loading annotations)
        preproc_output = self.preproc_pipeline(preproc_input)
        # collect in array as will convert to tensor, but currently it is a list of arrays (n, h, w, c)
        imgs = np.array(preproc_output["img"])
        # move back to the nchw format
        imgs = np.moveaxis(imgs, -1, 1)
        # convert to tensor and move to device
        imgs = torch.from_numpy(imgs).to(self.device)
        # img should be (1, n, 3, h, w)
        imgs = imgs.unsqueeze(0)
        # move other input to the device as well
        l2g_t = (
            torch.from_numpy(input.pose[:3, 3]).to(self.device).unsqueeze(0)
        )  # should be 1x3
        l2g_r_mat = (
            torch.from_numpy(input.pose[:3, :3]).to(self.device).unsqueeze(0)
        )  # should be 1x3x3
        timestamp = (
            torch.from_numpy(np.array([input.timestamp])).to(self.device).unsqueeze(0)
        )

        # we need to emulate the img_metas here in order to run the model.
        img_metas = {
            "scene_token": self.scene_token,
            "can_bus": input.can_bus_signals,
            "lidar2img": input.lidar2img,  # lidar2cam @ camera2img
            "img_shape": preproc_output["img_shape"],
            # we need this as they are used in the model somewhere.
            "box_type_3d": LiDARInstance3DBoxes,
        }

        outs_track = self.model.simple_test_track(
            imgs, l2g_t, l2g_r_mat, img_metas, timestamp
        )
        outs_track[0] = self.model.upsample_bev_if_tiny(outs_track[0])

        # get the bev embedding
        bev_embed = outs_track[0]["bev_embed"]

        # get the segmentation result using the bev embedding
        outs_seg = self.model.seg_head.forward(bev_embed)

        # get the motion
        _, outs_motion = self.model.motion_head.forward_test(
            bev_embed, outs_track, outs_seg
        )
        outs_motion["bev_pos"] = outs_track[0]["bev_pos"]

        # get the occ result
        occ_no_query = outs_motion["track_query"].shape[1] == 0
        if occ_no_query:
            pass
        # more stuff here

        ins_query = self.model.occ_head.merge_queries(
            outs_motion, self.model.occ_head.detach_query_pos
        )
        _, pred_ins_logits = self.model.occ_head.forward(bev_embed, ins_query=ins_query)
        pred_ins_logits = pred_ins_logits[:, :, : 1 + self.model.n_future]
        pred_ins_sigmoid = pred_ins_logits.sigmoid()
        pred_seg_scores = pred_ins_sigmoid.max(1)[0]
        occ_mask = (
            (pred_seg_scores > self.model.occ_head.test_seg_thresh).long().unsqueeze(2)
        )

        # get the planning output
        outs_planning = self.model.planning_head.forward(
            bev_embed,
            occ_mask,
            outs_motion["bev_pos"],
            outs_motion["sdc_traj_query"],
            outs_motion["sdc_track_query"],
            command=command,
        )

        return UniADInferenceOutput(trajectory=outs_planning["sdc_traj"].cpu().numpy())

    def _emulate_nuscenes_canbus_signals(
        self, prev_pose, prev_timestamp, prev_vel, current_pose, current_timestamp
    ):
        """Emulate the can bus signals using backward difference.

        Args:
            prev_pose: np.ndarray, shape: (3, 4)
            prev_timestamp: float
            prev_vel: np.ndarray, shape: (3,)
            current_pose: np.ndarray, shape: (3, 4)
            current_timestamp: float

        Returns:
            can_bus: np.ndarray, shape: (18,)
                - 0-3 translation in global frame (to ego-frame) (z is always zero)
                - 3-7 rotation in global frame (to ego-frame) (4:6 is always zero)
                - 7-10 acceleration in ego-frame (we have an positive g in z)
                - 10-13 rotation rate in ego-frame
                - 13-16 velocity in ego-frame (14, 15 are zeros)
                - 16 patch angle in degrees
                - 17 patch angle in radians
        """
        delta_t = current_timestamp - prev_timestamp
        assert delta_t > 0, "delta_t should be positive"

        can_bus = np.zeros(18)
        # translation
        can_bus[:3] = current_pose[:3, 3].copy()
        # rotation
        rotation = Quaternion(matrix=current_pose[:3, :3].copy())
        can_bus[3:7] = rotation

        # rotation rate, we approximate this as the angle difference around z axis divided by delta_t
        delta_yaw = quaternion_yaw(rotation.copy()) - quaternion_yaw(
            Quaternion(matrix=prev_pose[:3, :3].copy())
        )
        yaw_rate = delta_yaw / delta_t
        can_bus[10] = yaw_rate

        # velocity
        delta_pos = current_pose[:3, 3].copy() - prev_pose[:3, 3].copy()

        # note that we set the velocity to be zero in y and z axis.
        vel = np.sqrt((delta_pos.copy() ** 2).sum()) / delta_t
        can_bus[13] = vel

        # acceleration
        acc = (vel - prev_vel) / delta_t
        can_bus[7] = acc
        # set the gravity to be 9.8
        can_bus[9] = 9.8

        # patch angle
        patch_angle = quaternion_yaw(rotation.copy()) / np.pi * 180
        if patch_angle < 0:
            patch_angle += 360
        can_bus[-2] = patch_angle / 180 * np.pi
        can_bus[-1] = patch_angle

        return can_bus


if __name__ == "__main__":
    # config = Config.fromfile("/UniAD/projects/configs/stage2_e2e/inference_e2e.py")
    # model = build_model(config.model, train_cfg=None, test_cfg=config.get("test_cfg"))
    # preproc_pipeline = Compose(config.inference_pipeline)

    # maybe try with this sample token
    # 30e55a3ec6184d8cb1944b39ba19d622
    device = "cuda" if torch.cuda.is_available() else "cpu"
    runner = UniADRunner(
        config_path="/UniAD/projects/configs/stage2_e2e/inference_e2e.py",
        checkpoint_path=None,
        device=torch.device(device),
    )

    # only load this for testing
    from nuscenes.nuscenes import NuScenes
    from nuscenes.can_bus.can_bus_api import NuScenesCanBus
    import matplotlib.pyplot as plt

    # load the first surround-cam in nusc mini
    nusc = NuScenes(version="v1.0-mini", dataroot="/data/nuscenes")
    nusc_can = NuScenesCanBus(dataroot="/data/nuscenes")
    scene = "scene-0061"

    # get the first sample in the scene
    sample_token = nusc.get("sample", scene)["first_sample_token"]
    sample = nusc.get("sample", sample_token)
    timestamp = sample["timestamp"]
    # get the cameras for this sample
    camera_types = [
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_FRONT_LEFT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
    ]

    # ego pose via lidar sensor sample data
    lidar_token = sample["data"]["LIDAR_TOP"]
    lidar_sample_data = nusc.get("sample_data", lidar_token)
    ego_pose = nusc.get("ego_pose", lidar_sample_data["ego_pose_token"])
    ego_translation = np.array(ego_pose["translation"])
    ego_rotation_quat = Quaternion(array=ego_pose["rotation"])
    ego2global = np.eye(4)
    ego2global[:3, 3] = ego_translation
    ego2global[:3, :3] = ego_rotation_quat.rotation_matrix

    # get cameras
    camera_tokens = [sample["data"][camera_type] for camera_type in camera_types]
    cams = [nusc.get_sample_data(cam_token) for cam_token in camera_tokens]
    image_filepaths = [cam[0] for cam in cams]
    cam_instrinsics = [np.array(cam[2]) for cam in cams]
    camera2img = []
    for i in range(len(camera_types)):
        c2i = np.eye(4)
        c2i[:3, :3] = cam_instrinsics[i]
        camera2img.append(c2i)

    # load the images in rgb hwc format
    images = ...

    # get the calibration for each camera
    camera_calibs = [nusc.get("calibrated_sensor", token) for token in camera_tokens]
    # get the lidar calibration
    lidar_calib = nusc.get("calibrated_sensor", sample["data"]["LIDAR_TOP"])

    ego_pose = nusc.get("ego_pose", sample["ego_pose_token"])

    lidar2camera = ...  # fix this

    lidar2img = ...  # fix this

    lidar2ego = ...  # fix this
    lidar2global = lidar2ego @ ego2global

    # get the canbus signals
    canbus_signals = _get_can_bus_info(nusc, nusc_can, sample)

    inference_input = UniADInferenceInput(
        imgs=images,
        pose=lidar2global,
        lidar2img=lidar2img,
        timestamp=timestamp,
        can_bus_signals=canbus_signals,
    )

    plan = runner.forward_inference(inference_input, command=2)
    # plot in bev
    fig, ax = plt.subplots(1, 1)
    # do stuff here
