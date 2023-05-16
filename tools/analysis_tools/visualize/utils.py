import numpy as np
from nuscenes.utils.data_classes import LidarPointCloud, Box
from pyquaternion import Quaternion


color_mapping = np.asarray([
    [0, 0, 0],
    [255, 179, 0],
    [128, 62, 117],
    [255, 104, 0],
    [166, 189, 215],
    [193, 0, 32],
    [206, 162, 98],
    [129, 112, 102],
    [0, 125, 52],
    [246, 118, 142],
    [0, 83, 138],
    [255, 122, 92],
    [83, 55, 122],
    [255, 142, 0],
    [179, 40, 81],
    [244, 200, 0],
    [127, 24, 13],
    [147, 170, 0],
    [89, 51, 21],
    [241, 58, 19],
    [35, 44, 22],
    [112, 224, 255],
    [70, 184, 160],
    [153, 0, 255],
    [71, 255, 0],
    [255, 0, 163],
    [255, 204, 0],
    [0, 255, 235],
    [255, 0, 235],
    [255, 0, 122],
    [255, 245, 0],
    [10, 190, 212],
    [214, 255, 0],
    [0, 204, 255],
    [20, 0, 255],
    [255, 255, 0],
    [0, 153, 255],
    [0, 255, 204],
    [41, 255, 0],
    [173, 0, 255],
    [0, 245, 255],
    [71, 0, 255],
    [0, 255, 184],
    [0, 92, 255],
    [184, 255, 0],
    [255, 214, 0],
    [25, 194, 194],
    [92, 0, 255],
    [220, 220, 220],
    [255, 9, 92],
    [112, 9, 255],
    [8, 255, 214],
    [255, 184, 6],
    [10, 255, 71],
    [255, 41, 10],
    [7, 255, 255],
    [224, 255, 8],
    [102, 8, 255],
    [255, 61, 6],
    [255, 194, 7],
    [0, 255, 20],
    [255, 8, 41],
    [255, 5, 153],
    [6, 51, 255],
    [235, 12, 255],
    [160, 150, 20],
    [0, 163, 255],
    [140, 140, 140],
    [250, 10, 15],
    [20, 255, 0],
])/255


class AgentPredictionData:
    """
    Agent data class, includes bbox, traj, and occflow
    """

    def __init__(self,
                 pred_score,
                 pred_label,
                 pred_center,
                 pred_dim,
                 pred_yaw,
                 pred_vel,
                 pred_traj,
                 pred_traj_score,
                 pred_track_id=None,
                 pred_occ_map=None,
                 is_sdc=False,
                 past_pred_traj=None,
                 command=None,
                 attn_mask=None,
                 ):
        self.pred_score = pred_score
        self.pred_label = pred_label
        self.pred_center = pred_center
        self.pred_dim = pred_dim
        self.pred_yaw = -pred_yaw-np.pi/2
        self.pred_vel = pred_vel
        self.pred_traj = pred_traj
        self.pred_traj_score = pred_traj_score
        self.pred_track_id = pred_track_id
        self.pred_occ_map = pred_occ_map
        if self.pred_traj is not None:
            if isinstance(self.pred_traj_score, int):
                self.pred_traj_max = self.pred_traj
            else:
                self.pred_traj_max = self.pred_traj[self.pred_traj_score.argmax(
                )]
        else:
            self.pred_traj_max = None
        self.nusc_box = Box(
            center=pred_center,
            size=pred_dim,
            orientation=Quaternion(axis=[0, 0, 1], radians=self.pred_yaw),
            label=pred_label,
            score=pred_score
        )
        if is_sdc:
            self.pred_center = [0, 0, -1.2+1.56/2]
        self.is_sdc = is_sdc
        self.past_pred_traj = past_pred_traj
        self.command = command
        self.attn_mask = attn_mask
