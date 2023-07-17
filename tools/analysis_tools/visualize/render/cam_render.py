import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from nuscenes.utils.data_classes import LidarPointCloud, Box
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
from tools.analysis_tools.visualize.utils import color_mapping, AgentPredictionData
from tools.analysis_tools.visualize.render.base_render import BaseRender
from pyquaternion import Quaternion

# Define a constant for camera names
CAM_NAMES = [
    'CAM_FRONT_LEFT',
    'CAM_FRONT',
    'CAM_FRONT_RIGHT',
    'CAM_BACK_RIGHT',
    'CAM_BACK',
    'CAM_BACK_LEFT',
]


class CameraRender(BaseRender):
    """
    Render class for Camera View
    """

    def __init__(self,
                 figsize=(53.3333, 20),
                 show_gt_boxes=False):
        super().__init__(figsize)
        self.cams = CAM_NAMES
        self.show_gt_boxes = show_gt_boxes

    def get_axis(self, index):
        """Retrieve the corresponding axis based on the index."""
        return self.axes[index//3, index % 3]

    def project_to_cam(self,
                       agent_prediction_list,
                       sample_data_token,
                       nusc,
                       lidar_cs_record,
                       project_traj=False,
                       cam=None,
                       ):
        """Project predictions to camera view."""
        _, cs_record, pose_record, cam_intrinsic, imsize = self.get_image_info(
            sample_data_token, nusc)
        boxes = []
        for agent in agent_prediction_list:
            box = Box(agent.pred_center, agent.pred_dim, Quaternion(axis=(0.0, 0.0, 1.0), radians=agent.pred_yaw),
                      name=agent.pred_label, token='predicted')
            box.is_sdc = agent.is_sdc
            if project_traj:
                box.pred_traj = np.zeros((agent.pred_traj_max.shape[0]+1, 3))
                box.pred_traj[:, 0] = agent.pred_center[0]
                box.pred_traj[:, 1] = agent.pred_center[1]
                box.pred_traj[:, 2] = agent.pred_center[2] - \
                    agent.pred_dim[2]/2
                box.pred_traj[1:, :2] += agent.pred_traj_max[:, :2]
                box.pred_traj = (Quaternion(
                    lidar_cs_record['rotation']).rotation_matrix @ box.pred_traj.T).T
                box.pred_traj += np.array(
                    lidar_cs_record['translation'])[None, :]
            box.rotate(Quaternion(lidar_cs_record['rotation']))
            box.translate(np.array(lidar_cs_record['translation']))
            boxes.append(box)
        # Make list of Box objects including coord system transforms.

        box_list = []
        tr_id_list = []
        for i, box in enumerate(boxes):
            #  Move box to sensor coord system.
            box.translate(-np.array(cs_record['translation']))
            box.rotate(Quaternion(cs_record['rotation']).inverse)
            if project_traj:
                box.pred_traj += -np.array(cs_record['translation'])[None, :]
                box.pred_traj = (Quaternion(
                    cs_record['rotation']).inverse.rotation_matrix @ box.pred_traj.T).T

            tr_id = agent_prediction_list[i].pred_track_id
            if box.is_sdc and cam == 'CAM_FRONT':
                box_list.append(box)
            if not box_in_image(box, cam_intrinsic, imsize):
                continue
            box_list.append(box)
            tr_id_list.append(tr_id)
        return box_list, tr_id_list, cam_intrinsic, imsize

    def render_image_data(self, sample_token, nusc):
        """Load and annotate image based on the provided path."""
        sample = nusc.get('sample', sample_token)
        for i, cam in enumerate(self.cams):
            sample_data_token = sample['data'][cam]
            data_path, _, _, _, _ = self.get_image_info(
                sample_data_token, nusc)
            image = self.load_image(data_path, cam)
            self.update_image(image, i, cam)

    def load_image(self, data_path, cam):
        """Update the axis of the plot with the provided image."""
        image = np.array(Image.open(data_path))
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (50, 60)
        fontScale = 2
        color = (0, 0, 0)
        thickness = 4
        return cv2.putText(image, cam, org, font, fontScale, color, thickness, cv2.LINE_AA)

    def update_image(self, image, index, cam):
        """Render image data for each camera."""
        ax = self.get_axis(index)
        ax.imshow(image)
        plt.axis('off')
        ax.axis('off')
        ax.grid(False)

    def render_pred_track_bbox(self, predicted_agent_list, sample_token, nusc):
        """Render bounding box for predicted tracks."""
        sample = nusc.get('sample', sample_token)
        lidar_cs_record = nusc.get('calibrated_sensor', nusc.get(
            'sample_data', sample['data']['LIDAR_TOP'])['calibrated_sensor_token'])
        for i, cam in enumerate(self.cams):
            sample_data_token = sample['data'][cam]
            box_list, tr_id_list, camera_intrinsic, imsize = self.project_to_cam(
                predicted_agent_list, sample_data_token, nusc, lidar_cs_record)
            for j, box in enumerate(box_list):
                if box.is_sdc:
                    continue
                tr_id = tr_id_list[j]
                if tr_id is None:
                    tr_id = 0
                c = color_mapping[tr_id % len(color_mapping)]
                box.render(
                    self.axes[i//3, i % 3], view=camera_intrinsic, normalize=True, colors=(c, c, c))
            # plot gt
            if self.show_gt_boxes:
                data_path, boxes, camera_intrinsic = nusc.get_sample_data(
                    sample_data_token, selected_anntokens=sample['anns'])
                for j, box in enumerate(boxes):
                    c = [0, 1, 0]
                    box.render(
                        self.axes[i//3, i % 3], view=camera_intrinsic, normalize=True, colors=(c, c, c))
            self.axes[i//3, i % 3].set_xlim(0, imsize[0])
            self.axes[i//3, i % 3].set_ylim(imsize[1], 0)

    def render_pred_traj(self, predicted_agent_list, sample_token, nusc, render_sdc=False, points_per_step=10):
        """Render predicted trajectories."""
        sample = nusc.get('sample', sample_token)
        lidar_cs_record = nusc.get('calibrated_sensor', nusc.get(
            'sample_data', sample['data']['LIDAR_TOP'])['calibrated_sensor_token'])
        for i, cam in enumerate(self.cams):
            sample_data_token = sample['data'][cam]
            box_list, tr_id_list, camera_intrinsic, imsize = self.project_to_cam(
                predicted_agent_list, sample_data_token, nusc, lidar_cs_record, project_traj=True, cam=cam)
            for j, box in enumerate(box_list):
                traj_points = box.pred_traj[:, :3]

                total_steps = (len(traj_points)-1) * points_per_step + 1
                total_xy = np.zeros((total_steps, 3))
                for k in range(total_steps-1):
                    unit_vec = traj_points[k//points_per_step +
                                           1] - traj_points[k//points_per_step]
                    total_xy[k] = (k/points_per_step - k//points_per_step) * \
                        unit_vec + traj_points[k//points_per_step]
                in_range_mask = total_xy[:, 2] > 0.1
                traj_points = view_points(
                    total_xy.T, camera_intrinsic, normalize=True)[:2, :]
                traj_points = traj_points[:2, in_range_mask]
                if box.is_sdc:
                    if render_sdc:
                        self.axes[i//3, i % 3].scatter(
                            traj_points[0], traj_points[1], color=(1, 0.5, 0), s=150)
                    else:
                        continue
                else:
                    tr_id = tr_id_list[j]
                    if tr_id is None:
                        tr_id = 0
                    c = color_mapping[tr_id % len(color_mapping)]
                    self.axes[i//3, i %
                              3].scatter(traj_points[0], traj_points[1], color=c, s=15)
            self.axes[i//3, i % 3].set_xlim(0, imsize[0])
            self.axes[i//3, i % 3].set_ylim(imsize[1], 0)

    def get_image_info(self, sample_data_token, nusc):
        """Retrieve image information."""
        sd_record = nusc.get('sample_data', sample_data_token)
        cs_record = nusc.get('calibrated_sensor',
                             sd_record['calibrated_sensor_token'])
        sensor_record = nusc.get('sensor', cs_record['sensor_token'])
        pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

        data_path = nusc.get_sample_data_path(sample_data_token)

        if sensor_record['modality'] == 'camera':
            cam_intrinsic = np.array(cs_record['camera_intrinsic'])
            imsize = (sd_record['width'], sd_record['height'])
        else:
            cam_intrinsic = None
            imsize = None
        return data_path, cs_record, pose_record, cam_intrinsic, imsize
