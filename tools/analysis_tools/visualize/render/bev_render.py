import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pyquaternion import Quaternion
from nuscenes.prediction import PredictHelper, convert_local_coords_to_global
from tools.analysis_tools.visualize.render.base_render import BaseRender
from tools.analysis_tools.visualize.utils import color_mapping, AgentPredictionData


class BEVRender(BaseRender):
    """
    Render class for BEV
    """

    def __init__(self,
                 figsize=(20, 20),
                 margin: float = 50,
                 view: np.ndarray = np.eye(4),
                 show_gt_boxes=False):
        super(BEVRender, self).__init__(figsize)
        self.margin = margin
        self.view = view
        self.show_gt_boxes = show_gt_boxes

    def set_plot_cfg(self):
        self.axes.set_xlim([-self.margin, self.margin])
        self.axes.set_ylim([-self.margin, self.margin])
        self.axes.set_aspect('equal')
        self.axes.grid(False)

    def render_sample_data(self, canvas, sample_token):
        pass

    def render_anno_data(
            self,
            sample_token,
            nusc,
            predict_helper):
        sample_record = nusc.get('sample', sample_token)
        assert 'LIDAR_TOP' in sample_record['data'].keys(
        ), 'Error: No LIDAR_TOP in data, unable to render.'
        lidar_record = sample_record['data']['LIDAR_TOP']
        data_path, boxes, _ = nusc.get_sample_data(
            lidar_record, selected_anntokens=sample_record['anns'])
        for box in boxes:
            instance_token = nusc.get('sample_annotation', box.token)[
                'instance_token']
            future_xy_local = predict_helper.get_future_for_agent(
                instance_token, sample_token, seconds=6, in_agent_frame=True)
            if future_xy_local.shape[0] > 0:
                trans = box.center
                rot = Quaternion(matrix=box.rotation_matrix)
                future_xy = convert_local_coords_to_global(
                    future_xy_local, trans, rot)
                future_xy = np.concatenate(
                    [trans[None, :2], future_xy], axis=0)
                c = np.array([0, 0.8, 0])
                box.render(self.axes, view=self.view, colors=(c, c, c))
                self._render_traj(future_xy, line_color=c, dot_color=(0, 0, 0))
        self.axes.set_xlim([-self.margin, self.margin])
        self.axes.set_ylim([-self.margin, self.margin])

    def show_lidar_data(
            self,
            sample_token,
            nusc):
        sample_record = nusc.get('sample', sample_token)
        assert 'LIDAR_TOP' in sample_record['data'].keys(
        ), 'Error: No LIDAR_TOP in data, unable to render.'
        lidar_record = sample_record['data']['LIDAR_TOP']
        data_path, boxes, _ = nusc.get_sample_data(
            lidar_record, selected_anntokens=sample_record['anns'])
        LidarPointCloud.from_file(data_path).render_height(
            self.axes, view=self.view)
        self.axes.set_xlim([-self.margin, self.margin])
        self.axes.set_ylim([-self.margin, self.margin])
        self.axes.axis('off')
        self.axes.set_aspect('equal')

    def render_pred_box_data(self, agent_prediction_list):
        for pred_agent in agent_prediction_list:
            c = np.array([0, 1, 0])
            if hasattr(pred_agent, 'pred_track_id') and pred_agent.pred_track_id is not None:  # this is true
                tr_id = pred_agent.pred_track_id
                c = color_mapping[tr_id % len(color_mapping)]
            pred_agent.nusc_box.render(
                axis=self.axes, view=self.view, colors=(c, c, c))
            if pred_agent.is_sdc:
                c = np.array([1, 0, 0])
                pred_agent.nusc_box.render(
                    axis=self.axes, view=self.view, colors=(c, c, c))

    def render_pred_traj(self, agent_prediction_list, top_k=3):
        for pred_agent in agent_prediction_list:
            if pred_agent.is_sdc:
                continue
            sorted_ind = np.argsort(pred_agent.pred_traj_score)[
                ::-1]  # from high to low
            num_modes = len(sorted_ind)
            sorted_traj = pred_agent.pred_traj[sorted_ind, :, :2]
            sorted_score = pred_agent.pred_traj_score[sorted_ind]
            # norm_score = np.sum(np.exp(sorted_score))
            norm_score = np.exp(sorted_score[0])

            sorted_traj = np.concatenate(
                [np.zeros((num_modes, 1, 2)), sorted_traj], axis=1)
            trans = pred_agent.pred_center
            rot = Quaternion(axis=np.array([0, 0.0, 1.0]), angle=np.pi/2)
            vehicle_id_list = [0, 1, 2, 3, 4, 6, 7]
            if pred_agent.pred_label in vehicle_id_list:
                dot_size = 150
            else:
                dot_size = 25
            # print(sorted_score)
            for i in range(top_k-1, -1, -1):
                viz_traj = sorted_traj[i, :, :2]
                viz_traj = convert_local_coords_to_global(viz_traj, trans, rot)
                traj_score = np.exp(sorted_score[i])/norm_score
                # traj_score = [1.0, 0.01, 0.01, 0.01, 0.01, 0.01][i]
                self._render_traj(viz_traj, traj_score=traj_score,
                                  colormap='winter', dot_size=dot_size)

    def render_pred_map_data(self, predicted_map_seg):
        # rendered_map = map_color_dict
        # divider, crossing, contour
        map_color_dict = np.array(
            [(204, 128, 0), (102, 255, 102), (102, 255, 102)])
        rendered_map = map_color_dict[predicted_map_seg.argmax(
            -1).reshape(-1)].reshape(200, 200, -1)
        bg_mask = predicted_map_seg.sum(-1) == 0
        rendered_map[bg_mask, :] = 255
        self.axes.imshow(rendered_map, alpha=0.6,
                         interpolation='nearest', extent=(-51.2, 51.2, -51.2, 51.2))

    def render_occ_map_data(self, agent_list):
        rendered_map = np.ones((200, 200, 3))
        rendered_map_hsv = matplotlib.colors.rgb_to_hsv(rendered_map)
        occ_prob_map = np.zeros((200, 200))
        for i in range(len(agent_list)):
            pred_agent = agent_list[i]
            if pred_agent.pred_occ_map is None:
                continue
            if hasattr(pred_agent, 'pred_track_id') and pred_agent.pred_track_id is not None:  # this is true
                tr_id = pred_agent.pred_track_id
                c = color_mapping[tr_id % len(color_mapping)]
            pred_occ_map = pred_agent.pred_occ_map.max(0)
            update_mask = pred_occ_map > occ_prob_map
            occ_prob_map[update_mask] = pred_occ_map[update_mask]
            pred_occ_map *= update_mask
            hsv_c = matplotlib.colors.rgb_to_hsv(c)
            rendered_map_hsv[pred_occ_map > 0.1] = (
                np.ones((200, 200, 1)) * hsv_c)[pred_occ_map > 0.1]
            max_prob = pred_occ_map.max()
            renorm_pred_occ_map = (pred_occ_map - max_prob) * 0.7 + 1
            sat_map = (renorm_pred_occ_map * hsv_c[1])
            rendered_map_hsv[pred_occ_map > 0.1,
                             1] = sat_map[pred_occ_map > 0.1]
            rendered_map = matplotlib.colors.hsv_to_rgb(rendered_map_hsv)
        self.axes.imshow(rendered_map, alpha=0.8,
                         interpolation='nearest', extent=(-50, 50, -50, 50))

    def render_occ_map_data_time(self, agent_list, t):
        rendered_map = np.ones((200, 200, 3))
        rendered_map_hsv = matplotlib.colors.rgb_to_hsv(rendered_map)
        occ_prob_map = np.zeros((200, 200))
        for i in range(len(agent_list)):
            pred_agent = agent_list[i]
            if pred_agent.pred_occ_map is None:
                continue
            if hasattr(pred_agent, 'pred_track_id') and pred_agent.pred_track_id is not None:  # this is true
                tr_id = pred_agent.pred_track_id
                c = color_mapping[tr_id % len(color_mapping)]
            pred_occ_map = pred_agent.pred_occ_map[t]
            update_mask = pred_occ_map > occ_prob_map
            occ_prob_map[update_mask] = pred_occ_map[update_mask]
            pred_occ_map *= update_mask
            hsv_c = matplotlib.colors.rgb_to_hsv(c)
            rendered_map_hsv[pred_occ_map > 0.1] = (
                np.ones((200, 200, 1)) * hsv_c)[pred_occ_map > 0.1]
            max_prob = pred_occ_map.max()
            renorm_pred_occ_map = (pred_occ_map - max_prob) * 0.7 + 1
            sat_map = (renorm_pred_occ_map * hsv_c[1])
            rendered_map_hsv[pred_occ_map > 0.1,
                             1] = sat_map[pred_occ_map > 0.1]
            rendered_map = matplotlib.colors.hsv_to_rgb(rendered_map_hsv)
        self.axes.imshow(rendered_map, alpha=0.8,
                         interpolation='nearest', extent=(-50, 50, -50, 50))

    def render_planning_data(self, predicted_planning, show_command=False):
        planning_traj = predicted_planning.pred_traj
        planning_traj = np.concatenate(
            [np.zeros((1, 2)), planning_traj], axis=0)
        self._render_traj(planning_traj, colormap='autumn', dot_size=50)
        if show_command:
            self._render_command(predicted_planning.command)

    def render_planning_attn_mask(self, predicted_planning):
        planning_attn_mask = predicted_planning.attn_mask
        planning_attn_mask = planning_attn_mask/planning_attn_mask.max()
        cmap_name = 'plasma'
        self.axes.imshow(planning_attn_mask, alpha=0.8, interpolation='nearest', extent=(
            -51.2, 51.2, -51.2, 51.2), vmax=0.2, cmap=matplotlib.colormaps[cmap_name])

    def render_hd_map(self, nusc, nusc_maps, sample_token):
        sample_record = nusc.get('sample', sample_token)
        sd_rec = nusc.get('sample_data', sample_record['data']['LIDAR_TOP'])
        cs_record = nusc.get('calibrated_sensor',
                             sd_rec['calibrated_sensor_token'])
        pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
        info = {
            'lidar2ego_translation': cs_record['translation'],
            'lidar2ego_rotation': cs_record['rotation'],
            'ego2global_translation': pose_record['translation'],
            'ego2global_rotation': pose_record['rotation'],
            'scene_token': sample_record['scene_token']
        }

        layer_names = ['road_divider', 'road_segment', 'lane_divider',
                       'lane',  'road_divider', 'traffic_light', 'ped_crossing']
        map_mask = obtain_map_info(nusc,
                                   nusc_maps,
                                   info,
                                   patch_size=(102.4, 102.4),
                                   canvas_size=(1024, 1024),
                                   layer_names=layer_names)
        map_mask = np.flip(map_mask, axis=1)
        map_mask = np.rot90(map_mask, k=-1, axes=(1, 2))
        map_mask = map_mask[:, ::-1] > 0
        map_show = np.ones((1024, 1024, 3))
        map_show[map_mask[0], :] = np.array([1.00, 0.50, 0.31])
        map_show[map_mask[1], :] = np.array([159./255., 0.0, 1.0])
        self.axes.imshow(map_show, alpha=0.2, interpolation='nearest',
                         extent=(-51.2, 51.2, -51.2, 51.2))

    def _render_traj(self, future_traj, traj_score=1, colormap='winter', points_per_step=20, line_color=None, dot_color=None, dot_size=25):
        total_steps = (len(future_traj)-1) * points_per_step + 1
        dot_colors = matplotlib.colormaps[colormap](
            np.linspace(0, 1, total_steps))[:, :3]
        dot_colors = dot_colors*traj_score + \
            (1-traj_score)*np.ones_like(dot_colors)
        total_xy = np.zeros((total_steps, 2))
        for i in range(total_steps-1):
            unit_vec = future_traj[i//points_per_step +
                                   1] - future_traj[i//points_per_step]
            total_xy[i] = (i/points_per_step - i//points_per_step) * \
                unit_vec + future_traj[i//points_per_step]
        total_xy[-1] = future_traj[-1]
        self.axes.scatter(
            total_xy[:, 0], total_xy[:, 1], c=dot_colors, s=dot_size)

    def _render_command(self, command):
        command_dict = ['TURN RIGHT', 'TURN LEFT', 'KEEP FORWARD']
        self.axes.text(-48, -45, command_dict[int(command)], fontsize=45)

    def render_sdc_car(self):
        sdc_car_png = cv2.imread('sources/sdc_car.png')
        sdc_car_png = cv2.cvtColor(sdc_car_png, cv2.COLOR_BGR2RGB)
        self.axes.imshow(sdc_car_png, extent=(-1, 1, -2, 2))

    def render_legend(self):
        legend = cv2.imread('sources/legend.png')
        legend = cv2.cvtColor(legend, cv2.COLOR_BGR2RGB)
        self.axes.imshow(legend, extent=(23, 51.2, -50, -40))
