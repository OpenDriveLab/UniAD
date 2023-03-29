import numpy as np
from nuscenes.prediction import (PredictHelper,
                                 convert_local_coords_to_global,
                                 convert_global_coords_to_local)
from mmdet3d.core.bbox import Box3DMode, Coord3DMode, LiDARInstance3DBoxes
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.pipelines import to_tensor

class NuScenesTraj(object):
    def __init__(self,
                 nusc,
                 predict_steps,
                 planning_steps,
                 past_steps,
                 fut_steps,
                 with_velocity,
                 CLASSES,
                 box_mode_3d,
                 use_nonlinear_optimizer=False):
        super().__init__()
        self.nusc = nusc
        self.prepare_sdc_vel_info()
        self.predict_steps = predict_steps
        self.planning_steps = planning_steps
        self.past_steps = past_steps
        self.fut_steps = fut_steps
        self.with_velocity = with_velocity
        self.CLASSES = CLASSES
        self.box_mode_3d = box_mode_3d
        self.predict_helper = PredictHelper(self.nusc)
        self.use_nonlinear_optimizer = use_nonlinear_optimizer

    def get_traj_label(self, sample_token, ann_tokens):
        sd_rec = self.nusc.get('sample', sample_token)
        fut_traj_all = []
        fut_traj_valid_mask_all = []
        past_traj_all = []	
        past_traj_valid_mask_all = []
        _, boxes, _ = self.nusc.get_sample_data(sd_rec['data']['LIDAR_TOP'], selected_anntokens=ann_tokens)
        for i, ann_token in enumerate(ann_tokens):
            box = boxes[i]
            instance_token = self.nusc.get('sample_annotation', ann_token)['instance_token']
            fut_traj_local = self.predict_helper.get_future_for_agent(instance_token, sample_token, seconds=6, in_agent_frame=True)
            past_traj_local = self.predict_helper.get_past_for_agent(instance_token, sample_token, seconds=2, in_agent_frame=True)

            fut_traj = np.zeros((self.predict_steps, 2))
            fut_traj_valid_mask = np.zeros((self.predict_steps, 2))
            past_traj = np.zeros((self.past_steps + self.fut_steps, 2))		
            past_traj_valid_mask = np.zeros((self.past_steps + self.fut_steps, 2))
            if fut_traj_local.shape[0] > 0:
                if self.use_nonlinear_optimizer:
                    trans = box.center
                else:
                    trans = np.array([0, 0, 0])
                rot = Quaternion(matrix=box.rotation_matrix)
                fut_traj_scence_centric = convert_local_coords_to_global(fut_traj_local, trans, rot) 
                fut_traj[:fut_traj_scence_centric.shape[0], :] = fut_traj_scence_centric
                fut_traj_valid_mask[:fut_traj_scence_centric.shape[0], :] = 1
            if past_traj_local.shape[0] > 0:			
                trans = np.array([0, 0, 0])		
                rot = Quaternion(matrix=box.rotation_matrix)		
                past_traj_scence_centric = convert_local_coords_to_global(past_traj_local, trans, rot) 		
                past_traj[:past_traj_scence_centric.shape[0], :] = past_traj_scence_centric		
                past_traj_valid_mask[:past_traj_scence_centric.shape[0], :] = 1

                if fut_traj_local.shape[0] > 0:
                    fut_steps = min(self.fut_steps, fut_traj_scence_centric.shape[0])
                    past_traj[self.past_steps:self.past_steps+fut_steps, :] = fut_traj_scence_centric[:fut_steps]
                    past_traj_valid_mask[self.past_steps:self.past_steps+fut_steps, :] = 1

            fut_traj_all.append(fut_traj)		
            fut_traj_valid_mask_all.append(fut_traj_valid_mask)		
            past_traj_all.append(past_traj)		
            past_traj_valid_mask_all.append(past_traj_valid_mask)		
        if len(ann_tokens) > 0:		
            fut_traj_all = np.stack(fut_traj_all, axis=0)		
            fut_traj_valid_mask_all = np.stack(fut_traj_valid_mask_all, axis=0)		
            past_traj_all = np.stack(past_traj_all, axis=0)		
            past_traj_valid_mask_all = np.stack(past_traj_valid_mask_all, axis=0)		
        else:		
            fut_traj_all = np.zeros((0, self.predict_steps, 2))		
            fut_traj_valid_mask_all = np.zeros((0, self.predict_steps, 2))		
            past_traj_all = np.zeros((0, self.predict_steps, 2))		
            past_traj_valid_mask_all = np.zeros((0, self.predict_steps, 2))		
        return fut_traj_all, fut_traj_valid_mask_all, past_traj_all, past_traj_valid_mask_all

    def get_vel_transform_mats(self, sample):
        sd_rec = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        cs_record = self.nusc.get('calibrated_sensor',
                             sd_rec['calibrated_sensor_token'])
        pose_record = self.nusc.get('ego_pose', sd_rec['ego_pose_token'])

        l2e_r = cs_record['rotation']
        l2e_t = cs_record['translation']
        e2g_r = pose_record['rotation']
        e2g_t = pose_record['translation']
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        return l2e_r_mat, e2g_r_mat

    def get_vel_and_time(self, sample):
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_top = self.nusc.get('sample_data', lidar_token)
        pose = self.nusc.get('ego_pose', lidar_top['ego_pose_token'])
        xyz = pose['translation']
        timestamp = sample['timestamp']
        return xyz, timestamp
        
    def prepare_sdc_vel_info(self):
        # generate sdc velocity info for all samples
        # Note that these velocity values are converted from 
        # global frame to lidar frame
        # as aligned with bbox gts

        self.sdc_vel_info = {}
        for scene in self.nusc.scene:
            scene_token = scene['token']

            # we cannot infer vel for the last sample, therefore we skip it
            last_sample_token = scene['last_sample_token']
            sample_token = scene['first_sample_token']
            sample = self.nusc.get('sample', sample_token)
            xyz, time = self.get_vel_and_time(sample)
            while sample['token'] != last_sample_token:
                next_sample_token = sample['next']
                next_sample = self.nusc.get('sample', next_sample_token)
                next_xyz, next_time = self.get_vel_and_time(next_sample)
                dc = np.array(next_xyz) - np.array(xyz) 
                dt = (next_time - time) / 1e6
                vel = dc/dt

                # global frame to lidar frame
                l2e_r_mat, e2g_r_mat = self.get_vel_transform_mats(sample)
                vel = vel @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(
                    l2e_r_mat).T
                vel = vel[:2]

                self.sdc_vel_info[sample['token']] = vel
                xyz, time = next_xyz, next_time
                sample = next_sample

            # set first sample's vel equal to second sample's
            last_sample = self.nusc.get('sample', last_sample_token)
            second_last_sample_token = last_sample['prev']
            self.sdc_vel_info[last_sample_token] = self.sdc_vel_info[second_last_sample_token]                

    def generate_sdc_info(self, sdc_vel, as_lidar_instance3d_box=False):
        # sdc dim from https://forum.nuscenes.org/t/dimensions-of-the-ego-vehicle-used-to-gather-data/550
        psudo_sdc_bbox = np.array([0.0, 0.0, 0.0, 1.73, 4.08, 1.56, -np.pi])
        if self.with_velocity:
            psudo_sdc_bbox = np.concatenate([psudo_sdc_bbox, sdc_vel], axis=-1)
        gt_bboxes_3d = np.array([psudo_sdc_bbox]).astype(np.float32)
        gt_names_3d = ['car']
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)
        
        if as_lidar_instance3d_box:
            # if we do not want the batch the box in to DataContrainer
            return gt_bboxes_3d

        gt_labels_3d = DC(to_tensor(gt_labels_3d))
        gt_bboxes_3d = DC(gt_bboxes_3d, cpu_only=True)

        return gt_bboxes_3d, gt_labels_3d

    def get_sdc_traj_label(self, sample_token):
        sd_rec = self.nusc.get('sample', sample_token)
        lidar_top_data_start = self.nusc.get('sample_data', sd_rec['data']['LIDAR_TOP'])
        ego_pose_start = self.nusc.get('ego_pose', lidar_top_data_start['ego_pose_token'])

        sdc_fut_traj = []
        for _ in range(self.predict_steps):
            next_annotation_token = sd_rec['next']
            if next_annotation_token=='':
                break
            sd_rec = self.nusc.get('sample', next_annotation_token)
            lidar_top_data_next = self.nusc.get('sample_data', sd_rec['data']['LIDAR_TOP'])
            ego_pose_next = self.nusc.get('ego_pose', lidar_top_data_next['ego_pose_token'])
            sdc_fut_traj.append(ego_pose_next['translation'][:2])  # global xy pos of sdc at future step i
        
        sdc_fut_traj_all = np.zeros((1, self.predict_steps, 2))
        sdc_fut_traj_valid_mask_all = np.zeros((1, self.predict_steps, 2))
        n_valid_timestep = len(sdc_fut_traj)
        if n_valid_timestep>0:
            sdc_fut_traj = np.stack(sdc_fut_traj, axis=0)  #(t,2)
            sdc_fut_traj = convert_global_coords_to_local(
                coordinates=sdc_fut_traj,
                translation=ego_pose_start['translation'],
                rotation=ego_pose_start['rotation'],
            )
            sdc_fut_traj_all[:,:n_valid_timestep,:] = sdc_fut_traj
            sdc_fut_traj_valid_mask_all[:,:n_valid_timestep,:] = 1

        return sdc_fut_traj_all, sdc_fut_traj_valid_mask_all
    
    def get_l2g_transform(self, sample):
        sd_rec = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        cs_record = self.nusc.get('calibrated_sensor',
                             sd_rec['calibrated_sensor_token'])
        pose_record = self.nusc.get('ego_pose', sd_rec['ego_pose_token'])

        l2e_r = cs_record['rotation']
        l2e_t = np.array(cs_record['translation'])
        e2g_r = pose_record['rotation']
        e2g_t = np.array(pose_record['translation'])
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        return l2e_r_mat, l2e_t, e2g_r_mat, e2g_t

    def get_sdc_planning_label(self, sample_token):
        sd_rec = self.nusc.get('sample', sample_token)
        l2e_r_mat_init, l2e_t_init, e2g_r_mat_init, e2g_t_init = self.get_l2g_transform(sd_rec)
        

        planning = []
        for _ in range(self.planning_steps):
            next_annotation_token = sd_rec['next']
            if next_annotation_token=='':
                break
            sd_rec = self.nusc.get('sample', next_annotation_token)
            l2e_r_mat_curr, l2e_t_curr, e2g_r_mat_curr, e2g_t_curr = self.get_l2g_transform(sd_rec)  # (lidar to global at current frame)
            
            # bbox of sdc under current lidar frame
            next_bbox3d = self.generate_sdc_info(self.sdc_vel_info[next_annotation_token], as_lidar_instance3d_box=True)

            # to bbox under curr ego frame
            next_bbox3d.rotate(l2e_r_mat_curr.T)
            next_bbox3d.translate(l2e_t_curr)

            # to bbox under world frame
            next_bbox3d.rotate(e2g_r_mat_curr.T)
            next_bbox3d.translate(e2g_t_curr)

            # to bbox under initial ego frame, first inverse translate, then inverse rotate 
            next_bbox3d.translate(- e2g_t_init)
            m1 = np.linalg.inv(e2g_r_mat_init)
            next_bbox3d.rotate(m1.T)

            # to bbox under curr ego frame, first inverse translate, then inverse rotate
            next_bbox3d.translate(- l2e_t_init)
            m2 = np.linalg.inv(l2e_r_mat_init)
            next_bbox3d.rotate(m2.T)
            
            planning.append(next_bbox3d)

        planning_all = np.zeros((1, self.planning_steps, 3))
        planning_mask_all = np.zeros((1, self.planning_steps, 2))
        n_valid_timestep = len(planning)
        if n_valid_timestep>0:
            planning = [p.tensor.squeeze(0) for p in planning]
            planning = np.stack(planning, axis=0)  # (valid_t, 9)
            planning = planning[:, [0,1,6]]  # (x, y, yaw)
            planning_all[:,:n_valid_timestep,:] = planning
            planning_mask_all[:,:n_valid_timestep,:] = 1

        mask = planning_mask_all[0].any(axis=1)
        if mask.sum() == 0:
            command = 2 #'FORWARD'
        elif planning_all[0, mask][-1][0] >= 2:
            command = 0 #'RIGHT' 
        elif planning_all[0, mask][-1][0] <= -2:
            command = 1 #'LEFT'
        else:
            command = 2 #'FORWARD'
        
        return planning_all, planning_mask_all, command