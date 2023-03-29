import argparse
import copy
import json
import numpy as np
import os
import time
from typing import Tuple, Dict, Any
import tqdm
from matplotlib import pyplot as plt
from pyquaternion import Quaternion

from nuscenes import NuScenes
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.loaders import load_prediction, load_gt, add_center_dist, filter_eval_boxes
from nuscenes.eval.common.render import setup_axis
from nuscenes.eval.detection.algo import accumulate, calc_ap, calc_tp
from nuscenes.eval.detection.constants import TP_METRICS, TP_METRICS_UNITS, PRETTY_DETECTION_NAMES, PRETTY_TP_METRICS
from nuscenes.eval.detection.data_classes import DetectionConfig, DetectionMetrics, DetectionBox, DetectionMetricDataList
from nuscenes.eval.detection.evaluate import NuScenesEval
from nuscenes.eval.detection.render import summary_plot, class_pr_curve, dist_pr_curve
from nuscenes.eval.tracking.data_classes import TrackingBox
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points, BoxVisibility
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.eval.detection.utils import category_to_detection_name


Axis = Any

def class_tp_curve(md_list: DetectionMetricDataList,
                   metrics: DetectionMetrics,
                   detection_name: str,
                   min_recall: float,
                   dist_th_tp: float,
                   savepath: str = None,
                   ax: Axis = None) -> None:
    """
    Plot the true positive curve for the specified class.
    :param md_list: DetectionMetricDataList instance.
    :param metrics: DetectionMetrics instance.
    :param detection_name:
    :param min_recall: Minimum recall value.
    :param dist_th_tp: The distance threshold used to determine matches.
    :param savepath: If given, saves the the rendering here instead of displaying.
    :param ax: Axes onto which to render.
    """
    # Get metric data for given detection class with tp distance threshold.

    md = md_list[(detection_name, dist_th_tp)]
    min_recall_ind = round(100 * min_recall)
    if min_recall_ind <= md.max_recall_ind:
        # For traffic_cone and barrier only a subset of the metrics are plotted.
        rel_metrics = [m for m in TP_METRICS if not np.isnan(metrics.get_label_tp(detection_name, m))]
        ylimit = max([max(getattr(md, metric)[min_recall_ind:md.max_recall_ind + 1]) for metric in rel_metrics]) * 1.1
    else:
        ylimit = 1.0

    # Prepare axis.
    if ax is None:
        ax = setup_axis(title=PRETTY_DETECTION_NAMES[detection_name], xlabel='Recall', ylabel='Error', xlim=1,
                        min_recall=min_recall)
    ax.set_ylim(0, ylimit)

    # Plot the recall vs. error curve for each tp metric.
    for metric in TP_METRICS:
        tp = metrics.get_label_tp(detection_name, metric)

        # Plot only if we have valid data.
        if tp is not np.nan and min_recall_ind <= md.max_recall_ind:
            recall, error = md.recall[:md.max_recall_ind + 1], getattr(md, metric)[:md.max_recall_ind + 1]
        else:
            recall, error = [], []

        # Change legend based on tp value
        if tp is np.nan:
            label = '{}: n/a'.format(PRETTY_TP_METRICS[metric])
        elif min_recall_ind > md.max_recall_ind:
            label = '{}: nan'.format(PRETTY_TP_METRICS[metric])
        else:
            label = '{}: {:.2f} ({})'.format(PRETTY_TP_METRICS[metric], tp, TP_METRICS_UNITS[metric])
        if metric == 'trans_err':
            label += f' ({md.max_recall_ind})'  # add recall
            print(f'Recall: {detection_name}: {md.max_recall_ind/100}')
        ax.plot(recall, error, label=label)
    ax.axvline(x=md.max_recall, linestyle='-.', color=(0, 0, 0, 0.3))
    ax.legend(loc='best')

    if savepath is not None:
        plt.savefig(savepath)
        plt.close()


class DetectionBox_modified(DetectionBox):
    def __init__(self, *args, token=None, visibility=None, index=None, **kwargs):
        '''
        add annotation token
        '''
        super().__init__(*args, **kwargs)
        self.token = token
        self.visibility = visibility
        self.index = index

    def serialize(self) -> dict:
        """ Serialize instance into json-friendly format. """
        return {
            'token': self.token,
            'sample_token': self.sample_token,
            'translation': self.translation,
            'size': self.size,
            'rotation': self.rotation,
            'velocity': self.velocity,
            'ego_translation': self.ego_translation,
            'num_pts': self.num_pts,
            'detection_name': self.detection_name,
            'detection_score': self.detection_score,
            'attribute_name': self.attribute_name,
            'visibility': self.visibility,
            'index': self.index

        }

    @classmethod
    def deserialize(cls, content: dict):
        """ Initialize from serialized content. """
        return cls(
            token=content['token'],
            sample_token=content['sample_token'],
            translation=tuple(content['translation']),
            size=tuple(content['size']),
            rotation=tuple(content['rotation']),
            velocity=tuple(content['velocity']),
            ego_translation=(0.0, 0.0, 0.0) if 'ego_translation' not in content
            else tuple(content['ego_translation']),
            num_pts=-1 if 'num_pts' not in content else int(content['num_pts']),
            detection_name=content['detection_name'],
            detection_score=-1.0 if 'detection_score' not in content else float(content['detection_score']),
            attribute_name=content['attribute_name'],
            visibility=content['visibility'],
            index=content['index'],
        )


def center_in_image(box, intrinsic: np.ndarray, imsize: Tuple[int, int], vis_level: int = BoxVisibility.ANY) -> bool:
    """
    Check if a box is visible inside an image without accounting for occlusions.
    :param box: The box to be checked.
    :param intrinsic: <float: 3, 3>. Intrinsic camera matrix.
    :param imsize: (width, height).
    :param vis_level: One of the enumerations of <BoxVisibility>.
    :return True if visibility condition is satisfied.
    """

    center_3d = box.center.reshape(3, 1)
    center_img = view_points(center_3d, intrinsic, normalize=True)[:2, :]

    visible = np.logical_and(center_img[0, :] > 0, center_img[0, :] < imsize[0])
    visible = np.logical_and(visible, center_img[1, :] < imsize[1])
    visible = np.logical_and(visible, center_img[1, :] > 0)
    visible = np.logical_and(visible, center_3d[2, :] > 1)

    in_front = center_3d[2, :] > 0.1  # True if a corner is at least 0.1 meter in front of the camera.

    if vis_level == BoxVisibility.ALL:
        return all(visible) and all(in_front)
    elif vis_level == BoxVisibility.ANY:
        return any(visible) and all(in_front)
    elif vis_level == BoxVisibility.NONE:
        return True
    else:
        raise ValueError("vis_level: {} not valid".format(vis_level))


def exist_corners_in_image_but_not_all(box, intrinsic: np.ndarray, imsize: Tuple[int, int],
                                       vis_level: int = BoxVisibility.ANY) -> bool:
    """
    Check if a box is visible in images but not all corners in image .
    :param box: The box to be checked.
    :param intrinsic: <float: 3, 3>. Intrinsic camera matrix.
    :param imsize: (width, height).
    :param vis_level: One of the enumerations of <BoxVisibility>.
    :return True if visibility condition is satisfied.
    """

    corners_3d = box.corners()
    corners_img = view_points(corners_3d, intrinsic, normalize=True)[:2, :]

    visible = np.logical_and(corners_img[0, :] > 0, corners_img[0, :] < imsize[0])
    visible = np.logical_and(visible, corners_img[1, :] < imsize[1])
    visible = np.logical_and(visible, corners_img[1, :] > 0)
    visible = np.logical_and(visible, corners_3d[2, :] > 1)

    in_front = corners_3d[2, :] > 0.1  # True if a corner is at least 0.1 meter in front of the camera.

    if any(visible) and not all(visible) and all(in_front):
        return True
    else:
        return False


def load_gt(nusc: NuScenes, eval_split: str, box_cls, verbose: bool = False):
    """
    Loads ground truth boxes from DB.
    :param nusc: A NuScenes instance.
    :param eval_split: The evaluation split for which we load GT boxes.
    :param box_cls: Type of box to load, e.g. DetectionBox or TrackingBox.
    :param verbose: Whether to print messages to stdout.
    :return: The GT boxes.
    """

    # Init.
    if box_cls == DetectionBox_modified:
        attribute_map = {a['token']: a['name'] for a in nusc.attribute}

    if verbose:
        print('Loading annotations for {} split from nuScenes version: {}'.format(eval_split, nusc.version))
    # Read out all sample_tokens in DB.
    sample_tokens_all = [s['token'] for s in nusc.sample]
    assert len(sample_tokens_all) > 0, "Error: Database has no samples!"

    # Only keep samples from this split.
    splits = create_splits_scenes()

    # Check compatibility of split with nusc_version.
    version = nusc.version
    if eval_split in {'train', 'val', 'train_detect', 'train_track'}:
        assert version.endswith('trainval'), \
            'Error: Requested split {} which is not compatible with NuScenes version {}'.format(eval_split, version)
    elif eval_split in {'mini_train', 'mini_val'}:
        assert version.endswith('mini'), \
            'Error: Requested split {} which is not compatible with NuScenes version {}'.format(eval_split, version)
    elif eval_split == 'test':
        assert version.endswith('test'), \
            'Error: Requested split {} which is not compatible with NuScenes version {}'.format(eval_split, version)
    else:
        raise ValueError('Error: Requested split {} which this function cannot map to the correct NuScenes version.'
                         .format(eval_split))

    if eval_split == 'test':
        # Check that you aren't trying to cheat :).
        assert len(nusc.sample_annotation) > 0, \
            'Error: You are trying to evaluate on the test set but you do not have the annotations!'
    index_map = {}
    for scene in nusc.scene:
        first_sample_token = scene['first_sample_token']
        sample = nusc.get('sample', first_sample_token)
        index_map[first_sample_token] = 1
        index = 2
        while sample['next'] != '':
            sample = nusc.get('sample', sample['next'])
            index_map[sample['token']] = index
            index += 1

    sample_tokens = []
    for sample_token in sample_tokens_all:
        scene_token = nusc.get('sample', sample_token)['scene_token']
        scene_record = nusc.get('scene', scene_token)
        if scene_record['name'] in splits[eval_split]:
            sample_tokens.append(sample_token)

    all_annotations = EvalBoxes()

    # Load annotations and filter predictions and annotations.
    tracking_id_set = set()
    for sample_token in tqdm.tqdm(sample_tokens, leave=verbose):

        sample = nusc.get('sample', sample_token)
        sample_annotation_tokens = sample['anns']

        sample_boxes = []
        for sample_annotation_token in sample_annotation_tokens:

            sample_annotation = nusc.get('sample_annotation', sample_annotation_token)
            if box_cls == DetectionBox_modified:
                # Get label name in detection task and filter unused labels.
                detection_name = category_to_detection_name(sample_annotation['category_name'])
                if detection_name is None:
                    continue

                # Get attribute_name.
                attr_tokens = sample_annotation['attribute_tokens']
                attr_count = len(attr_tokens)
                if attr_count == 0:
                    attribute_name = ''
                elif attr_count == 1:
                    attribute_name = attribute_map[attr_tokens[0]]
                else:
                    raise Exception('Error: GT annotations must not have more than one attribute!')

                sample_boxes.append(
                    box_cls(
                        token=sample_annotation_token,
                        sample_token=sample_token,
                        translation=sample_annotation['translation'],
                        size=sample_annotation['size'],
                        rotation=sample_annotation['rotation'],
                        velocity=nusc.box_velocity(sample_annotation['token'])[:2],
                        num_pts=sample_annotation['num_lidar_pts'] + sample_annotation['num_radar_pts'],
                        detection_name=detection_name,
                        detection_score=-1.0,  # GT samples do not have a score.
                        attribute_name=attribute_name,
                        visibility=sample_annotation['visibility_token'],
                        index=index_map[sample_token]
                    )
                )
            elif box_cls == TrackingBox:
                assert False
            else:
                raise NotImplementedError('Error: Invalid box_cls %s!' % box_cls)

        all_annotations.add_boxes(sample_token, sample_boxes)

    if verbose:
        print("Loaded ground truth annotations for {} samples.".format(len(all_annotations.sample_tokens)))

    return all_annotations


def filter_eval_boxes_by_id(nusc: NuScenes,
                            eval_boxes: EvalBoxes,
                            id=None,
                            verbose: bool = False) -> EvalBoxes:
    """
    Applies filtering to boxes. Distance, bike-racks and points per box.
    :param nusc: An instance of the NuScenes class.
    :param eval_boxes: An instance of the EvalBoxes class.
    :param is: the anns token set that used to keep bboxes.
    :param verbose: Whether to print to stdout.
    """

    # Accumulators for number of filtered boxes.
    total, anns_filter = 0, 0
    for ind, sample_token in enumerate(eval_boxes.sample_tokens):

        # Filter on anns
        total += len(eval_boxes[sample_token])
        filtered_boxes = []
        for box in eval_boxes[sample_token]:
            if box.token in id:
                filtered_boxes.append(box)
        anns_filter += len(filtered_boxes)
        eval_boxes.boxes[sample_token] = filtered_boxes

    if verbose:
        print("=> Original number of boxes: %d" % total)
        print("=> After anns based filtering: %d" % anns_filter)

    return eval_boxes


def filter_eval_boxes_by_visibility(
        ori_eval_boxes: EvalBoxes,
        visibility=None,
        verbose: bool = False) -> EvalBoxes:
    """
    Applies filtering to boxes. Distance, bike-racks and points per box.
    :param nusc: An instance of the NuScenes class.
    :param eval_boxes: An instance of the EvalBoxes class.
    :param is: the anns token set that used to keep bboxes.
    :param verbose: Whether to print to stdout.
    """

    # Accumulators for number of filtered boxes.
    eval_boxes = copy.deepcopy(ori_eval_boxes)
    total, anns_filter = 0, 0
    for ind, sample_token in enumerate(eval_boxes.sample_tokens):
        # Filter on anns
        total += len(eval_boxes[sample_token])
        filtered_boxes = []
        for box in eval_boxes[sample_token]:
            if box.visibility == visibility:
                filtered_boxes.append(box)
        anns_filter += len(filtered_boxes)
        eval_boxes.boxes[sample_token] = filtered_boxes

    if verbose:
        print("=> Original number of boxes: %d" % total)
        print("=> After visibility based filtering: %d" % anns_filter)

    return eval_boxes


def filter_by_sample_token(ori_eval_boxes, valid_sample_tokens=[],  verbose=False):
    eval_boxes = copy.deepcopy(ori_eval_boxes)
    for sample_token in eval_boxes.sample_tokens:
        if sample_token not in valid_sample_tokens:
            eval_boxes.boxes.pop(sample_token)
    return eval_boxes


def filter_eval_boxes_by_overlap(nusc: NuScenes,
                                 eval_boxes: EvalBoxes,
                                 verbose: bool = False) -> EvalBoxes:
    """
    Applies filtering to boxes. basedon overlap .
    :param nusc: An instance of the NuScenes class.
    :param eval_boxes: An instance of the EvalBoxes class.
    :param verbose: Whether to print to stdout.
    """

    # Accumulators for number of filtered boxes.
    cams = ['CAM_FRONT',
            'CAM_FRONT_RIGHT',
            'CAM_BACK_RIGHT',
            'CAM_BACK',
            'CAM_BACK_LEFT',
            'CAM_FRONT_LEFT']

    total, anns_filter = 0, 0
    for ind, sample_token in enumerate(eval_boxes.sample_tokens):

        # Filter on anns
        total += len(eval_boxes[sample_token])
        sample_record = nusc.get('sample', sample_token)
        filtered_boxes = []
        for box in eval_boxes[sample_token]:
            count = 0
            for cam in cams:
                '''
                copy-paste form nuscens
                '''
                sample_data_token = sample_record['data'][cam]
                sd_record = nusc.get('sample_data', sample_data_token)
                cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
                sensor_record = nusc.get('sensor', cs_record['sensor_token'])
                pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])
                cam_intrinsic = np.array(cs_record['camera_intrinsic'])
                imsize = (sd_record['width'], sd_record['height'])
                new_box = Box(box.translation, box.size, Quaternion(box.rotation),
                              name=box.detection_name, token='')

                # Move box to ego vehicle coord system.
                new_box.translate(-np.array(pose_record['translation']))
                new_box.rotate(Quaternion(pose_record['rotation']).inverse)

                #  Move box to sensor coord system.
                new_box.translate(-np.array(cs_record['translation']))
                new_box.rotate(Quaternion(cs_record['rotation']).inverse)

                if center_in_image(new_box, cam_intrinsic, imsize, vis_level=BoxVisibility.ANY):
                    count += 1
                # if exist_corners_in_image_but_not_all(new_box, cam_intrinsic, imsize, vis_level=BoxVisibility.ANY):
                #    count += 1

            if count > 1:
                with open('center_overlap.txt', 'a') as f:
                    try:
                        f.write(box.token + '\n')
                    except:
                        pass
                filtered_boxes.append(box)
        anns_filter += len(filtered_boxes)
        eval_boxes.boxes[sample_token] = filtered_boxes

    verbose = True

    if verbose:
        print("=> Original number of boxes: %d" % total)
        print("=> After anns based filtering: %d" % anns_filter)

    return eval_boxes


class NuScenesEval_custom(NuScenesEval):
    """
    Dummy class for backward-compatibility. Same as DetectionEval.
    """

    def __init__(self,
                 nusc: NuScenes,
                 config: DetectionConfig,
                 result_path: str,
                 eval_set: str,
                 output_dir: str = None,
                 verbose: bool = True,
                 overlap_test=False,
                 eval_mask=False,
                 data_infos=None
                 ):
        """
        Initialize a DetectionEval object.
        :param nusc: A NuScenes object.
        :param config: A DetectionConfig object.
        :param result_path: Path of the nuScenes JSON result file.
        :param eval_set: The dataset split to evaluate on, e.g. train, val or test.
        :param output_dir: Folder to save plots and results to.
        :param verbose: Whether to print to stdout.
        """

        self.nusc = nusc
        self.result_path = result_path
        self.eval_set = eval_set
        self.output_dir = output_dir
        self.verbose = verbose
        self.cfg = config
        self.overlap_test = overlap_test
        self.eval_mask = eval_mask
        self.data_infos = data_infos
        # Check result file exists.
        assert os.path.exists(result_path), 'Error: The result file does not exist!'

        # Make dirs.
        self.plot_dir = os.path.join(self.output_dir, 'plots')
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.isdir(self.plot_dir):
            os.makedirs(self.plot_dir)

        # Load data.
        if verbose:
            print('Initializing nuScenes detection evaluation')
        self.pred_boxes, self.meta = load_prediction(self.result_path, self.cfg.max_boxes_per_sample, DetectionBox,
                                                     verbose=verbose)
        self.gt_boxes = load_gt(self.nusc, self.eval_set, DetectionBox_modified, verbose=verbose)

        assert set(self.pred_boxes.sample_tokens) == set(self.gt_boxes.sample_tokens), \
            "Samples in split doesn't match samples in predictions."

        # Add center distances.
        self.pred_boxes = add_center_dist(nusc, self.pred_boxes)
        self.gt_boxes = add_center_dist(nusc, self.gt_boxes)

        # Filter boxes (distance, points per box, etc.).

        if verbose:
            print('Filtering predictions')
        self.pred_boxes = filter_eval_boxes(nusc, self.pred_boxes, self.cfg.class_range, verbose=verbose)
        if verbose:
            print('Filtering ground truth annotations')
        self.gt_boxes = filter_eval_boxes(nusc, self.gt_boxes, self.cfg.class_range, verbose=verbose)

        if self.overlap_test:
            self.pred_boxes = filter_eval_boxes_by_overlap(self.nusc, self.pred_boxes)

            self.gt_boxes = filter_eval_boxes_by_overlap(self.nusc, self.gt_boxes, verbose=True)

        self.all_gt = copy.deepcopy(self.gt_boxes)
        self.all_preds = copy.deepcopy(self.pred_boxes)
        self.sample_tokens = self.gt_boxes.sample_tokens

        self.index_map = {}
        for scene in nusc.scene:
            first_sample_token = scene['first_sample_token']
            sample = nusc.get('sample', first_sample_token)
            self.index_map[first_sample_token] = 1
            index = 2
            while sample['next'] != '':
                sample = nusc.get('sample', sample['next'])
                self.index_map[sample['token']] = index
                index += 1

    def update_gt(self, type_='vis', visibility='1', index=1):
        if type_ == 'vis':
            self.visibility_test = True
            if self.visibility_test:
                '''[{'description': 'visibility of whole object is between 0 and 40%',
                'token': '1',
                'level': 'v0-40'},
                {'description': 'visibility of whole object is between 40 and 60%',
                'token': '2',
                'level': 'v40-60'},
                {'description': 'visibility of whole object is between 60 and 80%',
                'token': '3',
                'level': 'v60-80'},
                {'description': 'visibility of whole object is between 80 and 100%',
                'token': '4',
                'level': 'v80-100'}]'''

                self.gt_boxes = filter_eval_boxes_by_visibility(self.all_gt, visibility, verbose=True)

        elif type_ == 'ord':

            valid_tokens = [key for (key, value) in self.index_map.items() if value == index]
            # from IPython import embed
            # embed()
            self.gt_boxes = filter_by_sample_token(self.all_gt, valid_tokens)
            self.pred_boxes = filter_by_sample_token(self.all_preds, valid_tokens)
        self.sample_tokens = self.gt_boxes.sample_tokens


    def evaluate(self) -> Tuple[DetectionMetrics, DetectionMetricDataList]:
        """
        Performs the actual evaluation.
        :return: A tuple of high-level and the raw metric data.
        """
        start_time = time.time()

        # -----------------------------------
        # Step 1: Accumulate metric data for all classes and distance thresholds.
        # -----------------------------------
        if self.verbose:
            print('Accumulating metric data...')
        metric_data_list = DetectionMetricDataList()

        # print(self.cfg.dist_fcn_callable, self.cfg.dist_ths)
        # self.cfg.dist_ths = [0.3]
        # self.cfg.dist_fcn_callable
        for class_name in self.cfg.class_names:
            for dist_th in self.cfg.dist_ths:
                md = accumulate(self.gt_boxes, self.pred_boxes, class_name, self.cfg.dist_fcn_callable, dist_th)
                metric_data_list.set(class_name, dist_th, md)

        # -----------------------------------
        # Step 2: Calculate metrics from the data.
        # -----------------------------------
        if self.verbose:
            print('Calculating metrics...')
        metrics = DetectionMetrics(self.cfg)
        for class_name in self.cfg.class_names:
            # Compute APs.
            for dist_th in self.cfg.dist_ths:
                metric_data = metric_data_list[(class_name, dist_th)]
                ap = calc_ap(metric_data, self.cfg.min_recall, self.cfg.min_precision)
                metrics.add_label_ap(class_name, dist_th, ap)
            # Compute TP metrics.
            for metric_name in TP_METRICS:
                metric_data = metric_data_list[(class_name, self.cfg.dist_th_tp)]
                if class_name in ['traffic_cone'] and metric_name in ['attr_err', 'vel_err', 'orient_err']:
                    tp = np.nan
                elif class_name in ['barrier'] and metric_name in ['attr_err', 'vel_err']:
                    tp = np.nan
                else:
                    tp = calc_tp(metric_data, self.cfg.min_recall, metric_name)
                metrics.add_label_tp(class_name, metric_name, tp)

        # Compute evaluation time.
        metrics.add_runtime(time.time() - start_time)

        return metrics, metric_data_list

    def render(self, metrics: DetectionMetrics, md_list: DetectionMetricDataList) -> None:
        """
        Renders various PR and TP curves.
        :param metrics: DetectionMetrics instance.
        :param md_list: DetectionMetricDataList instance.
        """
        if self.verbose:
            print('Rendering PR and TP curves')

        def savepath(name):
            return os.path.join(self.plot_dir, name + '.pdf')

        summary_plot(md_list, metrics, min_precision=self.cfg.min_precision, min_recall=self.cfg.min_recall,
                     dist_th_tp=self.cfg.dist_th_tp, savepath=savepath('summary'))

        for detection_name in self.cfg.class_names:
            class_pr_curve(md_list, metrics, detection_name, self.cfg.min_precision, self.cfg.min_recall,
                           savepath=savepath(detection_name + '_pr'))

            class_tp_curve(md_list, metrics, detection_name, self.cfg.min_recall, self.cfg.dist_th_tp,
                           savepath=savepath(detection_name + '_tp'))

        for dist_th in self.cfg.dist_ths:
            dist_pr_curve(md_list, metrics, dist_th, self.cfg.min_precision, self.cfg.min_recall,
                          savepath=savepath('dist_pr_' + str(dist_th)))


if __name__ == "__main__":

    # Settings.
    parser = argparse.ArgumentParser(description='Evaluate nuScenes detection results.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('result_path', type=str, help='The submission as a JSON file.')
    parser.add_argument('--output_dir', type=str, default='~/nuscenes-metrics',
                        help='Folder to store result metrics, graphs and example visualizations.')
    parser.add_argument('--eval_set', type=str, default='val',
                        help='Which dataset split to evaluate on, train, val or test.')
    parser.add_argument('--dataroot', type=str, default='data/nuscenes',
                        help='Default nuScenes data directory.')
    parser.add_argument('--version', type=str, default='v1.0-trainval',
                        help='Which version of the nuScenes dataset to evaluate on, e.g. v1.0-trainval.')
    parser.add_argument('--config_path', type=str, default='',
                        help='Path to the configuration file.'
                             'If no path given, the CVPR 2019 configuration will be used.')
    parser.add_argument('--plot_examples', type=int, default=0,
                        help='How many example visualizations to write to disk.')
    parser.add_argument('--render_curves', type=int, default=1,
                        help='Whether to render PR and TP curves to disk.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Whether to print to stdout.')
    args = parser.parse_args()

    result_path_ = os.path.expanduser(args.result_path)
    output_dir_ = os.path.expanduser(args.output_dir)
    eval_set_ = args.eval_set
    dataroot_ = args.dataroot
    version_ = args.version
    config_path = args.config_path
    plot_examples_ = args.plot_examples
    render_curves_ = bool(args.render_curves)
    verbose_ = bool(args.verbose)

    if config_path == '':
        cfg_ = config_factory('detection_cvpr_2019')
    else:
        with open(config_path, 'r') as _f:
            cfg_ = DetectionConfig.deserialize(json.load(_f))

    nusc_ = NuScenes(version=version_, verbose=verbose_, dataroot=dataroot_)
    nusc_eval = NuScenesEval_custom(nusc_, config=cfg_, result_path=result_path_, eval_set=eval_set_,
                                    output_dir=output_dir_, verbose=verbose_)
    for vis in ['1', '2', '3', '4']:
        nusc_eval.update_gt(type_='vis', visibility=vis)
        print(f'================ {vis} ===============')
        nusc_eval.main(plot_examples=plot_examples_, render_curves=render_curves_)
