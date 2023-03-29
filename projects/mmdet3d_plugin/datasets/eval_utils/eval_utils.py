import json
import torch
import tqdm
from typing import List, Dict, Tuple, Callable, Union
from nuscenes import NuScenes
from pyquaternion import Quaternion
import numpy as np
from .metric_utils import min_ade, min_fde, miss_rate

from nuscenes.utils.splits import create_splits_scenes
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.prediction import PredictHelper, convert_local_coords_to_global
from nuscenes.eval.common.data_classes import EvalBox, EvalBoxes
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.detection.data_classes import DetectionMetricData, DetectionMetricDataList, DetectionMetrics
from nuscenes.eval.common.utils import center_distance, scale_iou, yaw_diff, velocity_l2, attr_acc, cummean

def category_to_motion_name(category_name: str):
    """
    Default label mapping from nuScenes to nuScenes detection classes.
    Note that pedestrian does not include personal_mobility, stroller and wheelchair.
    :param category_name: Generic nuScenes class.
    :return: nuScenes detection class.
    """
    detection_mapping = {
        'movable_object.barrier': 'barrier',
        'vehicle.bicycle': 'car',
        'vehicle.bus.bendy': 'car',
        'vehicle.bus.rigid': 'car',
        'vehicle.car': 'car',
        'vehicle.construction': 'car',
        'vehicle.motorcycle': 'car',
        'human.pedestrian.adult': 'pedestrian',
        'human.pedestrian.child': 'pedestrian',
        'human.pedestrian.construction_worker': 'pedestrian',
        'human.pedestrian.police_officer': 'pedestrian',
        'movable_object.trafficcone': 'barrier',
        'vehicle.trailer': 'car',
        'vehicle.truck': 'car'
    }

    if category_name in detection_mapping:
        return detection_mapping[category_name]
    else:
        return None

def detection_prediction_category_to_motion_name(category_name: str):
    """
    Default label mapping from nuScenes to nuScenes detection classes.
    Note that pedestrian does not include personal_mobility, stroller and wheelchair.
    :param category_name: Generic nuScenes class.
    :return: nuScenes detection class.
    """
    detection_mapping = {
        'car': 'car',
        'truck': 'car',
        'construction_vehicle': 'car',
        'bus': 'car',
        'trailer': 'car',
        'motorcycle': 'car',
        'bicycle': 'car',
        'pedestrian': 'pedestrian',
        'traffic_cone': 'barrier',
        'barrier': 'barrier',
    }

    if category_name in detection_mapping:
        return detection_mapping[category_name]
    else:
        return None

class DetectionMotionMetrics(DetectionMetrics):
    """ Stores average precision and true positive metric results. Provides properties to summarize. """

    @classmethod
    def deserialize(cls, content: dict):
        """ Initialize from serialized dictionary. """

        cfg = DetectionConfig.deserialize(content['cfg'])
        metrics = cls(cfg=cfg)
        metrics.add_runtime(content['eval_time'])

        for detection_name, label_aps in content['label_aps'].items():
            for dist_th, ap in label_aps.items():
                metrics.add_label_ap(detection_name=detection_name, dist_th=float(dist_th), ap=float(ap))

        for detection_name, label_tps in content['label_tp_errors'].items():
            for metric_name, tp in label_tps.items():
                metrics.add_label_tp(detection_name=detection_name, metric_name=metric_name, tp=float(tp))

        return metrics

class DetectionMotionMetricDataList(DetectionMetricDataList):
    """ This stores a set of MetricData in a dict indexed by (name, match-distance). """
    @classmethod
    def deserialize(cls, content: dict):
        mdl = cls()
        for key, md in content.items():
            name, distance = key.split(':')
            mdl.set(name, float(distance), DetectionMotionMetricData.deserialize(md))
        return mdl

class DetectionMotionMetricData(DetectionMetricData):
    """ This class holds accumulated and interpolated data required to calculate the detection metrics. """

    nelem = 101

    def __init__(self,
                 recall: np.array,
                 precision: np.array,
                 confidence: np.array,
                 trans_err: np.array,
                 vel_err: np.array,
                 scale_err: np.array,
                 orient_err: np.array,
                 attr_err: np.array,
                 min_ade_err: np.array,
                 min_fde_err: np.array,
                 miss_rate_err: np.array):

        # Assert lengths.
        assert len(recall) == self.nelem
        assert len(precision) == self.nelem
        assert len(confidence) == self.nelem
        assert len(trans_err) == self.nelem
        assert len(vel_err) == self.nelem
        assert len(scale_err) == self.nelem
        assert len(orient_err) == self.nelem
        assert len(attr_err) == self.nelem
        assert len(min_ade_err) == self.nelem
        assert len(min_fde_err) == self.nelem
        assert len(miss_rate_err) == self.nelem

        # Assert ordering.
        assert all(confidence == sorted(confidence, reverse=True))  # Confidences should be descending.
        assert all(recall == sorted(recall))  # Recalls should be ascending.

        # Set attributes explicitly to help IDEs figure out what is going on.
        self.recall = recall
        self.precision = precision
        self.confidence = confidence
        self.trans_err = trans_err
        self.vel_err = vel_err
        self.scale_err = scale_err
        self.orient_err = orient_err
        self.attr_err = attr_err
        self.min_ade_err = min_ade_err
        self.min_fde_err = min_fde_err
        self.miss_rate_err = miss_rate_err

    def __eq__(self, other):
        eq = True
        for key in self.serialize().keys():
            eq = eq and np.array_equal(getattr(self, key), getattr(other, key))
        return eq

    @property
    def max_recall_ind(self):
        """ Returns index of max recall achieved. """

        # Last instance of confidence > 0 is index of max achieved recall.
        non_zero = np.nonzero(self.confidence)[0]
        if len(non_zero) == 0:  # If there are no matches, all the confidence values will be zero.
            max_recall_ind = 0
        else:
            max_recall_ind = non_zero[-1]

        return max_recall_ind

    @property
    def max_recall(self):
        """ Returns max recall achieved. """

        return self.recall[self.max_recall_ind]

    def serialize(self):
        """ Serialize instance into json-friendly format. """
        return {
            'recall': self.recall.tolist(),
            'precision': self.precision.tolist(),
            'confidence': self.confidence.tolist(),
            'trans_err': self.trans_err.tolist(),
            'vel_err': self.vel_err.tolist(),
            'scale_err': self.scale_err.tolist(),
            'orient_err': self.orient_err.tolist(),
            'attr_err': self.attr_err.tolist(),
            'min_ade_err': self.min_ade_err.tolist(),
            'min_fde_err': self.min_fde_err.tolist(),
            'miss_rate_err': self.miss_rate_err.tolist(),
        }

    @classmethod
    def deserialize(cls, content: dict):
        """ Initialize from serialized content. """
        return cls(recall=np.array(content['recall']),
                   precision=np.array(content['precision']),
                   confidence=np.array(content['confidence']),
                   trans_err=np.array(content['trans_err']),
                   vel_err=np.array(content['vel_err']),
                   scale_err=np.array(content['scale_err']),
                   orient_err=np.array(content['orient_err']),
                   attr_err=np.array(content['attr_err']),
                   min_ade_err=np.array(content['min_ade_err']),
                   min_fde_err=np.array(content['min_fde_err']),
                   miss_rate_err=np.array(content['miss_rate_err']))

    @classmethod
    def no_predictions(cls):
        """ Returns a md instance corresponding to having no predictions. """
        return cls(recall=np.linspace(0, 1, cls.nelem),
                   precision=np.zeros(cls.nelem),
                   confidence=np.zeros(cls.nelem),
                   trans_err=np.ones(cls.nelem),
                   vel_err=np.ones(cls.nelem),
                   scale_err=np.ones(cls.nelem),
                   orient_err=np.ones(cls.nelem),
                   attr_err=np.ones(cls.nelem),
                   min_ade_err=np.ones(cls.nelem),
                   min_fde_err=np.ones(cls.nelem),
                   miss_rate_err=np.ones(cls.nelem))

    @classmethod
    def random_md(cls):
        """ Returns an md instance corresponding to a random results. """
        return cls(recall=np.linspace(0, 1, cls.nelem),
                   precision=np.random.random(cls.nelem),
                   confidence=np.linspace(0, 1, cls.nelem)[::-1],
                   trans_err=np.random.random(cls.nelem),
                   vel_err=np.random.random(cls.nelem),
                   scale_err=np.random.random(cls.nelem),
                   orient_err=np.random.random(cls.nelem),
                   attr_err=np.random.random(cls.nelem),
                   min_ade_err=np.random.random(cls.nelem),
                   min_fde_err=np.random.random(cls.nelem),
                   miss_rate_err=np.random.random(cls.nelem))


class DetectionMotionBox(DetectionBox):
    def __init__(self,
                 sample_token: str = "",
                 translation: Tuple[float, float, float] = (0, 0, 0),
                 size: Tuple[float, float, float] = (0, 0, 0),
                 rotation: Tuple[float, float, float, float] = (0, 0, 0, 0),
                 velocity: Tuple[float, float] = (0, 0),
                 ego_translation: [float, float, float] = (0, 0, 0),  # Translation to ego vehicle in meters.
                 num_pts: int = -1,  # Nbr. LIDAR or RADAR inside the box. Only for gt boxes.
                 detection_name: str = 'car',  # The class name used in the detection challenge.
                 detection_score: float = -1.0,  # GT samples do not have a score.
                 attribute_name: str = '',
                 traj=None,
                 traj_scores=None):  # Box attribute. Each box can have at most 1 attribute.
        super(DetectionBox, self).__init__(sample_token, translation, size, rotation, velocity, ego_translation, num_pts)
        assert detection_name is not None, 'Error: detection_name cannot be empty!'
        # assert detection_name in DETECTION_NAMES, 'Error: Unknown detection_name %s' % detection_name

        # assert attribute_name in ATTRIBUTE_NAMES or attribute_name == '', \
        #     'Error: Unknown attribute_name %s' % attribute_name

        assert type(detection_score) == float, 'Error: detection_score must be a float!'
        assert not np.any(np.isnan(detection_score)), 'Error: detection_score may not be NaN!'

        # Assign.
        self.detection_name = detection_name
        self.attribute_name = attribute_name
        self.detection_score = detection_score
        self.traj = traj
        self.traj_scores = traj_scores
        self.traj_index = None

    def __eq__(self, other):
        return (self.sample_token == other.sample_token and
                self.translation == other.translation and
                self.size == other.size and
                self.rotation == other.rotation and
                self.velocity == other.velocity and
                self.ego_translation == other.ego_translation and
                self.num_pts == other.num_pts and
                self.detection_name == other.detection_name and
                self.detection_score == other.detection_score and
                self.attribute_name == other.attribute_name and 
                np.all(self.traj == other.traj) and
                np.all(self.traj_scores == other.traj_scores))

    def serialize(self) -> dict:
        """ Serialize instance into json-friendly format. """
        return {
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
            'traj': self.traj,
            'traj_scores': self.traj_scores
        }

    @classmethod
    def deserialize(cls, content: dict):
        """ Initialize from serialized content. """
        return cls(sample_token=content['sample_token'],
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
                   traj=content['predict_traj'],
                   traj_scores=content['predict_traj_score'])

class DetectionMotionBox_modified(DetectionMotionBox):
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
            'index': self.index,
            'traj': self.traj,
            'traj_scores': self.traj_scores
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
            traj=content['traj'],
        )


def load_prediction(result_path: str, max_boxes_per_sample: int, box_cls, verbose: bool = False, category_convert_type='detection_category') \
        -> Tuple[EvalBoxes, Dict]:
    """
    Loads object predictions from file.
    :param result_path: Path to the .json result file provided by the user.
    :param max_boxes_per_sample: Maximim number of boxes allowed per sample.
    :param box_cls: Type of box to load, e.g. DetectionBox, DetectionMotionBox or TrackingBox.
    :param verbose: Whether to print messages to stdout.
    :return: The deserialized results and meta data.
    """

    # Load from file and check that the format is correct.
    with open(result_path) as f:
        data = json.load(f)
    assert 'results' in data, 'Error: No field `results` in result file. Please note that the result format changed.' \
                              'See https://www.nuscenes.org/object-detection for more information.'

    if category_convert_type == 'motion_category':
        for key in data['results'].keys():
            for i in range(len(data['results'][key])):
                data['results'][key][i]['detection_name'] = detection_prediction_category_to_motion_name(data['results'][key][i]['detection_name']) 
    # Deserialize results and get meta data.
    all_results = EvalBoxes.deserialize(data['results'], box_cls)
    meta = data['meta']
    if verbose:
        print("Loaded results from {}. Found detections for {} samples."
              .format(result_path, len(all_results.sample_tokens)))

    # Check that each sample has no more than x predicted boxes.
    for sample_token in all_results.sample_tokens:
        assert len(all_results.boxes[sample_token]) <= max_boxes_per_sample, \
            "Error: Only <= %d boxes per sample allowed!" % max_boxes_per_sample

    return all_results, meta

def load_gt(nusc: NuScenes, eval_split: str, box_cls, verbose: bool = False, category_convert_type='detection_category'):
    """
    Loads ground truth boxes from DB.
    :param nusc: A NuScenes instance.
    :param eval_split: The evaluation split for which we load GT boxes.
    :param box_cls: Type of box to load, e.g. DetectionBox or TrackingBox.
    :param verbose: Whether to print messages to stdout.
    :return: The GT boxes.
    """
    predict_helper = PredictHelper(nusc)
    # Init.
    if box_cls == DetectionMotionBox_modified:
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
            if box_cls == DetectionMotionBox_modified:
                # Get label name in detection task and filter unused labels.
                if category_convert_type == 'detection_category':
                    detection_name = category_to_detection_name(sample_annotation['category_name'])
                elif category_convert_type == 'motion_category':
                    detection_name = category_to_motion_name(sample_annotation['category_name'])
                else:
                    raise NotImplementedError
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
                instance_token = nusc.get('sample_annotation', sample_annotation['token'])['instance_token']
                fut_traj_local = predict_helper.get_future_for_agent(instance_token, sample_token, seconds=6, in_agent_frame=True)
                fut_traj_scence_centric = np.zeros((0,))
                if fut_traj_local.shape[0] > 0:
                    _, boxes, _ = nusc.get_sample_data(sample['data']['LIDAR_TOP'], selected_anntokens=[sample_annotation['token']])
                    box = boxes[0]
                    trans = box.center
                    rot = Quaternion(matrix=box.rotation_matrix)
                    fut_traj_scence_centric = convert_local_coords_to_global(fut_traj_local, trans, rot) 
                    
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
                        index=index_map[sample_token],
                        traj=fut_traj_scence_centric,
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

def prediction_metrics(gt_box_match, pred_box):
    pred_traj = np.array(pred_box.traj)
    gt_traj_steps = gt_box_match.traj.reshape((-1, 2))
    valid_steps = gt_traj_steps.shape[0]
    if valid_steps <= 0:
        return np.array([0]), np.array([0]), 0
    nmodes = pred_traj.shape[0]
    pred_steps = pred_traj.shape[1]
    valid_mask = np.zeros((pred_steps, ))
    gt_traj = np.zeros((pred_steps, 2))
    gt_traj[:valid_steps, :] = gt_traj_steps
    valid_mask[: valid_steps] = 1
    pred_traj = torch.tensor(pred_traj[None])
    gt_traj = torch.tensor(gt_traj[None])
    valid_mask = torch.tensor(valid_mask[None])
    ade_err, inds = min_ade(pred_traj, gt_traj, 1 - valid_mask)
    fde_err, inds = min_fde(pred_traj, gt_traj, 1 - valid_mask)
    mr_err = miss_rate(pred_traj, gt_traj, 1 - valid_mask, dist_thresh=2)
    return ade_err.numpy(), fde_err.numpy(), mr_err.numpy()


def accumulate(gt_boxes: EvalBoxes,
               pred_boxes: EvalBoxes,
               class_name: str,
               dist_fcn: Callable,
               dist_th: float,
               verbose: bool = False) -> DetectionMotionMetricData:
    """
    Average Precision over predefined different recall thresholds for a single distance threshold.
    The recall/conf thresholds and other raw metrics will be used in secondary metrics.
    :param gt_boxes: Maps every sample_token to a list of its sample_annotations.
    :param pred_boxes: Maps every sample_token to a list of its sample_results.
    :param class_name: Class to compute AP on.
    :param dist_fcn: Distance function used to match detections and ground truths.
    :param dist_th: Distance threshold for a match.
    :param verbose: If true, print debug messages.
    :return: (average_prec, metrics). The average precision value and raw data for a number of metrics.
    """
    # ---------------------------------------------
    # Organize input and initialize accumulators.
    # ---------------------------------------------

    # Count the positives.
    npos = len([1 for gt_box in gt_boxes.all if gt_box.detection_name == class_name])
    if verbose:
        print("Found {} GT of class {} out of {} total across {} samples.".
              format(npos, class_name, len(gt_boxes.all), len(gt_boxes.sample_tokens)))

    # For missing classes in the GT, return a data structure corresponding to no predictions.
    if npos == 0:
        return DetectionMotionMetricData.no_predictions(), 0, 0, 0

    # Organize the predictions in a single list.
    pred_boxes_list = [box for box in pred_boxes.all if box.detection_name == class_name]
    pred_confs = [box.detection_score for box in pred_boxes_list]

    if verbose:
        print("Found {} PRED of class {} out of {} total across {} samples.".
              format(len(pred_confs), class_name, len(pred_boxes.all), len(pred_boxes.sample_tokens)))

    # Sort by confidence.
    sortind = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(pred_confs))][::-1]

    # Do the actual matching.
    tp = []  # Accumulator of true positives.
    fp = []  # Accumulator of false positives.
    conf = []  # Accumulator of confidences.

    # match_data holds the extra metrics we calculate for each match.
    match_data = {'trans_err': [],
                  'vel_err': [],
                  'scale_err': [],
                  'orient_err': [],
                  'attr_err': [],
                  'conf': [],
                  'min_ade_err': [],
                  'min_fde_err': [],
                  'miss_rate_err': []}

    # ---------------------------------------------
    # Match and accumulate match data.
    # ---------------------------------------------

    taken = set()  # Initially no gt bounding box is matched.
    for ind in sortind:
        pred_box = pred_boxes_list[ind]
        min_dist = np.inf
        match_gt_idx = None

        for gt_idx, gt_box in enumerate(gt_boxes[pred_box.sample_token]):

            # Find closest match among ground truth boxes
            if gt_box.detection_name == class_name and not (pred_box.sample_token, gt_idx) in taken:
                this_distance = dist_fcn(gt_box, pred_box)
                if this_distance < min_dist:
                    min_dist = this_distance
                    match_gt_idx = gt_idx

        # If the closest match is close enough according to threshold we have a match!
        is_match = min_dist < dist_th

        if is_match:
            taken.add((pred_box.sample_token, match_gt_idx))

            #  Update tp, fp and confs.
            tp.append(1)
            fp.append(0)
            conf.append(pred_box.detection_score)

            # Since it is a match, update match data also.
            gt_box_match = gt_boxes[pred_box.sample_token][match_gt_idx]
            
            match_data['trans_err'].append(center_distance(gt_box_match, pred_box))
            match_data['vel_err'].append(velocity_l2(gt_box_match, pred_box))
            match_data['scale_err'].append(1 - scale_iou(gt_box_match, pred_box))

            # Barrier orientation is only determined up to 180 degree. (For cones orientation is discarded later)
            period = np.pi if class_name == 'barrier' else 2 * np.pi
            match_data['orient_err'].append(yaw_diff(gt_box_match, pred_box, period=period))

            match_data['attr_err'].append(1 - attr_acc(gt_box_match, pred_box))
            minade, minfde, m_r = prediction_metrics(gt_box_match, pred_box)
            
            match_data['min_ade_err'].append(minade)
            match_data['min_fde_err'].append(minfde)
            match_data['miss_rate_err'].append(m_r)
            match_data['conf'].append(pred_box.detection_score)

        else:
            # No match. Mark this as a false positive.
            tp.append(0)
            fp.append(1)
            conf.append(pred_box.detection_score)

    # Check if we have any matches. If not, just return a "no predictions" array.
    if len(match_data['trans_err']) == 0:
        return DetectionMotionMetricData.no_predictions(), 0, 0, 0

    # ---------------------------------------------
    # Calculate and interpolate precision and recall
    # ---------------------------------------------

    # Accumulate.
    N_tp = np.sum(tp)
    N_fp = np.sum(fp)
    tp = np.cumsum(tp).astype(float)
    fp = np.cumsum(fp).astype(float)
    conf = np.array(conf)


    # Calculate precision and recall.
    prec = tp / (fp + tp)
    rec = tp / float(npos)

    rec_interp = np.linspace(0, 1, DetectionMotionMetricData.nelem)  # 101 steps, from 0% to 100% recall.
    prec = np.interp(rec_interp, rec, prec, right=0)
    conf = np.interp(rec_interp, rec, conf, right=0)
    rec = rec_interp

    # ---------------------------------------------
    # Re-sample the match-data to match, prec, recall and conf.
    # ---------------------------------------------

    for key in match_data.keys():
        if key == "conf":
            continue  # Confidence is used as reference to align with fp and tp. So skip in this step.

        else:
            # For each match_data, we first calculate the accumulated mean.
            tmp = cummean(np.array(match_data[key]))

            # Then interpolate based on the confidences. (Note reversing since np.interp needs increasing arrays)
            match_data[key] = np.interp(conf[::-1], match_data['conf'][::-1], tmp[::-1])[::-1]

    # ---------------------------------------------
    # Done. Instantiate MetricData and return
    # ---------------------------------------------
    return DetectionMotionMetricData(recall=rec,
                               precision=prec,
                               confidence=conf,
                               trans_err=match_data['trans_err'],
                               vel_err=match_data['vel_err'],
                               scale_err=match_data['scale_err'],
                               orient_err=match_data['orient_err'],
                               attr_err=match_data['attr_err'],
                               min_ade_err=match_data['min_ade_err'],
                               min_fde_err=match_data['min_fde_err'],
                               miss_rate_err=match_data['miss_rate_err']
                               ), N_tp, N_fp, npos



def accumulate_motion(gt_boxes: EvalBoxes,
               pred_boxes: EvalBoxes,
               class_name: str,
               dist_fcn: Callable,
               traj_fcn: Callable,
               dist_th: float,
               traj_dist_th: float,
               verbose: bool = False,
               final_step: float = 12) -> DetectionMotionMetricData:
    """
    Average Precision over predefined different recall thresholds for a single distance threshold.
    The recall/conf thresholds and other raw metrics will be used in secondary metrics.
    :param gt_boxes: Maps every sample_token to a list of its sample_annotations.
    :param pred_boxes: Maps every sample_token to a list of its sample_results.
    :param class_name: Class to compute AP on.
    :param dist_fcn: Distance function used to match detections and ground truths.
    :param dist_th: Distance threshold for a match.
    :param verbose: If true, print debug messages.
    :return: (average_prec, metrics). The average precision value and raw data for a number of metrics.
    """
    # ---------------------------------------------
    # Organize input and initialize accumulators.
    # ---------------------------------------------

    # Count the positives.
    npos = len([1 for gt_box in gt_boxes.all if gt_box.detection_name == class_name])
    if verbose:
        print("Found {} GT of class {} out of {} total across {} samples.".
              format(npos, class_name, len(gt_boxes.all), len(gt_boxes.sample_tokens)))

    # For missing classes in the GT, return a data structure corresponding to no predictions.
    if npos == 0:
        return DetectionMotionMetricData.no_predictions(), 0, 0, 0

    # 
    # Organize the predictions in a single list.
    pred_boxes_list = []
    pred_confs = []

    pred_boxes_list = [box for box in pred_boxes.all if box.detection_name == class_name]
    pred_confs = [box.detection_score for box in pred_boxes_list]
    # for box in pred_boxes.all:
    #     if box.detection_name == class_name:
    #         box.traj_scores = np.exp(box.traj_scores)
    #         for i in range(len(box.traj_scores)):
    #             box.traj_index = i
    #             pred_boxes_list.append(box)
    # pred_confs = [box.detection_score * box.traj_scores[box.traj_index]  for box in pred_boxes_list]

    if verbose:
        print("Found {} PRED of class {} out of {} total across {} samples.".
              format(len(pred_confs), class_name, len(pred_boxes.all), len(pred_boxes.sample_tokens)))

    # Sort by confidence.
    sortind = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(pred_confs))][::-1]

    # Do the actual matching.
    tp = []  # Accumulator of true positives.
    fp = []  # Accumulator of false positives.
    conf = []  # Accumulator of confidences.

    # match_data holds the extra metrics we calculate for each match.
    match_data = {'trans_err': [],
                  'vel_err': [],
                  'scale_err': [],
                  'orient_err': [],
                  'attr_err': [],
                  'conf': [],
                  'min_ade_err': [],
                  'min_fde_err': [],
                  'miss_rate_err': []}

    # ---------------------------------------------
    # Match and accumulate match data.
    # ---------------------------------------------

    taken = set()  # Initially no gt bounding box is matched.
    for ind in sortind:
        pred_box = pred_boxes_list[ind]
        min_dist = np.inf
        match_gt_idx = None

        for gt_idx, gt_box in enumerate(gt_boxes[pred_box.sample_token]):

            # Find closest match among ground truth boxes
            if gt_box.detection_name == class_name and not (pred_box.sample_token, gt_idx) in taken:
                this_distance = dist_fcn(gt_box, pred_box)
                if this_distance < min_dist:
                    min_dist = this_distance
                    match_gt_idx = gt_idx
                    fde_distance = traj_fcn(gt_box, pred_box, final_step)
        # If the closest match is close enough according to threshold we have a match!
        is_match = min_dist < dist_th and fde_distance < traj_dist_th

        if is_match:
            taken.add((pred_box.sample_token, match_gt_idx))

            #  Update tp, fp and confs.
            tp.append(1)
            fp.append(0)
            conf.append(pred_box.detection_score)

            # Since it is a match, update match data also.
            gt_box_match = gt_boxes[pred_box.sample_token][match_gt_idx]
            
            match_data['trans_err'].append(center_distance(gt_box_match, pred_box))
            match_data['vel_err'].append(velocity_l2(gt_box_match, pred_box))
            match_data['scale_err'].append(1 - scale_iou(gt_box_match, pred_box))

            # Barrier orientation is only determined up to 180 degree. (For cones orientation is discarded later)
            period = np.pi if class_name == 'barrier' else 2 * np.pi
            match_data['orient_err'].append(yaw_diff(gt_box_match, pred_box, period=period))

            match_data['attr_err'].append(1 - attr_acc(gt_box_match, pred_box))
            minade, minfde, m_r = prediction_metrics(gt_box_match, pred_box)
            
            match_data['min_ade_err'].append(minade)
            match_data['min_fde_err'].append(minfde)
            match_data['miss_rate_err'].append(m_r)
            match_data['conf'].append(pred_box.detection_score)

        else:
            # No match. Mark this as a false positive.
            tp.append(0)
            fp.append(1)
            conf.append(pred_box.detection_score)
            # conf.append(pred_box.detection_score * pred_box.traj_scores[pred_box.traj_index])
    # 
    # Check if we have any matches. If not, just return a "no predictions" array.
    if len(match_data['trans_err']) == 0:
        return DetectionMotionMetricData.no_predictions(), 0, 0, 0

    # ---------------------------------------------
    # Calculate and interpolate precision and recall
    # ---------------------------------------------

    # Accumulate.
    N_tp = np.sum(tp)
    N_fp = np.sum(fp)
    tp = np.cumsum(tp).astype(float)
    fp = np.cumsum(fp).astype(float)
    conf = np.array(conf)

    # Calculate precision and recall.
    prec = tp / (fp + tp)
    rec = tp / float(npos)



    rec_interp = np.linspace(0, 1, DetectionMotionMetricData.nelem)  # 101 steps, from 0% to 100% recall.
    prec = np.interp(rec_interp, rec, prec, right=0)
    conf = np.interp(rec_interp, rec, conf, right=0)
    rec = rec_interp

    # ---------------------------------------------
    # Re-sample the match-data to match, prec, recall and conf.
    # ---------------------------------------------

    for key in match_data.keys():
        if key == "conf":
            continue  # Confidence is used as reference to align with fp and tp. So skip in this step.

        else:
            # For each match_data, we first calculate the accumulated mean.
            tmp = cummean(np.array(match_data[key]))

            # Then interpolate based on the confidences. (Note reversing since np.interp needs increasing arrays)
            match_data[key] = np.interp(conf[::-1], match_data['conf'][::-1], tmp[::-1])[::-1]

    # ---------------------------------------------
    # Done. Instantiate MetricData and return
    # ---------------------------------------------
    return DetectionMotionMetricData(recall=rec,
                               precision=prec,
                               confidence=conf,
                               trans_err=match_data['trans_err'],
                               vel_err=match_data['vel_err'],
                               scale_err=match_data['scale_err'],
                               orient_err=match_data['orient_err'],
                               attr_err=match_data['attr_err'],
                               min_ade_err=match_data['min_ade_err'],
                               min_fde_err=match_data['min_fde_err'],
                               miss_rate_err=match_data['miss_rate_err']
                               ), N_tp, N_fp, npos