from .track_instance import Instances
from mmdet3d.core.bbox.iou_calculators.iou3d_calculator import (
    bbox_overlaps_nearest_3d as iou_3d, )
from projects.mmdet3d_plugin.core.bbox.util import denormalize_bbox

class RuntimeTrackerBase(object):
    def __init__(self, score_thresh=0.5, filter_score_thresh=0.4,  miss_tolerance=5):
        self.score_thresh = score_thresh
        self.filter_score_thresh = filter_score_thresh
        self.miss_tolerance = miss_tolerance
        self.max_obj_id = 0

    def clear(self):
        self.max_obj_id = 0

    def update(self, track_instances: Instances, iou_thre=None):
        track_instances.disappear_time[track_instances.scores >= self.score_thresh] = 0
        for i in range(len(track_instances)):
            if (
                track_instances.obj_idxes[i] == -1
                and track_instances.scores[i] >= self.score_thresh
            ):  
                if iou_thre is not None and track_instances.pred_boxes[track_instances.obj_idxes>=0].shape[0]!=0:
                    iou3ds = iou_3d(denormalize_bbox(track_instances.pred_boxes[i].unsqueeze(0), None)[...,:7], denormalize_bbox(track_instances.pred_boxes[track_instances.obj_idxes>=0], None)[...,:7])
                    if iou3ds.max()>iou_thre:
                        continue
                # new track
                # print("track {} has score {}, assign obj_id {}".format(i, track_instances.scores[i], self.max_obj_id))
                track_instances.obj_idxes[i] = self.max_obj_id
                self.max_obj_id += 1
            elif (
                track_instances.obj_idxes[i] >= 0
                and track_instances.scores[i] < self.filter_score_thresh
            ):
                # sleep time ++
                track_instances.disappear_time[i] += 1
                if track_instances.disappear_time[i] >= self.miss_tolerance:
                    # mark deaded tracklets: Set the obj_id to -1.
                    # TODO: remove it by following functions
                    # Then this track will be removed by TrackEmbeddingLayer.
                    track_instances.obj_idxes[i] = -1
                    