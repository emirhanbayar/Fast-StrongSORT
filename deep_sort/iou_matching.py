# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
# from opts import opt


def iou(bbox, candidates):
    """Computer intersection over union.

    Parameters
    ----------
    bbox : ndarray
        A bounding box in format `(top left x, top left y, width, height)`.
    candidates : ndarray
        A matrix of candidate bounding boxes (one per row) in the same format
        as `bbox`.

    Returns
    -------
    ndarray
        The intersection over union in [0, 1] between the `bbox` and each
        candidate. A higher score means a larger fraction of the `bbox` is
        occluded by the candidate.

    """
    
    candidates = np.array(candidates)
    bbox_tl, bbox_br = bbox[:2], bbox[:2] + bbox[2:]
    candidates_tl = candidates[:, :2]
    candidates_br = candidates[:, :2] + candidates[:, 2:]

    tl = np.c_[np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
               np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis]]
    br = np.c_[np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
               np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis]]
    wh = np.maximum(0., br - tl)

    area_intersection = wh.prod(axis=1)
    area_bbox = bbox[2:].prod()
    area_candidates = candidates[:, 2:].prod(axis=1)
    return area_intersection / (area_bbox + area_candidates - area_intersection)

def aiou(bbox, candidates):
    """
    IoU - Aspect Ratio

    """
    candidates = np.array(candidates)
    bbox_tl, bbox_br = bbox[:2], bbox[:2] + bbox[2:]
    candidates_tl = candidates[:, :2]
    candidates_br = candidates[:, :2] + candidates[:, 2:]

    tl = np.c_[np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
               np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis]]
    br = np.c_[np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
               np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis]]
    wh = np.maximum(0., br - tl)

    area_intersection = wh.prod(axis=1)
    area_bbox = bbox[2:].prod()
    area_candidates = candidates[:, 2:].prod(axis=1)

    iou = area_intersection / (area_bbox + area_candidates - area_intersection)

    aspect_ratio = bbox[2] / bbox[3]
    candidates_aspect_ratio = candidates[:, 2] / candidates[:, 3]
    arctan = np.arctan(aspect_ratio) - np.arctan(candidates_aspect_ratio)
    v = 1 - ((4 / np.pi ** 2) * arctan ** 2)
    alpha = v / (1 - iou + v)

    # iou[alpha < 0.8] = 0

    return iou, alpha
    

def batch_iou(boxes1, boxes2):
    """
    Compute the Intersection-Over-Union of a batch of boxes with another
    batch of boxes.
    """

    bbox1_tl, bbox1_br = boxes1[:, :2], boxes1[:, :2] + boxes1[:, 2:]
    bbox2_tl, bbox2_br = boxes2[:, :2], boxes2[:, :2] + boxes2[:, 2:]

    tl = np.maximum(bbox1_tl[:, None], bbox2_tl)
    br = np.minimum(bbox1_br[:, None], bbox2_br)

    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area1 = np.prod(bbox1_br - bbox1_tl, axis=1)
    area2 = np.prod(bbox2_br - bbox2_tl, axis=1)

    return area_i / (area1[:, None] + area2 - area_i)


def iou_cost(tracks, detections, track_indices=None,
             detection_indices=None):
    """An intersection over union distance metric.

    Parameters
    ----------
    tracks : List[deep_sort.track.Track]
        A list of tracks.
    detections : List[deep_sort.detection.Detection]
        A list of detections.
    track_indices : Optional[List[int]]
        A list of indices to tracks that should be matched. Defaults to
        all `tracks`.
    detection_indices : Optional[List[int]]
        A list of indices to detections that should be matched. Defaults
        to all `detections`.

    Returns
    -------
    ndarray
        Returns a cost matrix of shape
        len(track_indices), len(detection_indices) where entry (i, j) is
        `1 - iou(tracks[track_indices[i]], detections[detection_indices[j]])`.

    """
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
    for row, track_idx in enumerate(track_indices):
        if tracks[track_idx].time_since_update > 1:
            cost_matrix[row, :] = 1e+5
            continue

        bbox = tracks[track_idx].to_tlwh()
        candidates = np.asarray([detections[i].tlwh for i in detection_indices])
        cost_matrix[row, :] = 1. - iou(bbox, candidates)
    return cost_matrix