# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track
from opts import opt
from .embedding import Embedder
import time

class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, metric, max_iou_distance=0.7, max_age=30, n_init=3):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.tracks = []
        self._next_id = 1

        self.ecc_time = 0
        self.prediction_time = 0
        self.gate_time = 0
        self.match_time = 0
        self.track_management_time = 0
        self.embedding_time = 0
        self.read_time = 0
        self.total_frames = 0
        self.total_detections = 0
        self.total_extracted_features = 0

        self.embedding = Embedder()

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict()

    def camera_update(self, video, frame):
        for track in self.tracks:
            track.camera_update(video, frame)

    def update(self, detections, frame):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        self.total_frames += 1
        self.total_detections += len(detections)

        # Run matching cascade.
        if opt.prematch:
            matches, unmatched_tracks, unmatched_detections = \
                self._match_prematch(detections, frame)
        else:
            matches, unmatched_tracks, unmatched_detections = \
                self._match(detections, frame)

        track_management_start = time.time()
        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            if not opt.EMA:
                track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)
        
        self.track_management_time += time.time() - track_management_start

    def _match_prematch(self, detections, frame):
        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = linear_assignment.gate_cost_matrix(
                cost_matrix, tracks, dets, track_indices,
                detection_indices)

            return cost_matrix
        

        match_start = time.time()
        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]
        self.match_time += time.time() - match_start
        
        prematches = []
        unprematched_detections = []
        unprematched_tracks = []
        for i, det in enumerate(detections):
            if len(confirmed_tracks) == 0:
                unprematched_detections.append(i)
                detections[i].gated = False
                continue
            ious, alpha = iou_matching.aiou(det.tlwh, [self.tracks[i].to_tlwh() for i in confirmed_tracks])
            if np.sum(ious > opt.occlusion_threshold) != 1:
                unprematched_detections.append(i)
                detections[i].gated = False
            else:
                candidate = np.argmax(ious)
                if alpha is not None and alpha[candidate] > opt.aiou_threshold:
                    prematches.append((candidate, i))
                    detections[i].feature = self.tracks[candidate].features[-1]
                    detections[i].gated = True
                else:
                    unprematched_detections.append(i)
                    detections[i].gated = False

        for i in range(len(self.tracks)):
            if i not in [p[0] for p in prematches] and i in confirmed_tracks:
                unprematched_tracks.append(i)

        # Associate confirmed tracks using appearance features.
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, track_indices=unprematched_tracks, detection_indices=unprematched_detections)
        
        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b + prematches
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b) - set([m[0] for m in matches]))
        return matches, unmatched_tracks, unmatched_detections

    def _match(self, detections, frame):

        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = linear_assignment.gate_cost_matrix(
                cost_matrix, tracks, dets, track_indices,
                detection_indices)

            return cost_matrix

        match_start = time.time()
        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]
        self.match_time += time.time() - match_start

        gate_start = time.time()
        non_gated_detecions = []
        tracklet_bboxes = [self.tracks[i].to_tlwh() for i in confirmed_tracks]
        det_tlwhs = [det.tlwh for det in detections]
        for i, det in enumerate(detections):
            if len(tracklet_bboxes) == 0:
                non_gated_detecions.append(detections[i].tlwh)
                detections[i].gated = False
                continue
            

            ious, alpha = iou_matching.aiou(det_tlwhs[i], tracklet_bboxes)
            if np.sum(ious > opt.occlusion_threshold) != 1:
                non_gated_detecions.append(detections[i].tlwh)
                detections[i].gated = False
            else:
                candidate = np.argmax(ious)
                if opt.EMA and alpha is not None and alpha[candidate] > opt.aiou_threshold:
                    detections[i].feature = self.tracks[candidate].features[-1]
                    detections[i].gated = True
                else:
                    non_gated_detecions.append(detections[i].tlwh)
                    detections[i].gated = False

        self.gate_time += time.time() - gate_start

        self.total_extracted_features += len(non_gated_detecions)
        if non_gated_detecions and not opt.offline:
            embedding_start = time.time()
            features = iter(self.embedding.inference(non_gated_detecions, frame))
            self.embedding_time += time.time() - embedding_start
            for i, det in enumerate(detections):
                if not det.gated:
                    det.feature = next(features)        

        match_start = time.time()
        # Associate confirmed tracks using appearance features.
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        self.match_time += time.time() - match_start
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        self.tracks.append(Track(
            detection.to_xyah(), self._next_id, self.n_init, self.max_age,
            detection.feature, detection.confidence))
        self._next_id += 1
