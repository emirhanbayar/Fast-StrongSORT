# vim: expandtab:ts=4:sw=4
from __future__ import division, print_function, absolute_import

import argparse
import os

import cv2
import numpy as np

import PIL
from application_util import preprocessing
from application_util import visualization
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort.ecc import ECC
from opts import opt
import time


def gather_sequence_info(sequence_dir, detection_file):
    """Gather sequence information, such as image filenames, detections,
    groundtruth (if available).

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detection file.

    Returns
    -------
    Dict
        A dictionary of the following sequence information:

        * sequence_name: Name of the sequence
        * image_filenames: A dictionary that maps frame indices to image
          filenames.
        * detections: A numpy array of detections in MOTChallenge format.
        * groundtruth: A numpy array of ground truth in MOTChallenge format.
        * image_size: Image size (height, width).
        * min_frame_idx: Index of the first frame.
        * max_frame_idx: Index of the last frame.

    """
    image_dir = os.path.join(sequence_dir, "img1")
    image_filenames = {
        int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
        for f in os.listdir(image_dir)}
    groundtruth_file = os.path.join(sequence_dir, "gt/gt.txt")

    detections = None
    if detection_file is not None:
        detections = np.load(detection_file)
    groundtruth = None
    if os.path.exists(groundtruth_file):
        groundtruth = np.loadtxt(groundtruth_file, delimiter=',')

    if len(image_filenames) > 0:
        image = cv2.imread(next(iter(image_filenames.values())),
                           cv2.IMREAD_GRAYSCALE)
        image_size = image.shape
    else:
        image_size = None

    if len(image_filenames) > 0:
        min_frame_idx = min(image_filenames.keys())
        max_frame_idx = max(image_filenames.keys())
    else:
        min_frame_idx = int(detections[:, 0].min())
        max_frame_idx = int(detections[:, 0].max())

    info_filename = os.path.join(sequence_dir, "seqinfo.ini")
    if os.path.exists(info_filename):
        with open(info_filename, "r") as f:
            line_splits = [l.split('=') for l in f.read().splitlines()[1:]]
            info_dict = dict(
                s for s in line_splits if isinstance(s, list) and len(s) == 2)

        update_ms = 1000 / int(info_dict["frameRate"])
    else:
        update_ms = None

    feature_dim = detections.shape[1] - 10 if detections is not None else 0
    seq_info = {
        "sequence_name": os.path.basename(sequence_dir),
        "image_filenames": image_filenames,
        "detections": detections,
        "groundtruth": groundtruth,
        "image_size": image_size,
        "min_frame_idx": min_frame_idx,
        "max_frame_idx": max_frame_idx,
        "feature_dim": feature_dim,
        "update_ms": update_ms
    }
    return seq_info


def create_detections(detection_mat, frame_idx, min_height=0):
    """Create detections for given frame index from the raw detection matrix.

    Parameters
    ----------
    detection_mat : ndarray
        Matrix of detections. The first 10 columns of the detection matrix are
        in the standard MOTChallenge detection format. In the remaining columns
        store the feature vector associated with each detection.
    frame_idx : int
        The frame index.
    min_height : Optional[int]
        A minimum detection bounding box height. Detections that are smaller
        than this value are disregarded.

    Returns
    -------
    List[tracker.Detection]
        Returns detection responses at given frame index.

    """
    frame_indices = detection_mat[:, 0].astype(np.int64)
    mask = frame_indices == frame_idx

    detection_list = []
    for row in detection_mat[mask]:
        bbox, confidence = row[2:6], row[6]
        if bbox[3] < min_height:
            continue
        if opt.offline:
            feature = row[10:]
            detection_list.append(Detection(bbox, confidence, feature))
        else:
            detection_list.append(Detection(bbox, confidence, None))

    return detection_list

def visualize(dets, tracklet_means, tracklet_bboxes, img):
    # draw traclket bboxes, dets bboxes and tracklet means
    for tracklet_bbox in tracklet_bboxes:
        x1, y1, w, h = tracklet_bbox[:4]
        cv2.rectangle(
            img,
            (int(x1), int(y1)),
            (int(x1 + w), int(y1 + h)),
            (0, 255, 0),
            1
        )

    for det in dets:
        x1, y1, w, h = det.tlwh
        if not det.gated:
            cv2.rectangle(
                img,
                (int(x1), int(y1)),
                (int(x1 + w), int(y1 + h)),
                (0, 255, 255),
                1
            )
        else:
            cv2.rectangle(
                img,
                (int(x1), int(y1)),
                (int(x1 + w), int(y1 + h)),
                (255, 0, 0),
                1
            )

    for tracklet_mean in tracklet_means:
        cv2.circle(
            img,
            (int(tracklet_mean[0]), int(tracklet_mean[1])),
            10,
            (0, 0, 255),
            1
        )

    #add legend
    cv2.putText(
        img,
        "Tracklet bbox",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1
    )
    cv2.putText(
        img,
        "gated detection bbox",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 0, 0),
        1
    )

    cv2.putText(
        img,
        "Tracklet mean",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        1
    )

    cv2.putText(
        img,
        "Non-gated detection",
        (10, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1
    )

def run(sequence_dir, detection_file, output_file, min_confidence,
        nms_max_overlap, min_detection_height, max_cosine_distance,
        nn_budget, display):
    """Run multi-target tracker on a particular sequence.

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detections file.
    output_file : str
        Path to the tracking output file. This file will contain the tracking
        results on completion.
    min_confidence : float
        Detection confidence threshold. Disregard all detections that have
        a confidence lower than this value.
    nms_max_overlap: float
        Maximum detection overlap (non-maxima suppression threshold).
    min_detection_height : int
        Detection height threshold. Disregard all detections that have
        a height lower than this value.
    max_cosine_distance : float
        Gating threshold for cosine distance metric (object appearance).
    nn_budget : Optional[int]
        Maximum size of the appearance descriptor gallery. If None, no budget
        is enforced.
    display : bool
        If True, show visualization of intermediate tracking results.

    """
    seq_info = gather_sequence_info(sequence_dir, detection_file)
    metric = nn_matching.NearestNeighborDistanceMetric(
        'cosine',
        max_cosine_distance,
        nn_budget
    )
    tracker = Tracker(metric)
    results = []
    cmc = ECC()

    def frame_callback(vis, frame_idx):
        # print("Processing frame %05d" % frame_idx)

        # Load image and generate detections.
        detections = create_detections(
            seq_info["detections"], frame_idx, min_detection_height)
        detections = [d for d in detections if d.confidence >= min_confidence]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(
            boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        if len(detections) == 0:
            return
    
        read_start = time.time()
        frame = PIL.Image.open(seq_info["image_filenames"][frame_idx])
        cv2_frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        tracker.read_time += time.time() - read_start

        ecc_start = time.time()
        # Update tracker.
        if opt.ECC:
            if len(tracker.tracks) >= 1:
                warp_matrix = cmc.apply(cv2_frame, [d.to_tlbr() for d in detections])
                for track in tracker.tracks:
                    track.camera_update(warp_matrix)
        tracker.ecc_time += time.time() - ecc_start

        predict_start = time.time()
        tracker.predict()
        tracker.prediction_time += time.time() - predict_start

        tracker.update(detections, frame)

        time_since_update_threshold = 0 if opt.dataset == 'dancetrack' else 1
        # Store results.
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > time_since_update_threshold:
                continue
            bbox = track.to_tlwh()
            results.append([
                    frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])
            
            if display:
                # write ids on the image
                scale = 1.0 * cv2_frame.shape[0] / 1080
                cv2.putText(
                    cv2_frame,
                    str(track.track_id),
                    (int(bbox[0]), int(bbox[1])),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    scale,
                    (0, 0, 255),
                    2
                )

        # Update visualization.
        if display and len(results) > 0:
            tracklet_means = [t.mean for t in tracker.tracks if t.is_confirmed()]
            tracklet_bboxes = [t.to_tlwh() for t in tracker.tracks if t.is_confirmed()]
            visualize(detections, tracklet_means, tracklet_bboxes, cv2_frame)

            os.makedirs(f'output/{opt.dataset}_{opt.mode}_{opt.aiou_threshold:.2f}_{opt.occlusion_threshold:.2f}_{opt.decay_ema}_{opt.prematch}/{sequence_dir.split("/")[-1]}/', exist_ok=True)
            cv2.imwrite(f'output/{opt.dataset}_{opt.mode}_{opt.aiou_threshold:.2f}_{opt.occlusion_threshold:.2f}_{opt.decay_ema}_{opt.prematch}/{sequence_dir.split("/")[-1]}/{frame_idx:05d}.jpg', cv2_frame)

    visualizer = visualization.NoVisualization(seq_info)
    visualizer.run(frame_callback)

    # Store results.
    os.makedirs(f'output/{opt.dataset}_{opt.mode}_{opt.aiou_threshold:.2f}_{opt.occlusion_threshold:.2f}_{opt.decay_ema}_{opt.prematch}/data/', exist_ok=True)
    f = open(f'output/{opt.dataset}_{opt.mode}_{opt.aiou_threshold:.2f}_{opt.occlusion_threshold:.2f}_{opt.decay_ema}_{opt.prematch}/data/{sequence_dir.split("/")[-1]}.txt', 'w')
    for row in results:
        print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
            row[0], row[1], row[2], row[3], row[4], row[5]),file=f)
    f.close()

    return tracker.total_extracted_features, tracker.total_detections, tracker.total_frames, tracker.ecc_time, tracker.prediction_time, tracker.gate_time, tracker.match_time, tracker.track_management_time, tracker.embedding_time

def bool_string(input_string):
    if input_string not in {"True","False"}:
        raise ValueError("Please Enter a valid Ture/False choice")
    else:
        return (input_string == "True")

def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Deep SORT")
    parser.add_argument(
        "--sequence_dir", help="Path to MOTChallenge sequence directory",
        default=None, required=True)
    parser.add_argument(
        "--detection_file", help="Path to custom detections.", default=None,
        required=True)
    parser.add_argument(
        "--output_file", help="Path to the tracking output file. This file will"
        " contain the tracking results on completion.",
        default="/tmp/hypotheses.txt")
    parser.add_argument(
        "--min_confidence", help="Detection confidence threshold. Disregard "
        "all detections that have a confidence lower than this value.",
        default=0.8, type=float)
    parser.add_argument(
        "--min_detection_height", help="Threshold on the detection bounding "
        "box height. Detections with height smaller than this value are "
        "disregarded", default=0, type=int)
    parser.add_argument(
        "--nms_max_overlap",  help="Non-maxima suppression threshold: Maximum "
        "detection overlap.", default=1.0, type=float)
    parser.add_argument(
        "--max_cosine_distance", help="Gating threshold for cosine distance "
        "metric (object appearance).", type=float, default=0.2)
    parser.add_argument(
        "--nn_budget", help="Maximum size of the appearance descriptors "
        "gallery. If None, no budget is enforced.", type=int, default=None)
    parser.add_argument(
        "--display", help="Show intermediate tracking results",
        default=True, type=bool_string)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        args.sequence_dir, args.detection_file, args.output_file,
        args.min_confidence, args.nms_max_overlap, args.min_detection_height,
        args.max_cosine_distance, args.nn_budget, args.display)
