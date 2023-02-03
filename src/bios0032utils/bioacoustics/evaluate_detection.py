"""Evaluate detection results."""
import os
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def maximal_matching(matrix: np.ndarray, maximize=True) -> pd.DataFrame:
    """
    Extract a Maximal Matching from a weighted cost matrix.

    The input is a matrix M that stores the edge weights of some bipartite
    graph.

    Parameters
    ----------
    matrix: array, shape = [n, m]

    Returns
    -------
    data: pd.DataFrame
    """
    rows, cols = linear_sum_assignment(matrix, maximize=maximize)
    return pd.DataFrame(
        {
            "annotation": rows,
            "prediction": cols,
            "affinity": matrix[rows, cols],
        }
    )


def interval_intersection(interval1: np.ndarray, interval2: np.ndarray) -> np.ndarray:
    """Compute the intersection of a pair of array of intervals.

    Intervals are in the format [start, end].
    Args:
        interval1: Array of intervals.
        interval2: Array of intervals.
    """
    starts = np.maximum(interval1[:, 0][:, None], interval2[:, 0][None, :])
    ends = np.minimum(interval1[:, 1][:, None], interval2[:, 1][None, :])
    return np.maximum(ends - starts, 0)


def bbox_area(bbox: np.ndarray) -> np.ndarray:
    """Compute the area of a set of bounding boxes.

    Bounding boxes are in the format [left, top, right, bottom].

    Args:
        bbox: Array of bounding boxes.
    """
    return (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])


def bbox_iou(bbox1: np.ndarray, bbox2: np.ndarray):
    """Compute the intersection over union of a pair of bounding boxes.

    Bounding boxes are in the format [left, top, right, bottom].

    Args:
        bbox1: Array of bounding boxes.
        bbox2: Array of bounding boxes.
    """

    area1 = bbox_area(bbox1)

    area2 = bbox_area(bbox2)

    xintersection = interval_intersection(
        bbox1[:, np.array([0, 2])],
        bbox2[:, np.array([0, 2])],
    )

    yintersection = interval_intersection(
        bbox1[:, np.array([1, 3])],
        bbox2[:, np.array([1, 3])],
    )

    intersection_area = xintersection * yintersection

    union_area = area1[:, None] + area2[None, :] - intersection_area  # type: ignore

    return intersection_area / union_area


def bboxes_from_tadarida_detections(detections: pd.DataFrame) -> np.ndarray:
    """Get bounding boxes from Tadarida detections.

    Tadarida detections are a dataframe with the following columns:
        StTime: Start time of the detection in milliseconds.
        Dur: Duration of the detection in milliseconds.
        Fmin: Minimum frequency of the detection in kHz.
        BW: Bandwidth of the detection in kHz.

    Args:
        detections: Dataframe with Tadarida detections.

    Returns:
        Array of bounding boxes. Bounding boxes are in the format
        [left, top, right, bottom].
    """
    # Create a dataframe and convert to numpy array
    bboxes = detections[["StTime", "Fmin", "Dur", "BW"]].copy()
    bboxes["StTime"] = bboxes["StTime"] / 1000
    bboxes["Fmin"] = bboxes["Fmin"] * 1000
    bboxes["Dur"] = bboxes["Dur"] / 1000
    bboxes["BW"] = bboxes["BW"] * 1000
    bboxes["End"] = bboxes["StTime"] + bboxes["Dur"]
    bboxes["Fmax"] = bboxes["Fmin"] + bboxes["BW"]
    return bboxes[["StTime", "Fmin", "End", "Fmax"]].values


def bboxes_from_annotations(annotations: pd.DataFrame) -> np.ndarray:
    """Get bounding boxes of a file from annotations.

    Annotations are a dataframe with the following columns:
        start_time: Start time of the annotation in seconds.
        end_time: End time of the annotation in seconds.
        low_freq: Minimum frequency of the annotation in Hz.
        high_freq: Maximum frequency of the annotation in Hz.
        recording_id: Name of the recording.

    Args:
        annotations: Dataframe with annotations.

    Returns:
        Array of bounding boxes. Bounding boxes are in the format
        [left, top, right, bottom].
    """
    # Create a dataframe and convert to numpy array
    bboxes = annotations[["start_time", "low_freq", "end_time", "high_freq"]].copy()
    return bboxes.values


def match_bboxes(
    true_bboxes: np.ndarray,
    pred_bboxes: np.ndarray,
    iou_threshold: float = 0.5,
) -> pd.DataFrame:
    """Match bounding boxes.

    Will only return matches with an intersection over union greater than
    the threshold.

    Args:
        true_bbox: Array of bounding boxes.
        pred_bbox: Array of bounding boxes.
        iou_threshold: Threshold for the intersection over union.

    Returns:
        Dataframe with the following columns:
            annotation: Index of the annotation.
            prediction: Index of the prediction.
            affinity: Intersection over union.
    """
    iou = bbox_iou(true_bboxes, pred_bboxes)
    matches = maximal_matching(iou)
    return matches[matches["affinity"] > iou_threshold]


def compute_file_precision_recall(
    filename: str,
    detections: pd.DataFrame,
    annotations: pd.DataFrame,
    iou_threshold: float = 0.5,
) -> Tuple[float, float]:
    """Compute the precision and recall for a file.

    Args:
        filename: Name of the file.
        detections: Dataframe with detections.
        annotations: Dataframe with annotations.

    Returns:
        Precision and recall.
    """
    # Select the predictions and annotations from the crowded recording
    file_detections = detections[detections.recording_id == os.path.basename(filename)]
    file_annotations = annotations[
        annotations.recording_id == os.path.basename(filename)
    ]

    # Match the bounding boxes by computing the IoU. Discard all matches with IoU less than 0.5
    pred_boxes = bboxes_from_annotations(file_detections)
    true_boxes = bboxes_from_annotations(file_annotations)
    matches = match_bboxes(true_boxes, pred_boxes, iou_threshold=iou_threshold)

    # total number of annotated sound events
    positives = len(file_annotations)

    num_predictions = len(file_detections)

    # number of matched prediction boxes
    true_positives = len(matches)

    if num_predictions == 0:
        precision = np.nan
    else:
        # Percentage of predictions that are correct
        precision = true_positives / num_predictions

    if positives == 0:
        recall = np.nan
    else:
        # Percentage of sound events that were detected
        recall = true_positives / positives

    return precision, recall


def compute_detection_metrics(
    filename: str,
    detections: pd.DataFrame,
    annotations: pd.DataFrame,
    iou_threshold: float = 0.5,
) -> Tuple[float, float, float, float]:
    """Compute the precision and recall for a file.

    Args:
        filename: Name of the file.
        detections: Dataframe with detections.
        annotations: Dataframe with annotations.

    Returns:
        positives, true_positives, false_positives, false_negatives
    """
    # Select the predictions and annotations from the crowded recording
    file_detections = detections[detections.recording_id == os.path.basename(filename)]
    file_annotations = annotations[
        annotations.recording_id == os.path.basename(filename)
    ]

    # Match the bounding boxes by computing the IoU. Discard all matches with IoU less than 0.5
    pred_boxes = bboxes_from_annotations(file_detections)
    true_boxes = bboxes_from_annotations(file_annotations)
    matches = match_bboxes(true_boxes, pred_boxes, iou_threshold=iou_threshold)

    # total number of annotated sound events
    positives = len(file_annotations)

    num_predictions = len(file_detections)

    # number of matched prediction boxes
    true_positives = len(matches)

    false_positives = num_predictions - true_positives

    false_negatives = positives - true_positives

    return positives, true_positives, false_positives, false_negatives
