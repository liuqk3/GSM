import numpy as np

from scipy.spatial.distance import cdist
# from sklearn.utils import linear_assignment_
import lapjv
from scipy.optimize import linear_sum_assignment
from lib.utils.cython_bbox import bbox_ious
from lib.utils import kalman_filter
from lib.utils.bbox import box_dist


def indices_to_matches(cost_matrix, indices, thresh):
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = (matched_cost <= thresh)

    matches = indices[matched_mask]
    unmatched_a = list(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = list(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def linear_assignment(cost_matrix, thresh, tool='lapjv'):
    """
    Simple linear assignment
    :type cost_matrix: np.ndarray
    :type thresh: float
    :return: matches, unmatched_a, unmatched_b
    """
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))

    cost_matrix[cost_matrix > thresh] = thresh + 1e-4

    if tool == 'scipy':
        indices = linear_sum_assignment(cost_matrix)
        if isinstance(indices, tuple):
            indices = np.array(indices).transpose() # [num_assign, 2], the first row is the index of tracks, and the second is the index of detections
    elif tool == 'lapjv':
        # pad the cost matrix to a square matrix for tool lapjv
        num_track, num_det = cost_matrix.shape
        if num_track > num_det:
            pad = np.zeros((num_track, num_track - num_det))
            pad.fill(1e5)
            cost_matrix_pad = np.concatenate((cost_matrix, pad), axis=1)

        elif num_det > num_track:
            pad = np.zeros((num_det - num_track, num_det))
            pad.fill(1e5)
            cost_matrix_pad = np.concatenate((cost_matrix, pad), axis=0)
        else:
            cost_matrix_pad = cost_matrix
        ass = lapjv.lapjv(cost_matrix_pad)
        r_ass, c_ass = ass[0], ass[1]
        if num_track <= num_det:
            r_ass = r_ass[0:num_track]
            indices = np.arange(0, num_track)
            indices = np.array([indices, r_ass]).transpose() # [num_assign, 2]
        else:
            c_ass = c_ass[0:num_det]
            indices = np.arange(0, num_det)
            indices = np.array([c_ass, indices]).transpose() # [num_assign, 2]

    return indices_to_matches(cost_matrix, indices, thresh)


def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if ious.size == 0:
        return ious

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float),
        np.ascontiguousarray(btlbrs, dtype=np.float)
    )

    return ious


def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """
    atlbrs = [track.tlbr() for track in atracks]
    btlbrs = [track.tlbr() for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix


def nearest_reid_distance(tracks, detections, metric='cosine', use_history=True):
    """
    Compute cost based on ReID features
    :type tracks: list[STrack]
    :type detections: list[BaseTrack]
    :type use_history: bool, whether to use the history features of tracks
    :rtype cost_matrix np.ndarray
    """
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix

    det_features = np.asarray([track.curr_feature for track in detections], dtype=np.float32)
    for i, track in enumerate(tracks):
        if use_history:
            dist_tmp = cdist(track.features, det_features, metric)
        else:
            dist_tmp = cdist([track.curr_feature], det_features, metric)
        dist_tmp = dist_tmp.min(axis=0)
        cost_matrix[i, :] = np.maximum(0.0, dist_tmp)

    return cost_matrix


def mean_reid_distance(tracks, detections, metric='cosine'):
    """
    Compute cost based on ReID features
    :type tracks: list[STrack]
    :type detections: list[BaseTrack]
    :type metric: str

    :rtype cost_matrix np.ndarray
    """
    cost_matrix = np.empty((len(tracks), len(detections)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix

    track_features = np.asarray([track.curr_feature for track in tracks], dtype=np.float32)
    det_features = np.asarray([track.curr_feature for track in detections], dtype=np.float32)
    cost_matrix = cdist(track_features, det_features, metric)

    return cost_matrix


def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
    if cost_matrix.size == 0:
        return cost_matrix
    if kf is not None:
        # gating the cost matrix based on kalman filter
        gating_dim = 2 if only_position else 4
        gating_threshold = kalman_filter.chi2inv95[gating_dim]
        # gating_threshold = kalman_filter.chi2inv99[gating_dim]
        measurements = np.asarray([det.to_xyah() for det in detections])
        for row, track in enumerate(tracks):
            gating_distance = kf.gating_distance(track.mean, track.covariance, measurements, only_position)

            if track.is_lost():
                release = 1. / (1 + np.power(np.e, 1-track.time_since_update))
                release = np.power(np.e, release - 0.5)
                gating_threshold = gating_threshold * release

            cost_matrix[row, gating_distance > gating_threshold] = np.inf
    else:
        # gating the cost matrix based on euclidean distance
        xy_norm_t = np.array([t.xy_norm() for t in tracks])
        xy_norm_d = np.array([d.xy_norm() for d in detections])
        dist = xy_norm_t[:, np.newaxis, :] - xy_norm_d[np.newaxis, :, :] # [num_t, num_d, 2]
        dist = (dist * dist).sum(-1) # [num_t, num_d]
        dist = dist / 2 # # normalize to the diagonal length of image
        dist = np.power(dist, 0.5)
        cost_matrix[dist > 0.2] = np.inf

    return cost_matrix


if __name__ == '__main__':

    a = int(5)
    b = int(2)

    print(a/b)


    a = 1




