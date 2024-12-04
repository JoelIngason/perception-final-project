import numpy as np
from scipy.optimize import linear_sum_assignment


def linear_assignment(cost_matrix: np.ndarray, thresh: float):
    """
    Perform linear assignment using the Hungarian algorithm.
    Only matches with cost below the threshold are accepted.

    Args:
        cost_matrix (np.ndarray): Cost matrix of shape (num_tracks, num_detections).
        thresh (float): Distance threshold for accepting matches.

    Returns:
        Tuple containing:
            - List of matched (track_idx, detection_idx) tuples.
            - Array of unmatched track indices.
            - Array of unmatched detection indices.

    """
    if cost_matrix.size == 0:
        return [], np.arange(cost_matrix.shape[0]), np.arange(cost_matrix.shape[1])

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matched_indices = []
    for r, c in zip(row_ind, col_ind, strict=False):
        if cost_matrix[r, c] > thresh:
            continue
        matched_indices.append((r, c))

    matched_track_idxs = set([m[0] for m in matched_indices])
    matched_det_idxs = set([m[1] for m in matched_indices])

    unmatched_tracks = np.array(
        [i for i in range(cost_matrix.shape[0]) if i not in matched_track_idxs],
    )
    unmatched_detections = np.array(
        [i for i in range(cost_matrix.shape[1]) if i not in matched_det_idxs],
    )

    return matched_indices, unmatched_tracks, unmatched_detections
