import numpy as np
from utils import pad


def ransac(keypoints1, keypoints2, matches, n_iters=200, threshold=20):
    """
    Use RANSAC to find a robust affine transformation:

        1. Select random set of matches
        2. Compute affine transformation matrix
        3. Compute inliers via Euclidean distance
        4. Keep the largest set of inliers
        5. Re-compute least-squares estimate on all of the inliers

    Update max_inliers as a boolean array where True represents the keypoint
    at this index is an inlier, while False represents that it is not an inlier.

    Hint:
        You can use np.linalg.lstsq function to solve the problem.

        Explicitly specify np.linalg.lstsq's new default parameter rcond=None
        to suppress deprecation warnings, and match the autograder.

        You can compute elementwise boolean operations between two numpy arrays,
        and use boolean arrays to select array elements by index:
        https://numpy.org/doc/stable/reference/arrays.indexing.html#boolean-array-indexing

    Args:
        keypoints1: M1 x 2 matrix, each row is a point
        keypoints2: M2 x 2 matrix, each row is a point
        matches: N x 2 matrix, each row represents a match
            [index of keypoint1, index of keypoint 2]
        n_iters: the number of iterations RANSAC will run
        threshold: the number of threshold to find inliers

    Returns:
        H: a robust estimation of affine transformation from keypoints2 to
        key_points 1
    """
    # Copy matches array, to avoid overwriting it
    tmp_matches = matches.copy()

    N = matches.shape[0]
    n_samples = int(N * 0.2)

    max_inliers = np.zeros(N, dtype=bool)
    max_n_inliers = 0
    # RANSAC iteration start
    for it in range(n_iters):
        np.random.shuffle(tmp_matches)
        samples = tmp_matches[:n_samples]
        X2 = pad(keypoints1[samples[:, 0]])
        X1 = pad(keypoints2[samples[:, 1]])
        H = np.linalg.lstsq(X2, X1, rcond=None)[0]
        # set inliers to 1, outliers to 0
        tmp_inliers = np.where(np.linalg.norm(X2 * H - X1) <= threshold, 1, 0)
        n_inliers = np.count_nonzero(max_inliers)
        # preserve the largest one
        if max_n_inliers < n_inliers:
            max_inliers = tmp_inliers.copy()
    # recompute use the inliers
    res_matches = matches[max_inliers]
    X2 = pad(keypoints1[res_matches[:, 0]])
    X1 = pad(keypoints2[res_matches[:, 1]])
    H = np.linalg.lstsq(X2, X1, rcond=None)
    return H, res_matches


