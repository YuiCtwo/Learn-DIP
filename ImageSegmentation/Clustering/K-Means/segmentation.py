import numpy as np
from scipy.spatial.distance import squareform, pdist, cdist


# Clustering Methods
def kmeans(features, k, num_iters=100):
    """ Use kmeans algorithm to group features into k clusters.

    K-Means algorithm can be broken down into following steps:
        1. Randomly initialize cluster centers
        2. Assign each point to the closest center
        3. Compute new center of each cluster
        4. Stop if cluster assignments did not change
        5. Go to step 2

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.
        num_iters - Maximum number of iterations the algorithm will run.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """

    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Randomly initalize cluster centers
    idxs = np.random.choice(N, size=k, replace=False)
    centers = features[idxs]
    center_sum = np.zeros((k, D), dtype=np.uint32)
    center_num = np.zeros(k, dtype=np.uint32)
    assignments = np.zeros(N, dtype=np.uint32)
    assignments_tmp = np.zeros(N, dtype=np.uint32)

    for n in range(num_iters):
        for i in range(N):
            min_idx = 0
            min_val = 1e9
            for kk in range(k):
                dis = np.linalg.norm(centers[k] - features[i, :])
                if dis < min_val:
                    min_idx = kk
                    min_val = dis
            center_sum[min_idx, :] += features[i, :]
            center_num[min_idx] += 1
            assignments_tmp[i] = min_idx
        if assignments_tmp.all() == assignments.all():
            # 收敛, 停止迭代
            break
        else:
            assignments = assignments_tmp.copy()
            # 重新计算中心点
            for kk in range(k):
                centers[kk] = center_sum[kk] / center_num[kk]
                # 重置
                center_sum[kk] = 0
                center_num[kk] = 0




    return assignments


def kmeans_fast(features, k, num_iters=100):
    """ Use kmeans algorithm to group features into k clusters.

    This function makes use of numpy functions and broadcasting to speed up the
    first part(cluster assignment) of kmeans algorithm.

    Hints
    - You may find cdist (imported from scipy.spatial.distance) and np.argmin useful

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.
        num_iters - Maximum number of iterations the algorithm will run.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """

    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Randomly initalize cluster centers
    idxs = np.random.choice(N, size=k, replace=False)
    centers = features[idxs]
    assignments = np.zeros(N, dtype=np.uint32)

    for n in range(num_iters):
        dis = cdist(centers, features)
        assign_idx = np.argmin(dis, axis=1)
        if assign_idx.all() == assignments.all():
            break
        else:
            assignments = assign_idx
        center_sum = np.zeros((k, D), dtype=np.uint32)
        center_num = np.zeros(k, dtype=np.uint32)
        for i in range(N):
            center_sum[assign_idx[i]] += features[i]
            center_num[assign_idx[i]] += 1
        for kk in range(k):
            centers[kk] = center_sum[kk] / center_num[kk]
    return assignments