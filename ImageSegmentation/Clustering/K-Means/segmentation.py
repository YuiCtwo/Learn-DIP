import numpy as np
from scipy.spatial.distance import squareform, pdist, cdist
from time import time
import matplotlib.pyplot as plt
from matplotlib import rc
from skimage import io


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
        assign_idx = np.argmin(dis, axis=0)
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


if __name__ == '__main__':
    plt.rcParams['figure.figsize'] = (15.0, 12.0)  # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'

    # Generate random data points for clustering

    # Set seed for consistency
    np.random.seed(0)

    # Cluster 1
    mean1 = [-1, 0]
    cov1 = [[0.1, 0], [0, 0.1]]
    X1 = np.random.multivariate_normal(mean1, cov1, 100)

    # Cluster 2
    mean2 = [0, 1]
    cov2 = [[0.1, 0], [0, 0.1]]
    X2 = np.random.multivariate_normal(mean2, cov2, 100)

    # Cluster 3
    mean3 = [1, 0]
    cov3 = [[0.1, 0], [0, 0.1]]
    X3 = np.random.multivariate_normal(mean3, cov3, 100)

    # Cluster 4
    mean4 = [0, -1]
    cov4 = [[0.1, 0], [0, 0.1]]
    X4 = np.random.multivariate_normal(mean4, cov4, 100)

    # Merge two sets of data points
    X = np.concatenate((X1, X2, X3, X4))

    np.random.seed(0)
    start = time()
    assignments = kmeans_fast(X, 4)
    end = time()

    kmeans_fast_runtime = end - start
    print("kmeans running time: %f seconds." % kmeans_fast_runtime)
    for i in range(4):
        cluster_i = X[assignments == i]
        plt.scatter(cluster_i[:, 0], cluster_i[:, 1])

    plt.axis('equal')
    plt.show()
