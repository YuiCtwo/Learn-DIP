import numpy as np
import matplotlib.pyplot as plt

from skimage import filters
from skimage.io import imread
from skimage.feature import corner_peaks
from scipy.spatial.distance import cdist
from scipy.ndimage.filters import convolve
from utils import plot_matches


def harris_corners(img, window_size=3, k=0.04):
    """
    Compute Harris corner response map. Follow the math equation
    R=Det(M)-k(Trace(M)^2).

    Hint:
        You may use the function scipy.ndimage.filters.convolve,
        which is already imported above. If you use convolve(), remember to
        specify zero-padding to match our equations, for example:

            out_image = convolve(in_image, kernel, mode='constant', cval=0)

        You can also use for nested loops compute M and the subsequent Harris
        corner response for each output pixel, instead of using convolve().
        Your implementation of conv_fast or conv_nested in HW1 may be a
        useful reference!

    Args:
        img: Grayscale image of shape (H, W)
        window_size: size of the window function
        k: sensitivity parameter

    Returns:
        response: Harris response image of shape (H, W)
    """

    H, W = img.shape
    window = np.ones((window_size, window_size))

    response = np.zeros((H, W))

    # 1. Compute x and y derivatives (I_x, I_y) of an image
    dx = filters.sobel_v(img)
    dy = filters.sobel_h(img)

    # 2. Compute I_x^2, I_y^2, I_x*I_y
    dx2 = dx * dx
    dy2 = dy * dy
    dx_dy = dx * dy

    # 3. compute M
    # If lacked, use 0 filling
    M_dx2 = convolve(dx2, window, mode='constant', cval=0)
    M_dy2 = convolve(dy2, window, mode='constant', cval=0)
    M_dx_dy = convolve(dx_dy, window, mode='constant', cval=0)

    # 4. nested loop to compute R
    for h in range(H):
        for w in range(W):
            M = np.array([
                [M_dx2[h, w], M_dx_dy[h, w]],
                [M_dx_dy[h, w], M_dy2[h, w]]
            ])
            response[h, w] = np.linalg.det(M) - k * np.trace(M * M)

    return response


def simple_descriptor(patch):
    """
    Describe the patch by normalizing the image values into a standard
    normal distribution (having mean of 0 and standard deviation of 1)
    and then flattening into a 1D array.

    The normalization will make the descriptor more robust to change
    in lighting condition.

    Hint:
        In this case of normalization, if a denominator is zero, divide by 1 instead.

    Args:
        patch: grayscale image patch of shape (H, W)

    Returns:
        feature: 1D array of shape (H * W)
    """
    H, W = patch.shape
    feature = np.resize(patch, (H * W, 1))

    mu = np.mean(feature)
    sigma = np.std(feature)
    if sigma == 0:
        sigma = 1
    feature = (feature - mu) / sigma
    return feature


def describe_keypoints(image, keypoints, desc_func, patch_size=16):
    """
    Args:
        image: grayscale image of shape (H, W)
        keypoints: 2D array containing a keypoint (y, x) in each row
        desc_func: function that takes in an image patch and outputs
            a 1D feature vector describing the patch
        patch_size: size of a square patch at each keypoint

    Returns:
        desc: array of features describing the key_points
    """

    image.astype(np.float32)
    desc = []

    for i, kp in enumerate(keypoints):
        y, x = kp
        patch = image[y - (patch_size // 2):y + ((patch_size + 1) // 2),
                x - (patch_size // 2):x + ((patch_size + 1) // 2)]
        desc.append(desc_func(patch))
    # h, w = len(desc), len(desc[0])
    res = np.asarray(desc)
    return res


def match_descriptors(desc1, desc2, threshold=0.5):
    """
    Match the feature descriptors by finding distances between them. A match is formed
    when the distance to the closest vector is much smaller than the distance to the
    second-closest, that is, the ratio of the distances should be strictly smaller
    than the threshold (not equal to). Return the matches as pairs of vector indices.

    Hint:
        The Numpy functions np.sort, np.argmin, np.asarray might be useful

        The Scipy function cdist calculates Euclidean distance between all
        pairs of inputs
    Args:
        desc1: an array of shape (M, P) holding descriptors of size P about M key_points
        desc2: an array of shape (N, P) holding descriptors of size P about N key_points

    Returns:
        matches: an array of shape (Q, 2) where each row holds the indices of one pair
        of matching descriptors
    """
    match = []

    M = desc1.shape[0]
    # there has some bugs with the array's shape
    # if len(desc1.shape) == 3:
    #     new_des1 = desc1.reshape((desc1.shape[0], desc1.shape[1]))
    # elif len(desc1.shape) == 1:
    #     new_des1 = desc1.reshape((desc1.shape[0], 1))
    # else:
    #     new_des1 = desc1
    # if len(desc2.shape) == 3:
    #     new_des2 = desc2.reshape((desc2.shape[0], desc2.shape[1]))
    # elif len(desc2.shape) == 1:
    #     new_des2 = desc2.reshape((desc2.shape[0], 1))
    # else:
    #     new_des2 = desc2
    # result array MxN, each one show the distance
    dists = cdist(desc1, desc2)
    sort_dists = np.argsort(dists)
    for m in range(M):
        # the closest is much small than the second
        idx1 = sort_dists[m, 0]
        idx2 = sort_dists[m, 1]
        if (dists[m, idx1] / dists[m, idx2]) <= threshold:
            match.append([m, idx1])

    return np.asarray(match)


if __name__ == '__main__':
    img1 = imread('uttower1.jpg', as_gray=True)
    img2 = imread('uttower2.jpg', as_gray=True)

    # Detect key_points in two images
    keypoints1 = corner_peaks(harris_corners(img1, window_size=3),
                              threshold_rel=0.05,
                              exclude_border=8)
    keypoints2 = corner_peaks(harris_corners(img2, window_size=3),
                              threshold_rel=0.05,
                              exclude_border=8)
    np.random.seed(131)
    patch_size = 5
    # use simple descriptor to test harris corner detector
    desc1 = describe_keypoints(img1, keypoints1,
                               desc_func=simple_descriptor,
                               patch_size=patch_size)
    desc2 = describe_keypoints(img2, keypoints2,
                               desc_func=simple_descriptor,
                               patch_size=patch_size)
    matches = match_descriptors(desc1, desc2, 0.7)
    fig, ax = plt.subplots(1, 1, figsize=(15, 12))
    ax.axis('off')
    plt.title('Matched Simple Descriptor')
    plot_matches(ax, img1, img2, keypoints1, keypoints2, matches)
    plt.show()
