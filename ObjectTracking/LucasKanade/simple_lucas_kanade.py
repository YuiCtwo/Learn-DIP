import numpy as np
from skimage.transform import pyramid_gaussian


def lucas_kanade(img1, img2, keypoints, window_size=5):
    """Estimate flow vector at each keypoint using Lucas-Kanade method.

    Args:
        img1 - Grayscale image of the current frame. Flow vectors are computed
            with respect to this frame.
        img2 - Grayscale image of the next frame.
        key_points - Keypoints to track. Numpy array of shape (N, 2).
        window_size - Window size to determine the neighborhood of each keypoint.
            A window is centered around the current keypoint location.
            You may assume that window_size is always an odd number.
    Returns:
        flow_vectors - Estimated flow vectors for key_points. flow_vectors[i] is
            the flow vector for keypoint[i]. Numpy array of shape (N, 2).
    """
    assert window_size % 2 == 1, "window_size must be an odd number"

    flow_vectors = []
    w = window_size // 2

    # Compute partial derivatives
    Iy, Ix = np.gradient(img1)
    It = img2 - img1
    # For each [y, x] in key_points, estimate flow vector [vy, vx]
    # using Lucas-Kanade method and append it to flow_vectors.
    for y, x in keypoints:
        # Keypoints can be located between integer pixels (subpixel locations).
        # For simplicity, we round the keypoint coordinates to nearest integer.
        # In order to achieve more accurate results, image brightness at subpixel
        # locations can be computed using bilinear interpolation.
        y_coord, x_coord = int(round(y)), int(round(x))
        A = np.zeros((window_size * window_size, 2))
        b = np.zeros((window_size * window_size, 1))
        i = 0
        for hi in range(y_coord - w, y_coord + w + 1):
            for wi in range(x_coord - w, x_coord + w + 1):
                A[i][0] = Ix[hi][wi]
                A[i][1] = Iy[hi][wi]
                b[i][0] = -It[hi][wi]
                i += 1
        # transform to square matrix for inv()
        A_square = np.dot(A.T, A)
        v = np.dot(np.linalg.inv(A_square), A.T).dot(b)
        flow_vectors.append(v.T.tolist()[0])
    flow_vectors = np.array(flow_vectors)

    return flow_vectors  # [[vx_1, vy_1], ...]
