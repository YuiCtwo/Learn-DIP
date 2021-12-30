import numpy as np
from skimage.transform import pyramid_gaussian


def iterative_lucas_kanade(img1, img2, keypoints, window_size=9, num_iters=7, g=None):
    """Estimate flow vector at each keypoint using iterative Lucas-Kanade method.
    Args:
        img1 - Grayscale image of the current frame. Flow vectors are computed
            with respect to this frame.
        img2 - Grayscale image of the next frame.
        key_points - Keypoints to track. Numpy array of shape (N, 2).
        window_size - Window size to determine the neighborhood of each keypoint.
            A window is centered around the current keypoint location.
            You may assume that window_size is always an odd number.
        num_iters - Number of iterations to update flow vector.
        g - Flow vector guessed from previous pyramid level.
    Returns:
        flow_vectors - Estimated flow vectors for key_points. flow_vectors[i] is
            the flow vector for keypoint[i]. Numpy array of shape (N, 2).
    """
    assert window_size % 2 == 1, "window_size must be an odd number"

    # Initialize g as zero vector if not provided
    if g is None:
        g = np.zeros(keypoints.shape)

    flow_vectors = []
    w = window_size // 2

    # Compute spatial gradients
    Iy, Ix = np.gradient(img1)
    Ix2 = Ix ** 2
    Iy2 = Iy ** 2
    IxIy = np.multiply(Ix, Iy)
    for y, x, gx, gy in np.hstack((keypoints, g)):
        yy, xx = int(round(y)), int(round(x))
        v = np.zeros((2, 1))  # [vx, vy]^T
        G00 = np.sum(Ix2[yy - w:yy + w + 1, xx - w:xx + w + 1])
        G11 = np.sum(Iy2[yy - w:yy + w + 1, xx - w:xx + w + 1])
        G01 = np.sum(IxIy[yy - w:yy + w + 1, xx - w:xx + w + 1])
        G = np.array([
            [G00, G01],
            [G01, G11]
        ])

        # Iteratively update flow vector
        for k in range(num_iters):
            vx, vy = v[0][0], v[1][0]
            # Refined position of the point in the next frame
            y2 = int(round(yy + gy + vy))
            x2 = int(round(xx + gx + vx))

            bk = np.zeros((2, 1))
            for yi in range(yy - w, yy + w + 1):
                for xi in range(xx - w, xx + w + 1):
                    delta_Ik = img1[yi, xi] - img2[y2, x2]
                    bk[0][0] += (delta_Ik * Ix[yi, xi])
                    bk[1][0] += (delta_Ik * Iy[yi, xi])

            vk = np.dot(np.linalg.inv(G), bk)
            # Update flow vector by vk
            v += vk

        vx, vy = v[0][0], v[1][0]
        flow_vectors.append([vx, vy])

    return np.array(flow_vectors)


def pyramid_lucas_kanade(
        img1, img2, keypoints, window_size=9, num_iters=7, level=2, scale=2
):
    """Pyramidal Lucas Kanade method
    Args:
        img1 - same as lucas_kanade
        img2 - same as lucas_kanade
        keypoints - same as lucas_kanade
        window_size - same as lucas_kanade
        num_iters - number of iterations to run iterative LK method
        level - Max level in image pyramid. Original image is at level 0 of
            the pyramid.
        scale - scaling factor of image pyramid.
    Returns:
        d - final flow vectors
    """

    # Build image pyramids of img1 and img2
    pyramid1 = tuple(pyramid_gaussian(img1, max_layer=level, downscale=scale))
    pyramid2 = tuple(pyramid_gaussian(img2, max_layer=level, downscale=scale))

    # Initialize pyramidal guess
    g = np.zeros(keypoints.shape)  # g: [gx, gy]^T
    d = np.zeros(keypoints.shape)
    for L in range(level, -1, -1):
        keypoints_L = keypoints / (scale ** L)
        d = iterative_lucas_kanade(pyramid1[L], pyramid2[L],
                                   keypoints_L, window_size, num_iters, g)
        g = scale * (g + d)
    d = g + d
    return d


def compute_error(patch1, patch2):
    """Compute MSE between patch1 and patch2

        - Normalize patch1 and patch2 each to zero mean, unit variance
        - Compute mean square error between patch1 and patch2

    Args:
        patch1 - Grayscale image patch of shape (patch_size, patch_size)
        patch2 - Grayscale image patch of shape (patch_size, patch_size)
    Returns:
        error - Number representing mismatch between patch1 and patch2
    """
    assert patch1.shape == patch2.shape, "Different patch shapes"

    patch1_flat = patch1.flatten()
    patch1_flat = (patch1_flat - np.mean(patch1_flat)) / np.std(patch1_flat)
    patch2_flat = patch2.flatten()
    patch2_flat = (patch2_flat - np.mean(patch2_flat)) / np.std(patch2_flat)

    error = (np.square(patch1_flat - patch2_flat)).mean()
    return error


def track_features(
        frames,
        keypoints,
        error_thresh=1.5,
        optflow_fn=pyramid_lucas_kanade,
        exclude_border=5,
        **kwargs
):
    """Track key_points over multiple frames

    Args:
        frames - List of grayscale images with the same shape.
        key_points - Keypoints in frames[0] to start tracking. Numpy array of
            shape (N, 2).
        error_thresh - Threshold to determine lost tracks.
        optflow_fn(img1, img2, key_points, **kwargs) - Optical flow function.
        kwargs - keyword arguments for optflow_fn.

    Returns:
        trajs - A list containing tracked key_points in each frame. trajs[i]
            is a numpy array of key_points in frames[i]. The shape of trajs[i]
            is (Ni, 2), where Ni is number of tracked points in frames[i].
    """

    kp_curr = keypoints
    trajs = [kp_curr]
    patch_size = 3  # Take 3x3 patches to compute error
    w = patch_size // 2  # patch_size//2 around a pixel

    for i in range(len(frames) - 1):
        I = frames[i]
        J = frames[i + 1]
        flow_vectors = optflow_fn(I, J, kp_curr, **kwargs)
        kp_next = kp_curr + flow_vectors

        new_keypoints = []
        for yi, xi, yj, xj in np.hstack((kp_curr, kp_next)):
            # Declare a keypoint to be 'lost' IF:
            # 1. the keypoint falls outside the image J
            # 2. the error between points in I and J is larger than threshold

            yi = int(round(yi))
            xi = int(round(xi))
            yj = int(round(yj))
            xj = int(round(xj))
            # Point falls outside the image
            if (
                    yj > J.shape[0] - exclude_border - 1
                    or yj < exclude_border
                    or xj > J.shape[1] - exclude_border - 1
                    or xj < exclude_border
            ):
                continue

            # Compute error between patches in image I and J
            patchI = I[yi - w: yi + w + 1, xi - w: xi + w + 1]
            patchJ = J[yj - w: yj + w + 1, xj - w: xj + w + 1]
            error = compute_error(patchI, patchJ)
            if error > error_thresh:
                continue

            new_keypoints.append([yj, xj])

        kp_curr = np.array(new_keypoints)
        trajs.append(kp_curr)

    return trajs


def IoU(bbox1, bbox2):
    """Compute IoU of two bounding boxes

    Args:
        bbox1 - 4-tuple (x, y, w, h) where (x, y) is the top left corner of
            the bounding box, and (w, h) are width and height of the box.
        bbox2 - 4-tuple (x, y, w, h) where (x, y) is the top left corner of
            the bounding box, and (w, h) are width and height of the box.
    Returns:
        score - IoU score
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    xs = sorted([x1, x1 + w1, x2, x2 + w2])
    ys = sorted([y1, y1 + h1, y2, y2 + h2])
    if xs[1] == x2 or xs[1] == x1:
        x_lens = xs[2] - xs[1]
        y_lens = ys[2] - ys[1]
        intersection_area = x_lens * y_lens
        union_area = w1 * h1 + w2 * h2 - intersection_area
        score = intersection_area / union_area
    else:
        score = 0

    return score
