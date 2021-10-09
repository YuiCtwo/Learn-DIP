import numpy as np


def hough_transform(img):
    """ Transform points in the input image into Hough space.

    Use the parameterization:
        rho = x * cos(theta) + y * sin(theta)
    to transform a point (x,y) to a sine-like function in Hough space.

    Args:
        img: binary image of shape (H, W).

    Returns:
        accumulator: numpy array of shape (m, n).
        rhos: numpy array of shape (m, ).
        thetas: numpy array of shape (n, ).
    """
    # Set rho and theta ranges
    W, H = img.shape
    diag_len = int(np.ceil(np.sqrt(W * W + H * H)))
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2 + 1)
    thetas = np.deg2rad(np.arange(-90.0, 90.0))

    # Cache some reusable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)
    num_rhos = len(rhos)
    # Initialize accumulator in the Hough space
    accumulator = np.zeros((2 * diag_len + 1, num_thetas), dtype=np.uint64)
    ys, xs = np.nonzero(img)

    # Transform each point (x, y) in image
    # Find rho corresponding to values in thetas
    # and increment the accumulator in the corresponding coordiates.
    for x in xs:
        for y in ys:
            for i in range(num_thetas):
                rho = x * cos_t[i] + y * sin_t[i]
                # 四舍五入已经是求近似的值了
                rho_int = int(rho)
                idx = rho_int - (-diag_len)
                # 确保不小于 0
                idx = max(idx, 0)
                # 确保不越界
                idx = min(idx, num_rhos-1)
                accumulator[idx, i] += 1

    return accumulator, rhos, thetas
