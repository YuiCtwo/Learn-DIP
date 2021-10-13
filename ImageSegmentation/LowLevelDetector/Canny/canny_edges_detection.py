"""
这个文件提供一步一步实现 Canny 边缘检测的代码（仅供学习使用）
使用 CS 131 hw2 的模板
在 demo 中会给出调用库实现的方法
"""

import numpy as np


def conv(image, kernel):
    """ An implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    # For this assignment, we will use edge values to pad the images.
    # Zero padding will make derivatives at the image boundary very big,
    # whereas we want to ignore the edges at the boundary.
    pad_width0 = Hk // 2
    pad_width1 = Wk // 2
    pad_width = ((pad_width0, pad_width0), (pad_width1, pad_width1))
    padded = np.pad(image, pad_width, mode='edge')

    for hi in range(Hi):
        for wi in range(Wi):
            # 计算的位置是 (hi, wi)
            out[hi, wi] = np.sum(padded[hi: hi + Hk, wi: wi + Wk] * kernel)
    return out


def gaussian_kernel(size, sigma):
    """ Implementation of Gaussian Kernel.

    This function follows the gaussian kernel formula,
    and creates a kernel matrix.

    Hints:
    - Use np.pi and np.exp to compute pi and exp.

    Args:
        size: int of the size of output matrix.
        sigma: float of sigma to calculate kernel.

    Returns:
        kernel: numpy array of shape (size, size).
    """
    # size = 2k+1
    assert (size - 1) % 2 == 0
    k = (size - 1) // 2
    kernel = np.zeros((size, size))

    for h in range(size):
        for w in range(size):
            kernel[h, w] = 1 / (2 * np.pi * (sigma ** 2)) \
                           * np.exp(-((h - k) ** 2 + (w - k) ** 2) / (2 * sigma * sigma))

    return kernel


def partial_x(img):
    """ Computes partial x-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: x-derivative image.
    """
    x_kernel = np.array([-1 / 2, 0, 1 / 2])
    x_kernel.resize((1, 3))  # 1行 3列
    return conv(img, x_kernel)


def partial_y(img):
    """ Computes partial y-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: y-derivative image.
    """

    y_kernel = [
        [-1 / 2],
        [0],
        [1 / 2]
    ]
    y_kernel = np.array(y_kernel)
    y_kernel.resize((3, 1))  # 3行 1列
    return conv(img, y_kernel)


def gradient(img):
    """ Returns gradient magnitude and direction of input img.

    Args:
        img: Grayscale image. Numpy array of shape (H, W).

    Returns:
        G: Magnitude of gradient at each pixel in img.
            Numpy array of shape (H, W).
        theta: Direction(in degrees, 0 <= theta < 360) of gradient
            at each pixel in img. Numpy array of shape (H, W).

    Hints:
        - Use np.sqrt and np.arctan2 to calculate square root and arctan
    """
    gx = partial_x(img)
    gy = partial_y(img)
    G = np.sqrt(gx * gx + gy * gy)
    theta = np.arctan2(gx, gy)

    return G, theta


def non_maximum_suppression(G, theta):
    """ Performs non-maximum suppression.

    This function performs non-maximum suppression along the direction
    of gradient (theta) on the gradient magnitude image (G).

    Args:
        G: gradient magnitude image with shape of (H, W).
        theta: direction of gradients with shape of (H, W).

    Returns:
        out: non-maxima suppressed image.
    """
    H, W = G.shape
    out = np.zeros((H, W))

    # Round the gradient direction to the nearest 45 degrees
    theta = np.floor((theta + 22.5) / 45) * 45
    # theta = {0, 45, 90, 135}
    # print(G)
    G_pad = np.pad(G, ((1, 1), (1, 1)))
    for h in range(1, H+1):
        for w in range(1, W+1):
            # comp [h, w]
            switch = theta[h-1, w-1]
            if switch == 0:
                comp = [0, 1]
            elif switch == 45:
                comp = [1, 1]
            elif switch == 90:
                comp = [1, 0]
            elif switch == 135:
                comp = [1, -1]
            else:
                print(switch)
                comp = [0, 0]
                exit(1)
            if G_pad[h, w] <= G_pad[h+comp[0], w+comp[1]] or G_pad[h, w] <= G_pad[h-comp[0], w-comp[1]]:
                # 非最大值, 抑制
                out[h-1, w-1] = 0
            else:
                out[h-1, w-1] = G_pad[h, w]

    return out


def double_thresholding(img, high, low):
    """
    Args:
        img: numpy array of shape (H, W) representing NMS edge response.
        high: high threshold(float) for strong edges.
        low: low threshold(float) for weak edges.

    Returns:
        strong_edges: Boolean array representing strong edges.
            Strong edges are the pixels with the values greater than
            the higher threshold.
        weak_edges: Boolean array representing weak edges.
            Weak edges are the pixels with the values smaller or equal to the
            higher threshold and greater than the lower threshold.
    """
    assert high > low
    # strong_edges = img[img > high]
    strong_edges = np.where((img > high), 1, 0)
    # weak_edges = img[(low < img) & (img <= high)]
    weak_edges = np.where(((low < img) & (img <= high)), 1, 0)
    strong_edges.astype(np.bool)
    weak_edges.astype(np.bool)
    return strong_edges, weak_edges


def get_neighbors(y, x, H, W):
    """ Return indices of valid neighbors of (y, x).

    Return indices of all the valid neighbors of (y, x) in an array of
    shape (H, W). An index (i, j) of a valid neighbor should satisfy
    the following:
        1. i >= 0 and i < H
        2. j >= 0 and j < W
        3. (i, j) != (y, x)

    Args:
        y, x: location of the pixel.
        H, W: size of the image.
    Returns:
        neighbors: list of indices of neighboring pixels [(i, j)].
    """
    neighbors = []

    for i in (y - 1, y, y + 1):
        for j in (x - 1, x, x + 1):
            if 0 <= i < H and 0 <= j < W:
                if i == y and j == x:
                    continue
                neighbors.append((i, j))

    return neighbors


def link_edges(strong_edges, weak_edges):
    """ Find weak edges connected to strong edges and link them.

    Iterate over each pixel in strong_edges and perform breadth first
    search across the connected pixels in weak_edges to link them.
    Here we consider a pixel (a, b) is connected to a pixel (c, d)
    if (a, b) is one of the eight neighboring pixels of (c, d).

    Args:
        strong_edges: binary image of shape (H, W).
        weak_edges: binary image of shape (H, W).

    Returns:
        edges: numpy boolean array of shape(H, W).
    """

    H, W = strong_edges.shape
    indices = list(np.stack(np.nonzero(strong_edges)).T)
    t = 0
    visit_weak_edges = np.copy(weak_edges)
    edges = np.copy(strong_edges)
    while len(indices) != 0:
        t += 1
        p = indices.pop()
        ph = p[0]
        pw = p[1]
        neighbors = get_neighbors(ph, pw, H, W)
        for pp in neighbors:
            if visit_weak_edges[pp[0], pp[1]] == 1:
                # 强边界和弱边界相连，该弱边界也为最终的确定边界
                visit_weak_edges[pp[0], pp[1]] = 0
                edges[pp[0], pp[1]] = 1
                indices.append([pp[0], pp[1]])
        if t % 1000 == 0:
            print("Edges Link: Turn-{}".format(t))

    return edges


def canny(img, kernel_size=5, sigma=1.4, high=20, low=15):
    """ Implement canny edge detector by calling functions above.

    Args:
        img: binary image of shape (H, W).
        kernel_size: int of size for kernel matrix.
        sigma: float for calculating kernel.
        high: high threshold for strong edges.
        low: low threashold for weak edges.
    Returns:
        edge: numpy array of shape(H, W).
    """
    kernel = gaussian_kernel(kernel_size, sigma)
    smoothed_img = conv(img, kernel)
    G, theta = gradient(smoothed_img)
    G = non_maximum_suppression(G, theta)
    strong_edges, weak_edges = double_thresholding(G, high, low)
    edges = link_edges(strong_edges, weak_edges)

    # 最大最小值归一化
    # Min = np.min(edges)
    # Max = np.max(edges)
    # edges = (edges - Min) / (Max - Min)
    # print(edges.dtype)
    # edges = edges.astype(np.uint8)
    # print(edges.dtype)
    # 二值化
    edges[edges > 0] = 255
    return edges
