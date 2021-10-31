from skimage.util import img_as_float
from matplotlib import rc
from skimage import io
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import numpy as np


# Pixel-Level Features
def color_features(img):
    """ Represents a pixel by its color.

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C)
    """
    H, W, C = img.shape
    img = img_as_float(img)
    features = np.zeros((H * W, C))

    for h in range(H):
        for w in range(W):
            features[h * W + w, :] = img[h, w]

    return features


def color_position_features(img):
    """ Represents a pixel by its color and position.

    Combine pixel's RGB value and xy coordinates into a feature vector.
    i.e. for a pixel of color (r, g, b) located at position (x, y) in the
    image. its feature vector would be (r, g, b, x, y).

    Don't forget to normalize features.

    Hints
    - You may find np.mgrid and np.dstack useful
    - You may use np.mean and np.std

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C+2)
    """
    H, W, C = img.shape
    color = img_as_float(img)
    grid = np.mgrid[0:H:1, 0:W:1]
    # 使用 transpose 变换维度
    grid = grid.transpose(1, 2, 0)
    features = np.dstack((color, grid))
    features = features.reshape((H*W, C+2))
    # features = (features - np.mean(features)) / np.std(features)
    return features


if __name__ == '__main__':
    # Load and display image
    img = io.imread('train.jpg')
    H, W, C = img.shape
    plt.rcParams['figure.figsize'] = (15.0, 12.0)  # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'
    np.random.seed(0)

    features = color_features(img)
    estimator = KMeans(n_clusters=8)
    estimator.fit(features)
    assignments = estimator.labels_
    segments = assignments.reshape((H, W))

    # Display segmentation
    plt.subplot(121)
    plt.imshow(segments, cmap='viridis')
    plt.axis('off')
    plt.subplot(122)
    advanced_features = color_position_features(img)
    estimator = KMeans(n_clusters=8)
    estimator.fit(advanced_features)
    assignments = estimator.labels_
    segments = assignments.reshape((H, W))
    plt.imshow(segments, cmap='viridis')
    plt.axis('off')
    plt.show()
