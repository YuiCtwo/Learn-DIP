from skimage.feature import corner_harris, corner_peaks
from utils import load_frames
from skimage.feature import corner_peaks

from simple_lucas_kanade import lucas_kanade
from pyramidal_lucas_kanade import iterative_lucas_kanade, pyramid_lucas_kanade

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

frames = load_frames("./data/images")
key_points = corner_peaks(corner_harris(frames[0]), exclude_border=5, threshold_rel=0.01)


def check_lucas_kanade():
    # Detect key_points to track

    flow_vectors = lucas_kanade(frames[0], frames[1], key_points, window_size=5)
    plt.figure(figsize=(15, 12))
    plt.imshow(frames[0])
    plt.axis('off')
    plt.title('Optical flow vectors')
    for y, x, vx, vy in np.hstack((key_points, flow_vectors)):
        plt.arrow(x, y, vx, vy, head_width=5, head_length=5, color='b')
    plt.show()
    plt.clf()


def check_iterative_lucas_kandas():
    # Run iterative Lucas-Kanade method
    flow_vectors = iterative_lucas_kanade(frames[0], frames[1], key_points)

    # Plot flow vectors
    plt.figure(figsize=(15, 12))
    plt.imshow(frames[0])
    plt.axis('off')
    plt.title('Optical flow vectors (iterative LK)')

    for y, x, vx, vy in np.hstack((key_points, flow_vectors)):
        plt.arrow(x, y, vx, vy, head_width=5, head_length=5, color='b')
    plt.show()
    plt.clf()


def check_pyramid_lucas_kanade():
    # Lucas-Kanade method for optical flow
    flow_vectors = pyramid_lucas_kanade(frames[0], frames[1], key_points)

    # Plot flow vectors
    plt.figure(figsize=(15, 12))
    plt.imshow(frames[0])
    plt.axis('off')
    plt.title('Optical flow vectors (pyramid LK)')

    for y, x, vx, vy in np.hstack((key_points, flow_vectors)):
        plt.arrow(x, y, vx, vy, head_width=3, head_length=3, color='b')
    plt.show()
    plt.clf()


if __name__ == '__main__':
    # check_lucas_kanade()
    # check_iterative_lucas_kandas()
    check_pyramid_lucas_kanade()