import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from skimage import io
from skimage import img_as_ubyte, img_as_float

from canny_edges_detection import conv, gaussian_kernel, partial_x, partial_y, gradient
from canny_edges_detection import non_maximum_suppression, link_edges

from canny_edges_detection import canny

plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


# 验证函数参考 CS 131 hw2 模板
def check_gaussian_kernel():
    # 验证 gaussian_kernel
    # Define 3x3 Gaussian kernel with std = 1

    kernel = gaussian_kernel(3, 1)
    kernel_test = np.array(
        [[0.05854983, 0.09653235, 0.05854983],
         [0.09653235, 0.15915494, 0.09653235],
         [0.05854983, 0.09653235, 0.05854983]]
    )
    # Test Gaussian kernel
    if not np.allclose(kernel, kernel_test):
        print('Incorrect values! Please check your implementation.')


def check_conv(img):
    # 验证 conv
    # Test with different kernel_size and sigma
    kernel_size = 5
    sigma = 1.4
    # Define 5x5 Gaussian kernel with std = sigma
    kernel = gaussian_kernel(kernel_size, sigma)
    # Convolve image with kernel to achieve smoothed effect
    smoothed = conv(img, kernel)
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title('Original image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(smoothed)
    plt.title('Smoothed image')
    plt.axis('off')

    plt.show()


def check_partial():
    # Test input
    I = np.array(
        [[0, 0, 0],
         [0, 1, 0],
         [0, 0, 0]]
    )

    # Expected outputs
    I_x_test = np.array(
        [[0, 0, 0],
         [0.5, 0, -0.5],
         [0, 0, 0]]
    )

    I_y_test = np.array(
        [[0, 0.5, 0],
         [0, 0, 0],
         [0, -0.5, 0]]
    )

    # Compute partial derivatives
    I_x = partial_x(I)
    I_y = partial_y(I)

    # Test correctness of partial_x and partial_y
    if not np.all(I_x == I_x_test):
        print('partial_x incorrect')

    if not np.all(I_y == I_y_test):
        print('partial_y incorrect')


def check_gradient(img):
    kernel = gaussian_kernel(5, 1.4)
    smoothed = conv(img, kernel)
    G, theta = gradient(smoothed)

    if not np.all(G >= 0):
        print('Magnitude of gradients should be non-negative.')

    if not np.all((theta >= 0) * (theta < 360)):
        print('Direction of gradients should be in range 0 <= theta < 360')

    plt.imshow(G)
    plt.title('Gradient magnitude')
    plt.axis('off')
    plt.show()


def check_non_maximum_suppression():
    # Test input
    g = np.array(
        [[0.4, 0.5, 0.6],
         [0.3, 0.5, 0.7],
         [0.4, 0.5, 0.6]]
    )
    # Print out non-maximum suppressed output
    # varying theta
    for angle in range(0, 180, 45):
        # print('Thetas:', angle)
        t = np.ones((3, 3)) * angle  # Initialize theta
        print(non_maximum_suppression(g, t))


def check_link_edges():
    test_strong = np.array(
        [[1, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 1]],
        dtype=np.bool
    )

    test_weak = np.array(
        [[0, 0, 0, 1],
         [0, 1, 0, 0],
         [1, 0, 0, 0],
         [0, 0, 1, 0]],
        dtype=np.bool
    )

    test_linked = link_edges(test_strong, test_weak)

    plt.subplot(1, 3, 1)
    plt.imshow(test_strong)
    plt.title('Strong edges')

    plt.subplot(1, 3, 2)
    plt.imshow(test_weak)
    plt.title('Weak edges')

    plt.subplot(1, 3, 3)
    plt.imshow(test_linked)
    plt.title('Linked edges')
    plt.show()


if __name__ == '__main__':
    img = io.imread('iguana.png', as_gray=True)
    # check_gaussian_kernel()
    # check_conv(img)
    # check_partial()
    # check_gradient(img)
    # check_non_maximum_suppression()
    # check_link_edges()
    # Run Canny edge detector
    edges = canny(img, kernel_size=5, sigma=1.4, high=0.03, low=0.02)
    print(edges.shape)

    plt.subplot(1, 3, 1)
    plt.imshow(edges)
    plt.axis('off')
    plt.title('Your result')

    plt.subplot(1, 3, 2)
    cv_image = cv.imread('iguana.png', 0)
    opencv_res = cv.Canny(cv.GaussianBlur(cv_image, (5, 5), 1.4), 0.03, 0.02)
    # cv.imshow("canny", opencv_res)
    # cv.waitKey()

    # 转换可能带来一些误差
    opencv_res = img_as_float(opencv_res)
    plt.imshow(opencv_res)
    plt.title('OpenCV result')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    reference = np.load('iguana_canny.npy')
    plt.imshow(reference)
    plt.title('Refer result')
    plt.axis('off')

    plt.show()
