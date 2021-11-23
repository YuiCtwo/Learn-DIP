# Image Pyramid
# pyramid函数参考 CS131 hw6 中的习题
from skimage.io import imread
from skimage.transform import rescale

import cv2 as cv


def pyramid(image, scale=0.9, min_size=(200, 100)):
    """
    Generate image pyramid using the given image and scale.
    Reducing the size of the image until either the height or
    width is below the minimum limit. In the ith iteration,
    the image is resized to scale^i of the original image.

    This function is mostly completed for you -- only a termination
    condition is needed.

    Args:
        image: np array of (h,w), an image to scale.
        scale: float of how much to rescale the image each time.
        min_size: pair of ints showing the minimum height and width.

    Returns:
        images: list containing pair of
            (the current scale of the image, resized image).
    """
    images = []
    h, w, d = image.shape
    # Yield the original image
    current_scale = 1.0
    images.append((current_scale, image))

    while True:
        if h * current_scale <= min_size[0] or w * current_scale <= min_size[1]:
            break
        # Compute the new dimensions of the image and resize it
        current_scale *= scale
        image = rescale(image, scale, mode='constant')
        # Yield the next image in the pyramid
        images.append((current_scale, image))

    return images


if __name__ == '__main__':
    # 自己实现的方法
    img = imread("./face.jpg")
    pyramid(img, min_size=(100, 50))
    # OpenCV 提供的函数
    # cv2.pyrUp()    cv2.pyrDown()
    cv_img = cv.imread("./face.jpg")
    low1 = cv.pyrDown(cv_img)
    low2 = cv.pyrDown(low1)
    cv.imshow("Origin", cv_img)
    cv.imshow("Down Sampled Once", low1)
    cv.imshow("Down Sampled Twice", low2)
    cv.waitKey(0)
