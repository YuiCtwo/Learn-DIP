import numpy as np
from skimage.feature import corner_peaks
from skimage.io import imread
import matplotlib.pyplot as plt
from utils import plot_matches

from hog import hog_descriptor
from ransac import ransac
from harris import harris_corners, describe_keypoints, match_descriptors

plt.rcParams['figure.figsize'] = (15.0, 12.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

img1 = imread('uttower1.jpg', as_gray=True)
img2 = imread('uttower2.jpg', as_gray=True)

# Detect key_points in both images
hog_keypoints1 = corner_peaks(harris_corners(img1, window_size=3),
                              threshold_rel=0.05,
                              exclude_border=8)
hog_keypoints2 = corner_peaks(harris_corners(img2, window_size=3),
                              threshold_rel=0.05,
                              exclude_border=8)
print("HoG keypoints1 shape = ", hog_keypoints1.shape)
print("HoG keypoints2 shape = ", hog_keypoints2.shape)
# Extract features from the corners
hog_desc1 = describe_keypoints(img1, hog_keypoints1,
                               desc_func=hog_descriptor,
                               patch_size=16)
hog_desc2 = describe_keypoints(img2, hog_keypoints2,
                               desc_func=hog_descriptor,
                               patch_size=16)

print("HoG desc1 shape = ", hog_desc1.shape)
print("HoG desc2 shape = ", hog_desc2.shape)
hog_matches = match_descriptors(hog_desc1, hog_desc2, 0.7)
# Set seed to compare output against solution
np.random.seed(131)
H, robust_matches = ransac(hog_keypoints1, hog_keypoints2, hog_matches, threshold=1)
print("Robust matches shape = ", robust_matches.shape)
print("H = \n", H)

# Plot matches
fig, ax = plt.subplots(1, 1, figsize=(15, 12))
plot_matches(ax, img1, img2, hog_keypoints1, hog_keypoints2, robust_matches)
plt.axis('off')
plt.title('Robust Matched HOG descriptor + RANSAC')
plt.show()
#
# plt.imshow(imread('solution_hog_ransac.png'))
# plt.axis('off')
# plt.title('Robust Matched HOG descriptor + RANSAC Solution')
# plt.show()
#
# plt.imshow(imread('solution_hog.png'))
# plt.axis('off')
# plt.title('Matched HOG Descriptor Solution')
# plt.show()
