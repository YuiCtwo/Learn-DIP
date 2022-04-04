import cv2 as cv
from metric_rectification import get_affine_homography_matrix
import numpy as np

# pre-select point that can determine parallel line in the real world
parallel_points = [
    [164, 273],
    [255, 205],
    [256, 357],
    [349, 270],
    [165, 273],
    [90, 209],
    [257, 204],
    [180, 152]
]

# pre-select point that can determine perpendicular line in the real world
perpendicular_points = [
    [161, 128],
    [93, 154],
    [131, 178],
    [125, 104],
    [92, 83],
    [25, 108]
]

# two-step approach
img = cv.imread('floor.jpg')
gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
pa_l1 = parallel_points[:4]
pa_l2 = parallel_points[4:]
pe_l1 = perpendicular_points[:3]
pe_l2 = perpendicular_points[3:]
size = gray_img.shape
size = (size[1], size[0])
# HA removes projective distortion, makes the sides parallel which are parallel in the original image
HA = get_affine_homography_matrix(pa_l1, pa_l2, "8p")
# HM removes affine distortion, makes the adjacent sides orthogonal.
HM = get_affine_homography_matrix(pe_l1, pe_l2, "6p")
HA = np.float32(HA)
print(HM)
HM_inv = np.float32(np.linalg.inv(HM))
# 2-step method for metric rectification
affine_rect = cv.warpPerspective(img, HA, size)
metric_rect = cv.warpPerspective(affine_rect, HM_inv, size)

cv.imshow("Metric Rectification", metric_rect)
cv.imwrite("out.jpg", metric_rect)
cv.waitKey(1000)
cv.destroyAllWindows()

