# Data and code based on https://github.com/mint-lab/3dv_tutorial

import cv2
import numpy as np
import copy
import random

# ChessBoard Setting
board_pattern = (10, 7)

# Open a Video
input = "./chessboard.avi"
cap = cv2.VideoCapture(input)
windows_name = "test"
# select some image for equation solver
num_samples = 5
# save image per i second
fps = cap.get(cv2.CAP_PROP_FPS)
seconds = 5
interval = int(fps * seconds)
# Get some Chessboard images
images = []
displays = []
while True:
    frame_id = int(round(cap.get(1)))
    ret, image = cap.read()
    if not ret:
        break
    cv2.waitKey(1)
    # cv2.imshow(windows_name, image)
    ret, pts = cv2.findChessboardCorners(image, board_pattern, None)  # No flags
    display = copy.deepcopy(image)
    display = cv2.drawChessboardCorners(display, board_pattern, pts, ret)
    cv2.imshow(windows_name, display)
    if frame_id % interval == 0:
        images.append(image)

cap.release()
cv2.destroyAllWindows()
assert len(images) != 0 and num_samples <= len(images)
print("Total {} images.".format(len(images)))
img_points = []
images_sampled = random.sample(images, num_samples)
for image in images_sampled:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    ret, corners = cv2.findChessboardCorners(gray, board_pattern)  # No flags
    if ret:
        img_points.append(corners)
objp = np.zeros((board_pattern[0]*board_pattern[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:board_pattern[1], 0:board_pattern[0]].T.reshape(-1, 2)
obj_points = [objp for _ in range(len(images_sampled))]
# Calibrate Camera
rms, K, dist_coeff, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (h, w), None, None)

# Report calibration results
print("## Camera Calibration Results")
print(f"* The number of applied images = {w}x{h}")
print(f"* RMS error = {rms}")
print(f"* Camera matrix (K) = \n{K}")
print(f"* Distortion coefficient (k1, k2, p1, p2, k3, ...)\n = {dist_coeff}")