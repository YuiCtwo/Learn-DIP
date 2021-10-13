import cv2
import numpy as np
import matplotlib.pyplot as plt

from hough import hough_transform

# 调用库
img = cv2.imread('sudoku.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 灰度图像
edges = cv2.Canny(gray, 150, 200)
plt.subplot(121)
plt.imshow(edges, 'gray')
plt.xticks([])
plt.yticks([])
# hough transform
lines = cv2.HoughLines(edges, 1, np.pi / 180, 160)
lines1 = lines[:, 0, :]  # 提取为为二维
for rho, theta in lines1[:]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * a)
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * a)
    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 1)

plt.subplot(122)
plt.imshow(img)
plt.xticks([])
plt.yticks([])


# 自己的实现

hough_transform(edges)
plt.show()
