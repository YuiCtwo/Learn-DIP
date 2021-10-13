import numpy as np
import math
import cv2 as cv
import matplotlib.pyplot as plt

from DeBlurFilter import inverse, wiener


def motion_process(image_size, motion_angle):
    # 生成运动模糊的滤波器
    PSF = np.zeros(image_size)
    center_position_h = (image_size[0] - 1) / 2
    center_position_w = (image_size[1] - 1) / 2
    slope_tan = math.tan(motion_angle * math.pi / 180)
    slope_cot = 1 / slope_tan
    if slope_tan <= 1:
        for i in range(15):
            offset = round(i * slope_tan)  # ((center_position-i)*slope_tan)
            PSF[int(center_position_h + offset), int(center_position_w - offset)] = 1
    else:
        for i in range(15):
            offset = round(i * slope_cot)
            PSF[int(center_position_h - offset), int(center_position_w + offset)] = 1
    PSF / PSF.sum()  # 对点扩散函数进行归一化亮度
    return PSF


def motion_blurred(input_img, PSF):
    input_fft = np.fft.fft2(input_img)
    PSF_fft = np.fft.fft2(PSF) + 1e-3
    # H(u, v) = F(u, v) * G(u, v)
    # 然后逆傅里叶变换
    result = np.fft.ifft2(input_fft * PSF_fft)
    # 中心平移, 频率域滤波都必须要有的步骤
    result = np.abs(np.fft.fftshift(result))
    return result


def normal(arr):
    # cv.normalize(arr, 0, 255, cv.NORM_MINMAX)
    arr = np.where(arr < 0,  0, arr)
    arr = np.where(arr > 255, 255, arr)
    array = arr.astype(np.uint8)
    return array


if __name__ == '__main__':
    img = cv.imread("tower.jpg")
    inverse_result = []
    wiener_result = []
    blurred_result = []
    b, g, r = cv.split(img)
    for one in [r, g, b]:
        img_h, img_w = one.shape[:2]
        PSF = motion_process((img_h, img_w), 10)
        blurred_img = motion_blurred(one, PSF)
        blurred_result.append(normal(blurred_img))
        # 记得调整到 0-255
        recover_inverse = normal(inverse(blurred_img, PSF))
        recover_wiener = normal(wiener(blurred_img, PSF))
        inverse_result.append(recover_inverse)
        wiener_result.append(recover_wiener)

    inverse_recover_result = cv.merge(inverse_result)
    wiener_recover_result = cv.merge(wiener_result)
    motion_blurred_result = cv.merge(blurred_result)
    # ========= 可视化 ==========
    plt.figure(1)
    plt.subplot(121)
    plt.xlabel("Original Image")
    plt.imshow(np.flip(img, axis=2))  # 用 flip 函数调整 bgr 到 rgb
    plt.xticks([])
    plt.yticks([])
    plt.subplot(122)
    plt.xlabel("Blurred Image")
    plt.imshow(motion_blurred_result)
    plt.xticks([])
    plt.yticks([])
    # 两种结果的显示
    plt.figure(2)
    plt.imshow(inverse_recover_result)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("Inverse Recovery Result")
    plt.figure(3)
    plt.imshow(wiener_recover_result)
    plt.xlabel("Wiener Recovery Result")
    plt.xticks([])
    plt.yticks([])
    plt.show()
