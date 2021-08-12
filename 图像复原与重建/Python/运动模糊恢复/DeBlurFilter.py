import numpy as np
import cv2 as cv


# 逆滤波
# 仅考虑除以退化函数
# F^ = F + N / H, F 未知 采用 epsilon 近似
# 对噪声很敏感
def inverse(inp, PSF):
    input_fft = np.fft.fft2(inp)
    PSF_fft = np.fft.fft2(PSF) + 1e-3
    result = np.fft.ifft2(input_fft / PSF_fft)
    result = np.abs(np.fft.fftshift(result))
    return result


# 改进逆滤波, 考虑让估计函数与原函数的均方误差尽可能的小
# 一般计算的时候, 我们会将其中的噪声功率谱与未退化图像的功率谱相除近似为一个常量
def wiener(inp, PSF, K=0.01):
    input_fft = np.fft.fft2(inp)
    PSF_fft = np.fft.fft2(PSF) + 1e-3
    PSF_fft_1 = np.conj(PSF_fft) / (np.abs(PSF_fft) ** 2 + K)
    result = np.fft.ifft2(input_fft * PSF_fft_1)
    result = np.abs(np.fft.fftshift(result))
    return result


# 约束最小二乘方滤波 Constrained Least Squares Filtering
def constrained_least_sq():
    pass
