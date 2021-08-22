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


# 该函数在 matlab 中可以直接使用等于将中心元素移动到左上角再做 fft2
# [ 0 -1 0       [ 4 -1 -1
#  -1 4 -1    ->  -1  0  0  -> fft2(x)
#   0 -1 0]       -1  0  0 ]
def psf2otf(psf, outSize):
    psfSize = np.array(psf.shape)
    outSize = np.array(outSize)
    padSize = outSize - psfSize
    psf = np.pad(psf, ((0, padSize[0]), (0, padSize[1])), 'constant')
    for i in range(len(psfSize)):
        psf = np.roll(psf, -int(psfSize[i] / 2), i)
    otf = np.fft.fftn(psf)
    nElem = np.prod(psfSize)
    nOps = 0
    for k in range(len(psfSize)):
        nffts = nElem / psfSize[k]
        nOps = nOps + psfSize[k] * np.log2(psfSize[k]) * nffts
    if np.max(np.abs(np.imag(otf))) / np.max(np.abs(otf)) <= nOps * np.finfo(np.float32).eps:
        otf = np.real(otf)
    return otf


# 约束最小二乘方滤波 Constrained Least Squares Filtering
# 博客讲解:
def constrained_least_sq(inp, PSF, gamma=0.05):
    kernel = np.array([[0, -1, 0],
                       [-1, 4, -1],
                       [0, -1, 0]])
    PSF_kernel = np.fft.fft2(kernel)
    inp_fft = np.fft.fft2(inp)
    PF = np.fft.fft2(PSF)
    numerator = np.conj(PF)
    denominator = PF ** 2 + gamma * (PSF_kernel ** 2)
    result = np.fft.ifft2(numerator * inp_fft / denominator)
    result = np.abs(np.fft.fftshift(result))
    return result


