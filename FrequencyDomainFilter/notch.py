from Filter import BTWNotchFilter
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    choice = "windmill_noise.png"
    img = Image.open(choice).convert("L")

    # sq = min(img.size[0], img.size[1])
    # img = img.resize((sq, sq))
    f = np.fft.fft2(img)
    f_shift = np.fft.fftshift(f)
    # TODO:这里参数设计讲究一定的学问
    filter_ma = BTWNotchFilter.generate_filter((img.size[1], img.size[0]), 15, [9.0], [(img.size[1]+3, img.size[0]+3)])
    transform_img = np.abs(np.fft.ifft2(np.fft.ifftshift(f_shift * filter_ma)))
    plt.rcParams['figure.figsize'] = (6.0, 3.0)
    plt.imshow(transform_img, cmap="gray")
    plt.axis("off")
    plt.savefig("陷波滤波器.png")
