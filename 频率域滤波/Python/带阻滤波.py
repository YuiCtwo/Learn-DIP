from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from Filter import BBRFFilter, GBRFFilter, IBRFFilter


if __name__ == '__main__':
    # 仅供测试使用
    choice = "material/windmill_noise.png"
    img = Image.open(choice).convert("L")

    # sq = min(img.size[0], img.size[1])
    # img = img.resize((sq, sq))
    f = np.fft.fft2(img)
    f_shift = np.fft.fftshift(f)
    choices = {
        "巴特沃斯带阻": BBRFFilter.generate_filter(15, (img.size[1], img.size[0]), n=3, w=5),
        "高斯带阻": GBRFFilter.generate_filter(15, (img.size[1], img.size[0]), w=5),
        "理想带阻": IBRFFilter.generate_filter(15, (img.size[1], img.size[0]), w=5)
    }
    for k, v in choices.items():

        filter_matrix = v
        transform_img = np.abs(np.fft.ifft2(np.fft.ifftshift(f_shift*filter_matrix)))
        plt.rcParams['figure.figsize'] = (6.0, 3.0)
        plt.subplot(121)
        plt.imshow(transform_img, cmap="gray")
        plt.axis("off")
        plt.subplot(122)
        plt.imshow(img, cmap="gray")
        plt.axis("off")
        plt.savefig("{}.png".format(k))
        plt.clf()

