from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def cal_distance(pa, pb):
    from math import sqrt
    dis = sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
    return dis


class Filter:

    @classmethod
    def generate_filter(cls, d, shape, *args, **kwargs):
        # 这里要反过来 shape 的两个维度
        transfer_matrix = np.zeros((shape[0], shape[1]))
        center_point = tuple(map(lambda x: (x - 1) // 2, shape))
        for i in range(transfer_matrix.shape[0]):
            for j in range(transfer_matrix.shape[1]):
                dist = cal_distance(center_point, (i, j))
                transfer_matrix[i, j] = cls.get_one(d, dist, *args, **kwargs)
        return transfer_matrix

    @classmethod
    def get_one(cls, d, dist, *args, **kwargs) -> float:
        return 1


class BBRFFilter(Filter):

    @classmethod
    def get_one(cls, d, dist, *args, **kwargs) -> float:
        w = kwargs["w"]
        n = kwargs["n"]
        if dist == d:
            return 0
        else:
            return 1 / (1 + ((dist * w) / (dist ** 2 - d ** 2)) ** (2 * n))


class GBRFFilter(Filter):

    @classmethod
    def get_one(cls, d, dist, *args, **kwargs) -> float:
        w = kwargs["w"]
        return 1-np.exp(-((dist ** 2 - d ** 2) / (d * w)) ** 2)


class IBRFFilter(Filter):

    @classmethod
    def get_one(cls, d, dist, *args, **kwargs) -> float:
        w = kwargs["w"]
        if d - w / 2 <= dist <= d + w / 2:
            return 0
        else:
            return 1


def abs_threshold_filter(matrix):
    matrix = np.abs(matrix)
    return matrix


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

