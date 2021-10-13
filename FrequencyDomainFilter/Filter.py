from typing import List, Tuple

import numpy as np


def cal_distance(pa, pb):
    from math import sqrt
    dis = sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
    return dis


# 滤波器模板
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


# 巴特沃斯带阻滤波
class BBRFFilter(Filter):

    @classmethod
    def get_one(cls, d, dist, *args, **kwargs) -> float:
        w = kwargs["w"]
        n = kwargs["n"]
        if dist == d:
            return 0
        else:
            return 1 / (1 + ((dist * w) / (dist ** 2 - d ** 2)) ** (2 * n))


# 高斯带阻滤波
class GBRFFilter(Filter):

    @classmethod
    def get_one(cls, d, dist, *args, **kwargs) -> float:
        w = kwargs["w"]
        return 1-np.exp(-((dist ** 2 - d ** 2) / (d * w)) ** 2)


# 理想带阻滤波器
class IBRFFilter(Filter):

    @classmethod
    def get_one(cls, d, dist, *args, **kwargs) -> float:
        w = kwargs["w"]
        if d - w / 2 <= dist <= d + w / 2:
            return 0
        else:
            return 1


# 理想低通滤波器
class ILPFFilter(Filter):

    @classmethod
    def get_one(cls, d, dist, *args, **kwargs) -> float:
        if dist <= d:
            return 1
        else:
            return 0


# 高斯低通滤波器
class GLPFFilter(Filter):

    @classmethod
    def get_one(cls, d, dist, *args, **kwargs) -> float:
        return np.exp(-(dist ** 2) / (2 * (d ** 2)))


# 巴特沃斯低通滤波器
class BLPFFilter(Filter):

    @classmethod
    def get_one(cls, d, dist, *args, **kwargs) -> float:
        n = kwargs["n"]
        return 1 / ((1 + dist / d) ** (2 * n))

# 高通滤波相当于低通反过来, 有 1减或者分之一等的操作
# 具体可以参考书上的


# 陷波滤波器
class BTWNotchFilter:

    @classmethod
    def generate_filter(cls, shape, n, d0: List[float], center_points: List[Tuple[int, int]]):
        num_notch = len(center_points)
        assert len(d0) == num_notch
        mid_M = shape[0] / 2
        mid_N = shape[1] / 2
        transfer_matrix = np.zeros((shape[0], shape[1]))
        for i in range(transfer_matrix.shape[0]):
            for j in range(transfer_matrix.shape[1]):
                res = 1
                for k in range(num_notch):
                    temp_p = (mid_M+center_points[k][0], mid_N+center_points[k][1])
                    Dk = cal_distance(temp_p, (i, j))
                    temp_p = (mid_M-center_points[k][0], mid_N-center_points[k][1])
                    _Dk = cal_distance(temp_p, (i, j))
                    res *= (1 / ((1 + d0[k]/Dk) ** n))
                    res *= (1 / ((1 + d0[k]/_Dk) ** n))
                transfer_matrix[i, j] = res
        return transfer_matrix

