from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np


class SLIC:

    def __init__(self,
                 img: np.array,
                 stride: int,
                 kernel_size: int = 3,
                 c: int = 40,
                 max_iterations: int = 10):

        self.img = img
        self.img_height = img.shape[0]
        self.img_width = img.shape[1]
        self.img_shape = img.shape[:2]
        self.stride = stride
        self.ks = kernel_size
        self.dcm = c
        self.dsm = stride
        self.max_iterations = max_iterations
        self.cluster_centers = None
        self.distances = None
        self.labels = None
        self.label_cache = None

    def initialize(self):
        center = []
        for y in range(self.stride, self.img_height - self.stride // 2, self.stride):
            for x in range(self.stride, self.img_width - self.stride // 2, self.stride):
                # 选择邻域的像素点: c1->(y, x), c2->(y, x+1), c3->(y+1, x)
                c1 = self.img[y - 1:y + self.ks - 1, x - 1:x + self.ks - 1]
                c2 = self.img[y - 1:y + self.ks - 1, x:x + self.ks]
                c3 = self.img[y:y + self.ks, x - 1:x + self.ks - 1]
                # 计算梯度
                grad = np.sum(np.abs(c1 - c3) + np.abs(c2 - c3), axis=2)
                yx_center = np.unravel_index(np.argmin(grad), grad.shape)
                center.append([
                    self.img[yx_center][0],  # r
                    self.img[yx_center][1],  # g
                    self.img[yx_center][2],  # b
                    yx_center[1] + x - 1,  # x
                    yx_center[0] + y - 1  # y
                ])
        # self.cluster_centers = np.array(center)
        self.cluster_centers = center
        self.distances = 1000000 * np.ones(self.img_shape)
        self.labels = np.zeros((len(center), self.img_height, self.img_width))
        self.label_cache = -1 * np.ones(self.img_shape, dtype=np.int16)

    def compute_distance(self, m1, m2):
        dist = (m1 - m2) ** 2
        dc = np.sqrt(np.sum(dist[:3]))  # RGB 距离
        ds = np.sqrt(np.sum(dist[3:]))  # 空间距离
        D = ((dc / self.dcm) ** 2 + (ds / self.dsm) ** 2) ** 0.5
        return D

    def assign_samples(self):
        for idx, cluster_center in enumerate(self.cluster_centers):
            center_y = cluster_center[4]
            center_x = cluster_center[3]
            # print("Center:({},{})".format(center_y, center_x))
            for y in range(center_y - 2 * self.stride, center_y + 2 * self.stride):
                if y < 0 or y >= self.img_height:
                    continue
                for x in range(center_x - 2 * self.stride, center_x + 2 * self.stride):
                    if x < 0 or x >= self.img_width:
                        continue
                    # 中心点 2sx2s大小的邻域
                    p = np.array(
                        [self.img[y, x][0], self.img[y, x][1], self.img[y, x][2], x, y]
                    )
                    Dp = self.compute_distance(np.array(cluster_center), p)
                    if Dp < self.distances[y][x]:
                        self.distances[y][x] = Dp
                        old_label = self.label_cache[y][x]
                        if old_label != -1:
                            self.labels[old_label][y][x] = 0
                        self.labels[idx][y][x] = 1
                        self.label_cache[y][x] = idx

    def update_cluster(self):
        ys = np.arange(0, self.img_height)
        ys = np.expand_dims(ys, axis=0).repeat(self.img_width, axis=0).T  # hxw
        xs = np.arange(0, self.img_width)
        xs = np.expand_dims(xs, axis=0).repeat(self.img_height, axis=0)
        # yxs = np.stack([ys, xs], axis=0)
        for idx in range(len(self.cluster_centers)):
            label = self.labels[idx]
            num_c = np.sum(label)
            _y = int(np.sum(label * ys) / num_c)
            _x = int(np.sum(label * xs) / num_c)
            self.cluster_centers[idx] = [
                self.img[_y, _x][0], self.img[_y, _x][1], self.img[_y, _x][2], _x, _y
            ]

    def post_processing(self):
        new_img = np.zeros((3, self.img_height, self.img_width))
        for idx, cluster_center in enumerate(self.cluster_centers):
            label = self.labels[idx, :, :]
            color = np.array(cluster_center[:3])
            color = np.expand_dims(color, axis=(1, 2)).repeat(self.img_height, axis=1).repeat(self.img_width, axis=2)
            new_img += (color * label)
        return new_img.transpose((1, 2, 0))

    def process(self, show_every_loop=False):
        self.initialize()
        print("Super-pixels num:{}".format(len(self.cluster_centers)))
        for it in range(self.max_iterations):
            print("Iteration {} start".format(it + 1))
            self.assign_samples()
            self.update_cluster()
            print("Iteration {} end".format(it + 1))
            if show_every_loop:
                new_img = self.post_processing()
                plt.figure(it)
                plt.imshow(new_img)
                plt.show()
        if not show_every_loop:
            new_img = self.post_processing()
            plt.imshow(new_img)
            plt.show()


if __name__ == '__main__':
    rgb = imread("./Lenna.png", "RGB")
    slic = SLIC(rgb, 25, max_iterations=1)
    # 计算时间相当的长......
    slic.process()
