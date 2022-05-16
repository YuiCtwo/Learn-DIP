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
        self.img_shape = img.shape
        self.img_height = img.shape[0]
        self.img_width = img.shape[1]
        self.stride = stride
        self.ks = kernel_size
        self.dcm = c
        self.dsm = stride
        self.max_iterations = max_iterations
        self.cluster_centers = None
        self.labels = None
        self.distances = None

    def initialize(self):
        center = []
        for y in range(self.stride, self.img_height - self.stride // 2, self.stride):
            for x in range(self.stride, self.img_width - self.stride // 2, self.stride):
                # 选择邻域的像素点: c1->(y, x), c2->(y, x+1), c3->(y+1, x)
                c1 = self.img[y - 1:y + self.ks - 1, x - 1:x + self.ks - 1]
                c2 = self.img[y - 1:y + self.ks - 1, x:x + self.ks]
                c3 = self.img[y:y + self.ks, x - 1:x + self.ks - 1]
                # 计算梯度
                grad = np.abs(c1 - c3) + np.abs(c2 - c3)
                yx_center = np.argmin(grad)
                center.append([
                    self.img[yx_center][0],  # r
                    self.img[yx_center][1],  # g
                    self.img[yx_center][2],  # b
                    yx_center[1],  # x
                    yx_center[0]  # y
                ])
        # self.cluster_centers = np.array(center)
        self.cluster_centers = center
        self.labels = -1 * np.ones(self.img_shape)
        self.distances = 1000000 * np.ones(self.img_shape)

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
            for y in range(center_y - 2 * self.stride, center_y + 2 * self.stride):
                if y < 0 or y >= self.img_height:
                    continue
                for x in range(center_x - 2 * self.stride, center_x + 2 * self.stride):
                    if x < 0 or x >= self.img_width:
                        # 中心点 2sx2s大小的邻域
                        p = np.array(
                            self.img[y, x][0], self.img[y, x][1], self.img[y, x][2], x, y
                        )
                        Dp = self.compute_distance(np.array(cluster_center), p)
                        if Dp < self.distances[y][x]:
                            self.distances[y][x] = Dp
                            self.labels[y][x] = idx

    def update_cluster(self):
        sum_zs = np.zeros((len(self.cluster_centers), 2))
        num_zs = [0] * len(self.cluster_centers)
        for y in range(self.img_height):
            for x in range(self.img_width):
                label = self.labels[y][x]
                if label == -1:
                    continue
                sum_zs[label, :] += np.array([y, x])
                num_zs[label] += 1
        for idx in range(len(self.cluster_centers)):
            num_z = num_zs[idx]
            if num_z == 0:
                num_z = 1
            _y = sum_zs[idx][0] // num_z
            _x = sum_zs[idx][1] // num_z
            self.cluster_centers[idx] = [
                self.img[_y, _x][0], self.img[_y, _x][1], self.img[_y, _x][2], _x, _y
            ]

    def post_processing(self):
        new_img = np.copy(self.img)
        for y in range(self.img_height):
            for x in range(self.img_width):
                label = self.labels[y][x]
                if label == -1:
                    continue
                new_img[y, x][0] = self.cluster_centers[label][0]
                new_img[y, x][1] = self.cluster_centers[label][1]
                new_img[y, x][2] = self.cluster_centers[label][2]
        return new_img

    def process(self, show_every_loop=False):
        self.initialize()
        for it in range(self.max_iterations):
            print("Iteration {} start".format(it + 1))
            self.assign_samples()
            self.update_cluster()
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
    rgb = imread("./cat.jpeg")
    print(rgb.shape)
    # slic = SLIC(rgb, 5)
