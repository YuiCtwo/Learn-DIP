# 以 skimage 库自带的图片为例子演示如何调用

from skimage import morphology, data, color
import matplotlib.pyplot as plt

image = color.rgb2gray(data.horse())
# 获得一个二值图像
# 白色部分为我们关心的图像内容
image = 1 - image
# 实施骨架算法
skeleton = morphology.skeletonize(image)

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
ax1.imshow(image, cmap=plt.cm.gray)
ax1.axis('off')
ax1.set_title('original')
ax2.imshow(skeleton, cmap=plt.cm.gray)
ax2.axis('off')
ax2.set_title('skeleton')
fig.tight_layout()
plt.show()
plt.savefig("res.jpg")
