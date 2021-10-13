## 骨架

前一段时间在看折痕消除的一篇论文的时候提到了提取文字的骨架，就重新复习一下《数字图像处理》这本书中关于骨架的知识。

概念原理见博客: [骨架提取](https://cyx0706.github.io/2021/09/29/dip4/#Skeleton)

`Python` 的 `skimage` 库中提供了可以直接调用的方法：

`skimage.morphology.skeletonize(image)`