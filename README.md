# Learn-DIP

《数字图像处理第四版》书中等一些基础算法的实现，包括相关库的调用

大三下选修了数字图像处理，书上讲的内容有很多，也有不少有意思的并且经典的算法，碍于学的时候的时间原因，没有整理实验课写的代码，我计划着大三暑假，大四整理整理并补充一些书本外的经典 DIP 用到的一些算法。

- 记录一些书上算法的复现和一些平时用到的算法，如果有库尽量会使用库函数来更快捷的完成，毕竟重复的轮子尽量少造。

## 构成

尽可能按照一个模块一个文件夹来，整体会用 Python 来实现，具体代码运行的环境参考文件：
- [Python 环境](./env_Python.md)


## 主要部分

- 频率域滤波
  - [带阻/带通滤波](FrequencyDomainFilter/band-stop.py)
  - [陷波滤波](FrequencyDomainFilter/notch.py)
- 图像去噪
  - [去除椒盐噪声(自适应中值法)](Denoising/AdaptMedianFilter)
  - [运动模糊恢复(维纳滤波&约束最小二乘方滤波)](Denoising/MotionBlurRecovery)
- 形态学
  - [骨架提取](Morphology/Skeleton)
- 图像分割
  - 点线和边缘检测
    - [线检测(霍夫变换)](ImageSegmentation/LowLevelDetector/Hough)
    - [Canny](ImageSegmentation/LowLevelDetector/Canny)
    - [Harris 角点检测](ImageSegmentation/FeatureDescriptors/HOG/harris.py)
  - 特征描述子
    - [HOG](ImageSegmentation/FeatureDescriptors/HOG)
  - 图割法分割
  - 聚类法分割
    - [K-Means](ImageSegmentation/Clustering/K-Means/segmentation.py)
    - [像素特征(基于像素点的图片分割)](ImageSegmentation/Clustering/Pixel-Features)
  - 超像素法分割
    - [简单线性迭代聚类(SLIC)](ImageSegmentation/SuperPixels/slic.py)
- 图像调整
  - Seam-Carving
  - [图像金字塔](ImageAdjusting/ImagePyramid/image_pyramid.py)
  - [Metric rectification](ImageAdjusting/AffineRectification/metric_rectification.py)
- 图像分类
  - [KNN](ImageClassification/KNN/k_nearest_neighbor.py)
- 目标追踪
  - Lucas-Kanade
    - [Simple Lucas-Kanade Feature Tracker](ObjectTracking/LucasKanade/simple_lucas_kanade.py)
    - [Pyramidal Lucas-Kanade Feature Tracker] WIP
  
## 额外部分

C++ 的代码整理了出来，姑且先放在这里一些，具体环境参考[C++ 环境](./env_CXX.md)

- [GMM](CXX/GMM/CXX)
- [LSD](CXX/LSD)
