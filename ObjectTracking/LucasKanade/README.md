# Lucas-Kanade 光流法

Lucas-Kanade 算法的基本思想是假定图片 **亮度连续（连续帧之间同一个位置的像素值不变）**，通过追踪该点的变化向量来追踪移动。
- 基于 CS131 给出了该算法的基本实现和一个改进算法：在迭代计算中加入了高斯图像金字塔。
- 更多细节参考个人博客

## **环境依赖注意**

需要处理视频流的工具 `FFmpeg`, 简述安装方式:

- 使用 Conda & Anaconda 安装
> conda install -c conda-forge ffmpeg

- Linux/Mac
> apt-get/brew install ffmpeg

- Windows 安装参考链接: [Install-FFmpeg-on-Windows](https://www.wikihow.com/Install-FFmpeg-on-Windows)