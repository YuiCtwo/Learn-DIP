# C++ 环境说明

C++ 的环境配起来相较于 Python 要费劲很多。

首先，所有代码的 C++ 版本在 C++ 14 标准及以上（可能），具体会在每个目录下的 CMake 文件中给出。

基础环境是 Windows + CMake + MinGW64，下面细说一下第三方库。

## OpenCV

我的环境是 OpenCV 在 MinGW64 的编译发行版，可以从 [这里下载](https://github.com/huihut/OpenCV-MinGW-Build)，具体的配置问题参考 [这篇博客](https://blog.csdn.net/huihut/article/details/81317102)

CMake 配置如下，供参考：
```CMake
set(OpenCV_DIR "path/of/opencv")
FIND_PACKAGE(OpenCV REQUIRED)
IF (OpenCV_FOUND)
    INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
    TARGET_LINK_LIBRARIES(your_project_name ${OpenCV_LIBS})
ELSE (OpenCV_FOUND)
    MESSAGE(FATAL_ERROR "OpenCV library not found")
ENDIF (OpenCV_FOUND)
```

## Eigen

从 [官网](https://eigen.tuxfamily.org/index.php?title=Main_Page) 下载即可

```CMake
include_directories("path/of/eigen")
```

上面两个是比较基础的库，基本所有里面都会用到。其余的第三方库会在使用的时候放入和项目同一个目录下的 `lib`，`include` 目录里，并且会有 `README.md` 来说明一些额外配置。 