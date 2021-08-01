cmake_minimum_required(VERSION 3.19)
project(CXX)

set(CMAKE_CXX_STANDARD 14)

add_executable(CXX main.cpp GMM.cpp GMM.h)

set(OpenCV_DIR "D:\\OpenCV-MinGW-Build")
FIND_PACKAGE(OpenCV REQUIRED)
IF (OpenCV_FOUND)
    INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
    TARGET_LINK_LIBRARIES(CXX ${OpenCV_LIBS})
ELSE (OpenCV_FOUND)
    MESSAGE(FATAL_ERROR "OpenCV library not found")
ENDIF (OpenCV_FOUND)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")