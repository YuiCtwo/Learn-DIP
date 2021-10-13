//
// Created by cyx on 2021/7/31.
//

#ifndef CXX_GMM_H
#define CXX_GMM_H
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class GMM {
public:
    // GMM 内部高斯模型的个数
    int gmm_component;
    explicit GMM(int component = 5);
    GMM(GMM &gmm)=delete;  // 不允许复制构造产生
    ~GMM();
    // 计算一个像素 RGB 属于这个混合高斯模型的概率
    double assign_to(const Vec3d& color) const;
    // 计算一个像素 RGB 属于第 k 个高斯模型的概率
    double assign_to(int k, const Vec3d& color) const;
    // 找到这个像素最可能属于 GMM 中的哪一个高斯模型
    int arg_min(const Vec3d& color) const;

    // 初始化模型
    void init();
    // 学习 GMM 参数
    void learning();
    // 交互过程中增加样本
    void add_sample(int k, const Vec3d& color);
    void print_param();

private:
    Mat_<double> cov;
    Mat_<double> mean;
    Mat_<double> weight;
    vector<double > det_cov;
    vector<Matx33d> cov_inv;
    // 缓存计算均值
    vector<Matx13d> sums_cache;
    // 缓存计算协方差
    vector<Matx33d> product_cache;
    int *k_samples{};
    int total_samples{};

};


#endif //CXX_GMM_H
