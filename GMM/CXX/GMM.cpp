//
// Created by cyx on 2021/7/31.
//

#include <opencv2/core.hpp>
#include "GMM.h"


GMM::GMM(int component) {
    // RGB 3 通道对应每一个高斯模型能的得到 1 个权重
    this->gmm_component = component;
    this->weight = Mat::zeros(gmm_component, 1, CV_64F);
    // RGB 3 通道对应每一个高斯模型能的得到 3x3 个协方差
    int sizes[] = {gmm_component, 3, 3};
    this->cov = Mat(3, sizes, CV_64F, Scalar::all(0));
    // RGB 3 通道对应每一个高斯模型能的得到 3 个均值
    this->mean = Mat::zeros(gmm_component, 3, CV_64F);
    this->k_samples = new int[gmm_component];
    for (int i = 0; i < gmm_component; ++i) {
        this->k_samples[i] = 0;
        this->det_cov.push_back(0);
        this->sums_cache.emplace_back(0, 0, 0);
        this->product_cache.emplace_back(0, 0, 0);
        this->cov_inv.emplace_back(0, 0, 0);
    }
}


double GMM::assign_to(const Vec3d& color) const {
    double res = 0;
    double numerator = 0;
    Vec3d dst;
    for(int k = 0; k < gmm_component; k++) {
        numerator = 0;
        // 只计算概率大于 0 的部分
        if (this->weight.at<double >(k) > 0) {
            // 高阶的高斯密度模型计算式
            dst = {this->mean.at<Vec3d>(k) - color};
            numerator += dst[0] * (dst[0] * this->cov_inv[k](0, 0)
                                   + dst[1] * this->cov_inv[k](1, 0)
                                   + dst[2] * this->cov_inv[k](2, 0));
            numerator += dst[1] * (dst[0] * this->cov_inv[k](0, 1)
                                   + dst[1] * this->cov_inv[k](1, 1)
                                   + dst[2] * this->cov_inv[k](2, 1));
            numerator += dst[2] * (dst[0] * this->cov_inv[k](0, 2)
                                   + dst[1] * this->cov_inv[k](1, 2)
                                   + dst[2] * this->cov_inv[k](2, 2));
            res += this->weight.at<double >(k) * (1.0f/sqrt(this->det_cov[k]) * exp(-0.5f*numerator));
        }
    }
    return res;

}

double GMM::assign_to(int k, const Vec3d& color) const {
    double res = 0;
    double numerator = 0;
    // 只计算概率大于 0 的部分
    if (this->weight.at<double >(k) > 0) {
        // 高阶的高斯密度模型计算式
        Vec3d dst = {this->mean.at<Vec3d>(k) - color};
        numerator += dst[0] * (dst[0] * this->cov_inv[k](0, 0)
                               + dst[1] * this->cov_inv[k](1, 0)
                               + dst[2] * this->cov_inv[k](2, 0));
        numerator += dst[1] * (dst[0] * this->cov_inv[k](0, 1)
                               + dst[1] * this->cov_inv[k](1, 1)
                               + dst[2] * this->cov_inv[k](2, 1));
        numerator += dst[2] * (dst[0] * this->cov_inv[k](0, 2)
                               + dst[1] * this->cov_inv[k](1, 2)
                               + dst[2] * this->cov_inv[k](2, 2));
//        double numerator = (dst * this->cov_inv[k] * dst.t())(0);
        res = 1.0f/sqrt(this->det_cov[k]) * exp(-0.5f*numerator);
    }
    return res;
}


int GMM::arg_min(const Vec3d& color) const {
    int k = 0;
    double maxx = 0;
    for (int i = 0; i < gmm_component; ++i) {
        double p = this->assign_to(i, color);
        if (p > maxx) {
            k = i;
            maxx = p;
        }
    }
    return k;
}

void GMM::add_sample(int k, const Vec3d& color) {
    Matx13d color_mat = Matx13d(color[0], color[1], color[2]);
    this->k_samples[k] += 1;
    this->total_samples += 1;
    this->sums_cache[k] += color_mat;
    this->product_cache[k] += (color_mat.t() * color_mat);
}

// 从图像数据中学习 GMM 的参数: 每一个高斯分量的权值,均值和协方差矩阵
void GMM::learning(){
    const double variance = 0.01;
    for (int ci = 0; ci < gmm_component; ++ci) {
        int n = this->k_samples[ci];
        if (n == 0){
            this->weight(ci, 0) = 0;
        }
        else {
            // 第 ci 个高斯模型的权重
            this->weight(ci, 0) = (double) n / this->total_samples;
            // 第 ci 个高斯模型的均值
            Mat dst;
            auto &ci_cov = cov.at<Matx33d>(ci);
            divide(this->sums_cache[ci], n, dst);
            dst.row(0).copyTo(this->mean.row(ci));
            // 第 ci 个高斯模型的协方差
            Mat cov_var;
            divide(this->product_cache[ci], n, cov_var);
            cov_var -= (dst.t() * dst);
            cov_var.copyTo(ci_cov);
            double det = determinant(ci_cov);
            // 防止行列式的值小于等于 0 导致无法计算逆矩阵
            if (det <= std::numeric_limits<double >::epsilon()) {
                this->cov.at<double >(ci, 0, 0) += variance;
                this->cov.at<double >(ci, 1, 1) += variance;
                this->cov.at<double >(ci, 2, 2) += variance;
                det = determinant(ci_cov);
            }
            // 确保行列式得到的结果是正确的
            CV_Assert( det > std::numeric_limits<double>::epsilon());
            this->det_cov[ci] = det;
            this->cov_inv[ci] = ci_cov.inv();
        }
    }
//    this->print_param();
}

void GMM::print_param() {
    cout << format(this->weight, Formatter::FMT_PYTHON) << endl;
    cout << format(this->mean, Formatter::FMT_PYTHON) << endl;
    cout << "det of cov" << endl;
    for (double i : this->det_cov) {
        cout << i << " ";
    }
    cout << "inv of cov" << endl;
    cout << format(this->cov_inv, Formatter::FMT_PYTHON);
    cout << endl;
}

void GMM::init() {
    for (int i = 0; i < gmm_component; ++i) {
        // 初始化学习
        this->k_samples[i] = 0;
        this->sums_cache[i] =  {0, 0, 0};
        // 不显示指定初始化为 0 可能会出现某个值是很小的数
        this->product_cache[i] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    }
    this->total_samples = 0;
}

GMM::~GMM() {
    delete this->k_samples;
}
