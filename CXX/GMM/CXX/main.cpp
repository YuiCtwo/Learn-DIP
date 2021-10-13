#include <iostream>
#include "GMM.h"

// 一个简易的自动获取 mask 的方法
void get_mask_contour(Mat &img, Mat &mask) {
    Mat gray_img;
    cv::cvtColor(img, gray_img, cv::COLOR_BGR2GRAY);
    uchar threshold = 190; // 随便设的值
    mask = Mat::zeros(gray_img.size(), CV_8UC1);
    for (int row = 0; row < gray_img.rows; row++) {
        for (int col = 0; col < gray_img.cols; col++) {
            if (gray_img.at<uchar>(row, col) < threshold) {
                mask.at<uchar>(row, col) = 255;
            }
        }
    }
    cv::Size m_Size = mask.size();
    Mat Temp = Mat::zeros(m_Size.height + 2, m_Size.width + 2, mask.type()); // 延展图像
    mask.copyTo(Temp(cv::Range(1, m_Size.height + 1), cv::Range(1, m_Size.width + 1)));
    cv::floodFill(Temp, cv::Point(0, 0), cv::Scalar(255));
    Mat cutImg; // 裁剪延展的图像
    Temp(cv::Range(1, m_Size.height + 1), cv::Range(1, m_Size.width + 1)).copyTo(cutImg);

    mask = mask | (~cutImg);
//    mask = ~mask;
    Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    Mat dilate_out, erode_out;
    // 膨胀
    cv::dilate(mask, dilate_out, element);
    //腐蚀
    cv::erode(dilate_out, mask, element);
}

void init_GMM(const Mat &img, const Mat &mask, GMM &bgGMM, GMM &fgGMM) {
    assert(bgGMM.gmm_component == fgGMM.gmm_component);
    // 用 K-means 算法初始化 GMM 前景和背景
    const int t = 10;  // KMeans 迭代次数
    Mat bg_label, fg_label;  // 前景背景每一个像素都属于哪个高斯模型, 对应论文中的 kn
    vector<Vec3d> bg_samples, fg_samples;  // 从图片中取出的样本集
    Point idx;
    for (idx.y = 0; idx.y < img.rows; ++idx.y) {
        for (idx.x = 0; idx.x < img.cols; ++idx.x) {
            // 按照 mask 中的类别收集 img 中的前后景样本
            // 黑色区域值为 0, 是背景
            int which_class = mask.at<uchar>(idx);
            if (which_class == GC_BGD || which_class == GC_PR_BGD) {
                bg_samples.push_back((Vec3d) img.at<Vec3b>(idx));
            } else {
                fg_samples.push_back((Vec3d) img.at<Vec3b>(idx));
            }
        }
    }
    // 不能为空
    CV_Assert(!bg_samples.empty() && !fg_samples.empty());

    // GMM 初始化: 用聚类方法将 bg_samples, fg_samples 分成 GMM 中高斯模型个数类
    // 这样就能得到每个像素所属的高斯模型, 可以估计概率值

    // 调整为 size()x3 的矩阵
    Mat _bg_samples((int) bg_samples.size(), 3, CV_32FC1, &bg_samples[0][0]);
    Mat _fg_samples((int) fg_samples.size(), 3, CV_32FC1, &fg_samples[0][0]);

    // 中心点
    Mat bg_centers = Mat::zeros(bgGMM.gmm_component, 3, CV_32FC1);
    Mat fg_centers = Mat::zeros(bgGMM.gmm_component, 3, CV_32FC1);
    kmeans(_bg_samples, bgGMM.gmm_component, bg_label,
           TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, t, 0.0),
           0, KMEANS_PP_CENTERS);
    kmeans(_fg_samples, bgGMM.gmm_component, fg_label,
           TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, t, 0.0),
           0, KMEANS_PP_CENTERS);
    // 初始化参数
    bgGMM.init();
    // 增加样本, 此时已经有初步的每个样本属于哪个高斯模型的划分
    for (int i = 0; i < (int) bg_samples.size(); ++i) {
        bgGMM.add_sample(bg_label.at<int>(i, 0), bg_samples[i]);
    }
    bgGMM.learning();
    fgGMM.init();
    for (int i = 0; i < (int) fg_samples.size(); ++i) {
        fgGMM.add_sample(fg_label.at<int>(i, 0), fg_samples[i]);
    }
    fgGMM.learning();
}
int main() {
    // 仅供测试使用
    Mat img = imread(R"(D:\code\learn_DIP\GMM\CXX\lena.jpg)");

    if (img.empty()) {
        cout << "Error" << endl;
        return 1;
    }
    Mat mask;
    get_mask_contour(img, mask);
    GMM bg_model, fg_model;
    init_GMM(img, mask, bg_model, fg_model);
    bg_model.print_param();
    fg_model.print_param();
    return 0;
}