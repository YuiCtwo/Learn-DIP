//
// Created by cyx on 2021/7/10.
//
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <algorithm>
#include "lsd.h"
#include "Coordinate.h"

typedef cv::Mat CVMat;
typedef Coordinate<double> CoordinateDouble;

using namespace std;

struct LineD {
    double row1, col1;
    double row2, col2;
    LineD(double row1, double col1, double row2, double col2) {
        this->row1 = row1;
        this->row2 = row2;
        this->col1 = col1;
        this->col2 = col2;
    }
    LineD() {
        row1 = 0; col1 = 0; row2 = 0; col2 = 0;
    }
    LineD(CoordinateDouble p1, CoordinateDouble p2){
        row1 = p1.row; row2 = p2.row;
        col1 = p1.col; col2 = p2.col;
    }

};

void draw_line(CVMat &img, LineD& line) {
    cv::Point start((int)line.col1, (int)line.row1);
    cv::Point end((int)line.col2, (int)line.row2);
    int thickness = 1;
    int lineType = 1;
    cv::line(img, start, end, cv::Scalar(0, 255, 0), thickness, lineType);
}

void detected_line_without_border(CVMat &img, CVMat &mask, vector<LineD>& result) {
    // ROI region: mask[i, j] == 0
    CVMat gray_img;
    cv::cvtColor(img, gray_img, cv::COLOR_BGR2GRAY);
    int rows = gray_img.rows;
    int cols = gray_img.cols;
    auto *image = new double[rows * cols];
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            image[row*cols+col] = gray_img.at<uchar>(row, col);
        }
    }
    double *out;
    int num_lines;
    out = lsd(&num_lines, image, cols, rows);
    for (int i = 0; i < num_lines; ++i) {
        LineD line(out[i*7+1], out[i*7+0], out[i*7+3], out[i*7+2]);
        if (mask.at<uchar>(line.row1, line.col1) == 0 && mask.at<uchar>(line.row2, line.col2) == 0) {
            result.push_back(line);
        }
    }
    delete[] image;

}

void detected_lines(CVMat &img, CVMat &mask, vector<LineD>& result) {
    detected_line_without_border(img, mask, result);
}

int main() {
    // revise your path here
    string path = R"(D:\code\learn_DIP\CXX\LSD\road.jpg)";
    CVMat mask, img;
    vector<LineD> line_segments;
    img = cv::imread(path);
    mask = CVMat::zeros(img.size(), CV_8UC1);
    detected_lines(img, mask, line_segments);
    for (auto p: line_segments) {
        draw_line(img, p);
    }
    cv::namedWindow("LSD", cv::WINDOW_AUTOSIZE);
    cv::imshow("LSD", img);
    cv::waitKey(0);
    return 0;
}
