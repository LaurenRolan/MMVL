#include "utilities.h"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <string>
#include <iostream>
#include <iomanip>
#include <random>

float get_gaussian_random_number(double mean, double var) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(mean, var);
    return static_cast<float>(dist(gen));
}

float random_number() {
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    return static_cast<float>(dist(rng));
}

float random_number(float min, float max) {
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<double> dist(min, max);
    return static_cast<float>(dist(rng));
}

void display_img(cv::Mat img, std::string name, cv::Size size) {
    cv::namedWindow(name, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
    cv::resizeWindow(name, size.width, size.height);
    cv::imshow(name, img);
}

int distance_between(std::array<int, 2> pos_a, std::array<int, 2> pos_b) {
    return static_cast<int>(std::round(std::sqrt(std::pow(pos_a[0] - pos_b[0], 2) + std::pow(pos_a[1] - pos_b[1], 2))));
}

void print_progress(size_t current, size_t total)
{
    const int bar_width = 50;

    float progress = static_cast<float>(current) / total;

    std::cout << "\r[";
    int pos = bar_width * progress;

    for (int i = 0; i < bar_width; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }

    std::cout << "] "
              << std::setw(3) << int(progress * 100.0) << "% ("
              << static_cast<long>(current) << "/" << static_cast<long>(total) << ")"
              << std::flush;

    if (current == total)
        std::cout << std::endl;
}