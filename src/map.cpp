#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include "particle.h"
#include "map.h"


Map::Map(std::string path, int patch_size) {
    cv::imread(path).convertTo(map, CV_32F);

    if (map.empty()) {
        std::cerr << "Cant load the map image:" << path << std::endl;
        exit(1);
    }

    map = map / 255.0;
    _patch_size = patch_size;
}

//Gets a patch from the image with the _patch_size as the size
cv::Mat Map::get_patch(int x, int y) {
    cv::Mat patch = cv::Mat(map, cv::Rect(x-(_patch_size/2), y-(_patch_size/2), _patch_size, _patch_size));
    patch = patch.clone();
    return patch;
}

cv::Mat Map::visualize_particles(std::vector<particle> particles, float sum) {
    cv::Mat img = cv::Mat::zeros(map.size().height / 4, map.size().width / 4, CV_8UC3);

    for (particle p : particles) {
        //std::cout << p.weight << " ";        
        cv::circle(img, cv::Point(p.x / 4, p.y / 4), 5, cv::Scalar(0, 0, 230), cv::FILLED, cv::LINE_AA);
    }

    cv::GaussianBlur(img, img, cv::Size(9, 9), 0);

    //std::cout << std::endl;

    return img;
}

cv::Mat Map::visualize_particles(std::vector<particle> particles, float sum, std::array<int, 2> best_particle, std::array<int, 2> average_particle, std::array<int, 2> average_weighted_particle) {
    cv::Mat img = cv::Mat::zeros(map.size().height / 4, map.size().width / 4, CV_8UC3);

    for (particle p : particles) {
        //std::cout << p.weight << " ";        
        cv::circle(img, cv::Point(p.x / 4, p.y / 4), 5, cv::Scalar(0, 0, 230), cv::FILLED, cv::LINE_AA);
    }

    cv::GaussianBlur(img, img, cv::Size(9, 9), 0);

    cv::circle(img, cv::Point(best_particle[0] / 4, best_particle[1] / 4), 9, cv::Scalar(0, 75, 150), cv::FILLED, cv::LINE_AA);
    cv::circle(img, cv::Point(average_particle[0] / 4, average_particle[1] / 4), 9, cv::Scalar(255, 0, 0), cv::FILLED, cv::LINE_AA);
    cv::circle(img, cv::Point(average_weighted_particle[0] / 4, average_weighted_particle[1] / 4), 9, cv::Scalar(0, 255, 255), cv::FILLED, cv::LINE_AA);

    return img;

}

cv::Mat Map::visualize(std::vector<particle> particles, std::array<int, 3> drone) {
    cv::Mat img = map.clone();

    for (particle p : particles) {
        cv::circle(img, cv::Point(p.x, p.y), 20, cv::Scalar(0, 0, 255), cv::FILLED, cv::LINE_AA);
    }

    cv::circle(img, cv::Point(drone[0], drone[1]), 30, cv::Scalar(0, 255, 0), cv::FILLED, cv::LINE_AA);

    return img;
}

cv::Size Map::size() {
    return map.size();
}
