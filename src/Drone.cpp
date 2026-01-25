#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

#include "Drone.h"


Drone::Drone(std::string path, std::string truth_path) {
    _path = path;

    file.open(truth_path);
    if (!file.is_open()) {
        std::cerr << "Failed to open file" << std::endl;
        exit(1);
    }
}

//Update the position of the drone
void Drone::read_next_line() {
    std::string line;

    //If the drone is done with "flying" the position will be all negatives
    //can be used to stop the program
    if (!std::getline(file, line)) {
        pos = {-1, -1, -1};
    }

    std::vector<std::string> words;

    for (int i = 0; i < 3; i++) {
        size_t pos = line.find(delim);
        words.push_back(line.substr(0, pos));
        line.erase(0, pos + delim.length());
    }
    words.push_back(line);

    pos = {std::stoi(words[0]), std::stoi(words[1]), static_cast<int>(std::stof(words[3]))};
}

std::array<int, 3> Drone::get_pos() {
    return pos;
}

//Updates the next image and position, returns this image
cv::Mat Drone::get_image() {
    cv::Mat img;

    //Makes the image path on the right format
    std::string img_path = _path + 
                          "/image" + 
                          std::string(3 - std::to_string(idx).size(), '0') + 
                          std::to_string(idx) + 
                          ".png";
    cv::imread(img_path).convertTo(img, CV_32F);

    if (img.empty()) {
        std::cerr << "Cant load the drone image:" << img_path << std::endl;
        exit(1);
    }

    idx++;
    img = square_image(img);
    read_next_line();
    img = img / 255.0;
    return img;
}

cv::Mat Drone::square_image(cv::Mat img) {
    if (img.cols == img.rows) {
        return img;
    }

    int dim = std::min(img.cols, img.rows);
    int center_x = img.cols / 2;
    int center_y = img.rows / 2;

    //Makes the drone image square
    img = cv::Mat(img, cv::Rect(center_x - static_cast<int>(dim/2), center_y - static_cast<int>(dim/2), dim, dim));
    return img;
}
