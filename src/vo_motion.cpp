#include <iostream>
#include <fstream>
#include <string>
#include <array>
#include <vector>
#include <math.h>

#include "vo_motion.h"
#include "utilities.h"

VO_Motion::VO_Motion(std::string const path) {
    file.open(path + "/odom.txt");
    if (!file.is_open()) {
        std::cerr << "Failed to open file" << std::endl;
        exit(1);
    }

    traj_file.open(path + "/traj.txt");
    if (!traj_file.is_open()) {
        std::cerr << "Failed to open file" << std::endl;
        exit(1);
    }

}

std::array<float, 4> VO_Motion::read_translation() {

    std::string line;

    //If the drone is done with "flying" the position will be all negatives
    //can be used to stop the program
    if (!std::getline(file, line)) {
        return {-1, -1};
    }

    std::vector<std::string> words;

    for (int i = 0; i < 3; i++) {
        size_t pos = line.find(delim);
        words.push_back(line.substr(0, pos));
        line.erase(0, pos + delim.length());
    }
    words.push_back(line);

    return {std::stof(words[0]), std::stof(words[1]), std::stof(words[2]), std::stof(words[3])};
}

float VO_Motion::read_rotation() {

    std::string line;

    //If the drone is done with "flying" the position will be all negatives
    //can be used to stop the program
    if (!std::getline(traj_file, line)) {
        return 0;
    }

    std::vector<std::string> words;

    for (int i = 0; i < 3; i++) {
        size_t pos = line.find("\t");
        words.push_back(line.substr(0, pos));
        line.erase(0, pos + 1);
    }
    words.push_back(line);

    return std::stof(words[3]) + ((random_number()*(COMPASS_RANGE*2)) - COMPASS_RANGE);
    // return std::stof(words[3]);

}

//Reads the next line in the file and calculate the motion
std::array<int, 3> VO_Motion::next_update() {
    std::array<float, 4> translation = read_translation();
    float rot = read_rotation();

    // scale = scale * translation[2];

    // rot += translation[2];
    // float x = 0.7*((translation[0] * std::cos(rot)) + (translation[1] * std::sin(rot)));
    // float y = 0.7*((translation[0] * std::sin(rot)) + (translation[1] * std::cos(rot)));

    std::array<int, 3> ret = {static_cast<int>(translation[0]), static_cast<int>(translation[1]), static_cast<int>(rot)};

    return ret;
}