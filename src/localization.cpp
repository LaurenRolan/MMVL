#include "opencv2/highgui.hpp"
#include <torch/script.h>
#include <opencv2/imgproc.hpp>
#include <chrono>

#include "localization.h"
#include "SiamNet.h"
#include "particle.h"
#include "utilities.h"
#include "map.h"
#include "cache.h"

std::vector<particle> Localization::get_particles() {
    return particles;
}

std::array<int, 2> Localization::get_best_particle() {
    return best_position;
}

std::array<int, 2> Localization::get_average_position() {
    return average_position;
}

std::array<int, 2> Localization::get_average_weighted_position() {
    return average_weighted_position;
}

/*
Makes a measurment of the environment and gives the particles a weight 
Parameters:
    drone_img: Image from the drone
    rot: Rotation of the drone
*/
void Localization::sensor_update(cv::Mat drone_img, int rot) {
    sum_weight = 0;
    max_weight = 0;

    //Rotate the image
    cv::Mat drone_rot;
    cv::Mat rotation = cv::getRotationMatrix2D(cv::Point2f((drone_img.cols-1) / 2.0, (drone_img.rows-1)/ 2.0), -rot, 1);
    cv::warpAffine(drone_img, drone_rot, rotation, drone_img.size());
    
    //Caching the embedding of the drone image
    at::Tensor drone_cache = model->cache_img(drone_rot);

    //Goes through the particles, extract patches and compare with the embedding from the drone
    for (long unsigned int i = 0; i < particles.size(); i++) {
        cv::Mat patch = m->get_patch(particles[i].x, particles[i].y);
        particles[i].weight = model->compare(drone_cache, patch);
        sum_weight += particles[i].weight;
        if (particles[i].weight > max_weight) {
            max_weight = particles[i].weight;
        }
    }
}

//Calculates and updates all of the stats
void Localization::update_stats() {
    average_position = {0, 0};
    average_weighted_position = {0, 0};
    float w = 0.0;

    std::array<float, 2> im = {0.0, 0.0};

    for (unsigned long i = 0; i < particles.size(); i++) {
        average_position[0] += particles[i].x;
        average_position[1] += particles[i].y;

        im[0] += particles[i].x * particles[i].weight;
        im[1] += particles[i].y * particles[i].weight;

        if (particles[i].weight > w) {
            w = particles[i].weight;
            best_position[0] = particles[i].x;
            best_position[1] = particles[i].y;
        }
    }

    average_position[0] /= particles.size();
    average_position[1] /= particles.size();
    average_weighted_position[0] = static_cast<int>(im[0]);
    average_weighted_position[1] = static_cast<int>(im[1]);
}

//Normalize the weights of the particles
void Localization::normalize() {

    max_weight = 0;

    //std::cout << "sum: " << sum_weight << std::endl;
    for (long unsigned int i = 0; i < particles.size(); i++) {
        particles[i].weight /= sum_weight;

        //std::cout << particles[i].weight << " ";

        if (particles[i].weight > max_weight) {
            max_weight = particles[i].weight;
        }
    }
}
