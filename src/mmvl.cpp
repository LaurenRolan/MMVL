
#include <vector>
#include <opencv2/imgproc.hpp>
#include "opencv2/highgui.hpp"
#include <chrono>
#include <math.h>
#include <algorithm>

#include "mmvl.h"
#include "hl.h"
#include "particle.h"
#include "map.h"
#include "utilities.h"
#include "SiamNet.h"

MMVL::MMVL(int num_particles, Map* map_ptr, SiamNet* model_ptr, std::vector<HL*> HL_list): MCL(num_particles, map_ptr, model_ptr) {
   
    this->HL_list = HL_list;
    number_maps = HL_list.size();

}

cv::Mat MMVL::interpolation() {
    int x = HL_list[0]->get_particles_dim()[0];
    int y = HL_list[0]->get_particles_dim()[1];
    int padding = 200 / STEP_SIZE;
    cv::Mat img = cv::Mat::zeros(x + (2 * padding), y + (2 * padding), CV_32FC1);

    std::vector<std::vector<particle>> particles;

    for (int i = 0; i < number_maps; i++) {
        particles.push_back(HL_list[i]->get_particles());
    }

    for (int i = 0; i < x; i++) {
        for (int k = 0; k < y; k++) {
            float im= 0.0;
            for (int p = 0; p < number_maps; p++) {
                im += particles[p][(i*x)+k].weight;
            }
            img.at<float>(k + padding, i + padding) = im / number_maps;
        }
    }

    cv::resize(img, img, cv::Size(m->size().width, m->size().height), cv::INTER_CUBIC);
    
    return img;
}

void MMVL::sensor_update(cv::Mat drone_img, int rot) {
    sum_weight = 0;
    cv::Mat drone_rot;
    cv::Mat rotation = cv::getRotationMatrix2D(cv::Point2f((drone_img.cols-1) / 2.0, (drone_img.rows-1)/ 2.0), -rot, 1);
    cv::warpAffine(drone_img, drone_rot, rotation, drone_img.size());
    at::Tensor drone_cache = model->cache_img(drone_rot);

    for (int i = 0; i < number_maps; i++) {
        HL_list[i]->next_step(drone_cache);
    }

    auto start = std::chrono::high_resolution_clock::now();
    cv::Mat img_weights = interpolation();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Interpolation " << duration.count() << std::endl;

    for (long unsigned int i = 0; i < particles.size(); i++) {
        particles[i].weight = img_weights.at<float>(particles[i].y, particles[i].x);

        sum_weight += particles[i].weight;
    }
}

void MMVL::next_step(std::array<int, 3> movement, cv::Mat drone_img) {

    MCL::motion_update(movement[0], movement[1], movement[2]);
    sensor_update(drone_img, movement[2]);
    Localization::normalize();   
    MCL::resample();
    Localization::update_stats();
}