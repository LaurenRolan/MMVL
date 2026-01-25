
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <torch/script.h>
#include <chrono>
#include <math.h>
#include <cmath>

#include "SiamNet.h"
#include "particle.h"
#include "utilities.h"
#include "map.h"
#include "hl.h"
#include "cache.h"

/*
Parameters:
    map_ptr: Pointer to the map object that should be used
    model_ptr: Pointer to the embedding and comparing model that should be used
*/
HL::HL(Map* map_ptr, SiamNet* model_ptr) {
    m = map_ptr;
    map_size = m->size();
    model = model_ptr;
    make_grid();
}

void HL::make_grid() {

    //Makes the particles in a grid based on the STEP_SIZE 
    //and caches the embedding of that particle
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 200; i < map_size.height - 200; i += STEP_SIZE) {
        for (int k = 200; k < map_size.width - 200; k += STEP_SIZE) {
            std::cout << "\r" << "Creating particle " << particles.size() + 1 << std::flush;
            particle p = {i, k, 0.0};
            particles.push_back(p);

            cv::Mat patch = m->get_patch(i, k);

            at::Tensor tensor = model->cache_img(patch);
            tensor = tensor.contiguous();
            std::array<float, 16384> a;
            std::memcpy(a.data(), tensor.data_ptr<float>(), 16384*sizeof(float));
            cache.push_back(a);
        }
    }

    //cache = new Cache("../cache.dat", map_size.width, map_size.height);

    std::cout << "\r"
              << "Created a grid with " 
              << particles.size() 
              << " particles with cache of "
              << cache.size()
              << ". It took " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count() 
              << "ms" 
              << std::endl;
}

//Gets the amount of particles in x ond y direction
std::array<int, 2> HL::get_particles_dim() {
    int x = (map_size.width - 400) / STEP_SIZE;
    int y = (map_size.height - 400) / STEP_SIZE;

    return {x, y};
}

/*
Makes a sensor update with the cache
Parameters:
    drone_img: Image from the drone 
    rot: rotation of the drone
*/
void HL::sensor_update_cache(at::Tensor drone_cache) {
    sum_weight = 0;
    max_weight = 0;
    
    for (long unsigned int i = 0; i < particles.size(); i++) {
        particles[i].weight = model->compare(drone_cache, cache[i]);

        // if (particles[i].weight < 0.2) {
        //     particles[i].weight = 0.2;
        // } else {
        //     particles[i].weight = 0.8;
        // }

        // if (particles[i].weight < 0.3) {
        //     particles[i].weight = 0.3;
        // } else if (particles[i].weight > 0.8)
        // {
        //     particles[i].weight = 0.8;
        // }
        
        // particles[i].weight = std::fabs(particles[i].weight);
        if (particles[i].weight < 0) {
            particles[i].weight = 0;
        }
        particles[i].weight += 0.45;
        if (particles[i].weight > 1.0) {
            particles[i].weight = 1.0;
        } 
        

        //particles[i].weight = 0.2 + 0.6*particles[i].weight;
        //particles[i].weight = 0.3 + (0.5 / (1 + std::exp(- 10 * particles[i].weight + 4.5))); 
        
        sum_weight += particles[i].weight;
        if (particles[i].weight > max_weight) {
            max_weight = particles[i].weight;
        }
    }
}


cv::Mat HL::interpolation() {
    int x = get_particles_dim()[0];
    int y = get_particles_dim()[1];
    int padding = 200 / STEP_SIZE;
    cv::Mat img = cv::Mat::zeros(x + (2 * padding), y + (2 * padding), CV_32FC1);

    std::vector<particle> particles = get_particles();

    for (int i = 0; i < x; i++) {
        for (int k = 0; k < y; k++) {
            img.at<float>(k + padding, i + padding) = particles[(i*x)+k].weight;
        }
    }

    cv::resize(img, img, cv::Size(m->size().width, m->size().height), cv::INTER_LINEAR);
    
    return img;
}

void HL::normalize() {
    for (long unsigned int i = 0; i < particles.size(); i++) {
        particles[i].weight /= sum_weight;
    }
}

void HL::next_step(std::array<int, 3> movement, cv::Mat drone_img) {
    
    cv::Mat drone_rot;
    cv::Mat rotation = cv::getRotationMatrix2D(cv::Point2f((drone_img.cols-1) / 2.0, (drone_img.rows-1)/ 2.0), -movement[2], 1);
    cv::warpAffine(drone_img, drone_rot, rotation, drone_img.size());
    at::Tensor drone_cache = model->cache_img(drone_rot);
    sensor_update_cache(drone_cache);
    normalize();
}

void HL::next_step(at::Tensor drone_cache) {
    sensor_update_cache(drone_cache);
    normalize();
}
