
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
#include "logger.h"
#include <chrono>

/*
Parameters:
    map_ptr: Pointer to the map object that should be used
    model_ptr: Pointer to the embedding and comparing model that should be used
*/
HL::HL(Map* map_ptr, SiamNet* model_ptr) {
    m = map_ptr;
    map_size = m->size();
    model = model_ptr;
    cache_file = "cache_" + m->name() + ".gridcache";
    make_grid();
}

void HL::make_grid() {

    if (load_grid_cache())
    {
        spdlog::info("Loaded grid from cache.\n");
        return;
    }

    //Makes the particles in a grid based on the STEP_SIZE 
    //and caches the embedding of that particle
    auto start = std::chrono::high_resolution_clock::now();
    auto last_update = std::chrono::steady_clock::now();
    size_t total_steps = ((map_size.width - 400) / STEP_SIZE) * ((map_size.height - 400) / STEP_SIZE);
    size_t current_step = 0;
    for (int i = 200; i < map_size.height - 200; i += STEP_SIZE) {
        for (int k = 200; k < map_size.width - 200; k += STEP_SIZE) {
            spdlog::debug("Creating particle {}", particles.size() + 1);
            particle p = {i, k, 0.0};
            particles.push_back(p);

            cv::Mat patch = m->get_patch(i, k);

            at::Tensor tensor = model->cache_img(patch);
            tensor = tensor.contiguous();
            std::array<float, EMBEDDING_SIZE> a;
            std::memcpy(a.data(), tensor.data_ptr<float>(), EMBEDDING_SIZE * sizeof(float));
            cache.push_back(a);

            auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_update).count() > 200) {
                print_progress(current_step + 1, total_steps);
                last_update = now;
            }
            current_step++;
        }
    }

    spdlog::info("Created a grid with {} particles with cache of {}. It took {} ms", particles.size(), cache.size(), std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count());
    save_grid_cache();
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
    
    spdlog::debug("[HL] Running through {} particles, comparing drone cache with particle cache...", particles.size());
    auto last_update = std::chrono::steady_clock::now();

    for (size_t i = 0; i < particles.size(); i++) {
        particles[i].weight = model->compare(drone_cache, cache[i]);
        if (particles[i].weight < 0) {
            particles[i].weight = 0;
        }
        particles[i].weight += 0.45;
        if (particles[i].weight > 1.0) {
            particles[i].weight = 1.0;
        } 
        
        sum_weight += particles[i].weight;
        if (particles[i].weight > max_weight) {
            max_weight = particles[i].weight;
        }
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_update).count() > 200) {
            print_progress(i + 1, particles.size());
            last_update = now;
        }
    }
    spdlog::debug("[HL] Sensor update completed. Sum of weights: {}, Max weight: {}", sum_weight, max_weight);
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
    for (size_t i = 0; i < particles.size(); i++) {
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

#pragma region CACHE_FUNCTIONS
bool HL::save_grid_cache()
{
    spdlog::info("Saving grid cache to {}", cache_file);
    std::ofstream out(cache_file, std::ios::binary);
    if (!out)
        return false;

    spdlog::debug("Particles: {}", particles.size());
    spdlog::debug("Cache size: {}", cache.size());
    spdlog::debug("Cache bytes: {}", cache.size() * sizeof(std::array<float, EMBEDDING_SIZE>));

    // ---- Save particles ----
    uint64_t particleCount = particles.size();
    out.write(reinterpret_cast<const char*>(&particleCount),
              sizeof(particleCount));

    out.write(reinterpret_cast<const char*>(particles.data()),
              particleCount * sizeof(particle));

    // ---- Save cache ----
    uint64_t cacheSize = cache.size();
    out.write(reinterpret_cast<const char*>(&cacheSize),
              sizeof(cacheSize));

    for (size_t i = 0; i < cache.size(); ++i)
    {
        out.write(reinterpret_cast<const char*>(cache[i].data()),
                EMBEDDING_SIZE * sizeof(float));
    }

    return true;
}

bool HL::load_grid_cache()
{
    spdlog::info("Trying to load grid cache from {}", cache_file);
    std::ifstream in(cache_file, std::ios::binary);
    if (!in)
        return false;

    // ---- Load particles ----
    uint64_t particleCount;
    in.read(reinterpret_cast<char*>(&particleCount),
            sizeof(particleCount));

    particles.resize(particleCount);
    in.read(reinterpret_cast<char*>(particles.data()),
            particleCount * sizeof(particle));

    // ---- Load cache ----
    uint64_t cacheSize;
    in.read(reinterpret_cast<char*>(&cacheSize),
            sizeof(cacheSize));

    cache.resize(cacheSize);

    for (auto& arr : cache)
    {
        in.read(reinterpret_cast<char*>(arr.data()),
                EMBEDDING_SIZE * sizeof(float));
    }

    return true;
}
#pragma endregion