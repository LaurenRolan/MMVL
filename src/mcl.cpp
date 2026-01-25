
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <torch/script.h>
#include <string>
#include <chrono>

#include "SiamNet.h"
#include "particle.h"
#include "utilities.h"
#include "map.h"
#include "cache.h"
#include "mcl.h"

/*
Parameters:
    num_particles: The number of particles that should be created and used
    map_ptr: Pointer to the map object that should be used
    model_ptr: Pointer to the embedding and comparing model that should be used
*/
MCL::MCL(int num_particles, Map* map_ptr, SiamNet* model_ptr) {
    num_p = num_particles;
    m = map_ptr;
    map_size = m->size();
    model = model_ptr;

    //Creates particles with random position within 200 pixels of the boarder of the image
    for (int i = 0; i < num_p; i++) {
        particle p = {static_cast<int>(random_number()*(map_size.height - 400)) + 200, static_cast<int>(random_number()*(map_size.width - 400)) + 200, 0};
        particles.push_back(p);   
    }

}

/*
Update the position of the particles.
If particles are within 200 pixels of the boarders the particles are placed 200 pixels from the boarder.
Parameters:
    x: value of the movement in x direction
    y: value of the movement in y direction
*/
void MCL::motion_update(int x, int y) {
    for (long unsigned int i = 0; i < particles.size(); i++) {
        particles[i].x += x;
        particles[i].y += y;

        if (particles[i].x < 200) {
            particles[i].x = 200;
        } else if (particles[i].x > map_size.width - 200) {
            particles[i].x = map_size.width - 200;
        }

        if (particles[i].y < 200) {
            particles[i].y = 200;
        } else if (particles[i].y > map_size.height - 200) {
            particles[i].y = map_size.height - 200;
        }

    }
}

/*
Update the position of the particles.
If particles are within 200 pixels of the boarders the particles are placed 200 pixels from the boarder.
Parameters:
    x: value of the movement in x direction relative to the drone
    y: value of the movement in y direction relative to the drone
    rot: rotation of the drone
*/
void MCL::motion_update(int x, int y, int rot) {
    for (long unsigned int i = 0; i < particles.size(); i++) {
        float rot_rad = (M_PI / 180) * (rot - 90);

        // float x_f = x + random_number(-15, 15);
        // float y_f = y + random_number(-10, 10);

        // float x_f = x + random_number(-x, x);
        // float y_f = y + random_number(-y, y);
        // float x_f = x + get_gaussian_random_number(0, 15);
        // float y_f = y + get_gaussian_random_number(0, 15);
        
        float x_f = x;
        float y_f = y;

        if (x_f < 10) {
            x_f += get_gaussian_random_number(0, 10);
        } else {
            x_f += get_gaussian_random_number(0, x);
        }

        if (y_f < 10) {
            y_f += get_gaussian_random_number(0, 10);
        } else {
            y_f += get_gaussian_random_number(0, y);
        }
                
                        
        particles[i].x += static_cast<int>(((x_f * std::cos(rot_rad)) + (y_f * std::sin(rot_rad))));
        particles[i].y += static_cast<int>(((x_f * std::sin(rot_rad)) + (y_f * std::cos(rot_rad))));

        // particles[i].x += x + get_gaussian_random_number(0, 15);
        // particles[i].y += y + get_gaussian_random_number(0, 15);

        // particles[i].x += ((x * std::cos(rot_rad)) + (y * std::sin(rot_rad)));
        // particles[i].y += ((x * std::sin(rot_rad)) + (y * std::cos(rot_rad)));

        if (particles[i].x < 200) {
            particles[i].x = 200;
        } else if (particles[i].x > map_size.width - 200) {
            particles[i].x = map_size.width - 200;
        }

        if (particles[i].y < 200) {
            particles[i].y = 200;
        } else if (particles[i].y > map_size.height - 200) {
            particles[i].y = map_size.height - 200;
        }

    }
}

/*
Resample the particles with a wheel algorithm
*/
void MCL::resample() {
    int index = random_number() * particles.size();
    double beta = 0.0;
    std::vector<particle> new_sample = particles;
    for (int i = 0; i < num_p; i++) {
        beta += random_number() * 2 * max_weight;
        while (beta > particles[index].weight) {
            beta -= particles[index].weight;
            index = (index + 1) % particles.size();
        }

        new_sample[i] = particles[index]; 
    }

    particles = new_sample;
}

//Adds noise to the particles
void MCL::add_gaussian() {
    for (long unsigned int i = 0; i < particles.size(); i++) {
        particles[i].x = get_gaussian_random_number(static_cast<double>(particles[i].x), 15);
        particles[i].y = get_gaussian_random_number(static_cast<double>(particles[i].y), 15);
    }
}

/*
Makes the next step in mcl
Parameters:
    movement: Array with the movement on the form [x, y, rot]
    drone_image: Drone image used to make a measurment
*/
void MCL::next_step(std::array<int, 3> movement, cv::Mat drone_img) {
    auto start = std::chrono::high_resolution_clock::now();
    motion_update(movement[0], movement[1], movement[2]);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Motion update: " << duration.count() << std::endl;

    start = std::chrono::high_resolution_clock::now(); 
    Localization::sensor_update(drone_img, movement[2]);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Sensor update: " << duration.count() << std::endl;

    Localization::normalize();

    start = std::chrono::high_resolution_clock::now();
    resample();
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Resample: " << duration.count() << std::endl;
    
    Localization::update_stats();
}
