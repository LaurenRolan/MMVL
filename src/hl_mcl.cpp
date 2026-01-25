
#include <vector>
#include <opencv2/imgproc.hpp>
#include "opencv2/highgui.hpp"
#include <chrono>
#include <math.h>

#include "hl_mcl.h"
#include "hl.h"
#include "particle.h"
#include "map.h"
#include "utilities.h"

HL_MCL::HL_MCL(int num_particles, Map* map_ptr, SiamNet* model_ptr, HL* hl_ptr): MCL(num_particles, map_ptr, model_ptr) {
   
    hl = hl_ptr;

}

cv::Mat HL_MCL::interpolation() {
    int x = hl->get_particles_dim()[0];
    int y = hl->get_particles_dim()[1];
    int padding = 200 / STEP_SIZE;
    cv::Mat img = cv::Mat::zeros(x + (2 * padding), y + (2 * padding), CV_32FC1);

    std::vector<particle> particles = hl->get_particles();

    for (int i = 0; i < x; i++) {
        for (int k = 0; k < y; k++) {
            img.at<float>(k + padding, i + padding) = particles[(i*x)+k].weight;
        }
    }

    // cv::resize(img, img, cv::Size(m->size().width, m->size().height), cv::INTER_CUBIC);
    cv::imwrite("small.png", img*255);
    cv::resize(img, img, cv::Size(m->size().width, m->size().height), cv::INTER_LINEAR);
    cv::imwrite("large.png", img*255);
    

    //display_img(img, "wo blur");

    // cv::Mat blur;

    // cv::blur(img, blur, cv::Size(150, 150));
    // cv::blur(blur, blur, cv::Size(150, 150));
    // cv::blur(blur, blur, cv::Size(100, 100));
    // cv::blur(blur, blur, cv::Size(100, 100));
    
    return img;
}

void HL_MCL::sensor_update(cv::Mat drone_img, int rot) {
    sum_weight = 0;
    hl->next_step({0, 0, rot}, drone_img);

    auto start = std::chrono::high_resolution_clock::now();
    cv::Mat img_weights = interpolation();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Interpolation " << duration.count() << std::endl;

    for (long unsigned int i = 0; i < particles.size(); i++) {
        particles[i].weight = img_weights.at<float>(particles[i].y, particles[i].x);

        /*if (particles[i].weight < 0.2) {
            particles[i].weight = 0.2;
        } else {
            particles[i].weight = 0.8;
        }*/


        /*if (particles[i].weight > 0.6) {
            particles[i].weight = 0.6;
        }*/

        //particles[i].weight = 0.1 + (0.8 / (1 + std::exp(-((40 * particles[i].weight) - 10))));

        //particles[i].weight = 0.4 + (0.5*particles[i].weight);

        sum_weight += particles[i].weight;
    }
}

void HL_MCL::next_step(std::array<int, 3> movement, cv::Mat drone_img) {
    auto start = std::chrono::high_resolution_clock::now();
    // MCL::add_gaussian();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Gaussian noise: " << duration.count() << std::endl;

    start = std::chrono::high_resolution_clock::now();
    MCL::motion_update(movement[0], movement[1], movement[2]);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Motion update: " << duration.count() << std::endl;
    
    start = std::chrono::high_resolution_clock::now();
    sensor_update(drone_img, movement[2]);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Sensor update: " << duration.count() << std::endl;
    
    start = std::chrono::high_resolution_clock::now();
    Localization::normalize();
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Normalize: " << duration.count() << std::endl;    

    start = std::chrono::high_resolution_clock::now();
    MCL::resample();
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Resample: " << duration.count() << std::endl;

    Localization::update_stats();
    
    // img = cv::Mat::zeros(m->size().height / 4, m->size().width / 4, CV_8UC3);

    // for (particle p : hl->get_particles()) {
    //     //std::cout << p.weight << " ";        
    //     cv::circle(img, cv::Point(p.x / 4, p.y / 4), 5, cv::Scalar(0, 0, 50+(200.0*p.weight)), cv::FILLED, cv::LINE_AA);
    // }
    
    // cv::GaussianBlur(img, img, cv::Size(9, 9), 0);
    // cv::namedWindow("Hist2", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
    // cv::resizeWindow("Hist2", 600, 600);
    // cv::imshow("Hist2", img);

    
}

