#ifndef localization_h
#define localization_h

#include <vector>
#include <opencv2/core.hpp>
#include <torch/script.h>

class SiamNet;
struct particle;
class Map;
class Cache;

/*
Abstract class that contain some common functions for localization 
and defines what public localization methods must have
*/
class Localization {
    public:
        float max_weight = 0;
        float sum_weight = 0;

        //Methods to get stats about the system
        std::array<int, 2> get_best_particle();
        std::array<int, 2> get_average_position();
        std::array<int, 2> get_average_weighted_position();

        std::vector<particle> get_particles();

        //Calls for the next step in the implemented localization method
        virtual void next_step(std::array<int, 3> movement, cv::Mat drone_img) = 0;
        

    protected:
        int num_p;
        cv::Size map_size;
        std::vector<particle> particles;
        SiamNet* model;
        Map* m;

        //Arrays to store stats
        std::array<int, 2> average_position = {0, 0};
        std::array<int, 2> average_weighted_position = {0, 0};
        std::array<int, 2> best_position = {0, 0};
        
        void drone_embedding(cv::Mat drone_img, int rot);

        //Makes a measurment of the environment and gives the particles a weight
        void sensor_update(cv::Mat drone_img, int rot);
        
        //Calculates and updates all of the stats 
        void update_stats(); 
        
        //Normalize the weights of the particles
        void normalize(); 
};

#endif
