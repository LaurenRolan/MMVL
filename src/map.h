#ifndef map_h
#define map_h

#include <opencv2/core.hpp>
#include <string>
#include <vector>

#include "particle.h"

/*
Class to contain the sattelite image and methods regarding this
*/
class Map {
    public:
        Map(std::string path, int patch_size);

        //Gets a patch from the image with the PATCH_SIZE as the size
        cv::Mat get_patch(int x, int y);

        //Draws particles and weight on a black image
        cv::Mat visualize_particles(std::vector<particle> particles, float sum);

        //Draws particles and the stats on a black image
        cv::Mat visualize_particles(std::vector<particle> particles,
                                    float sum,
                                    std::array<int, 2> best_particle, 
                                    std::array<int, 2> average_particle, 
                                    std::array<int, 2> average_weighted_particle);
        
        //Draws the particles and drone on the map
        cv::Mat visualize(std::vector<particle> particles, std::array<int, 3> drone);
        
        //Gets the size of the map
        cv::Size size();
    private:
        cv::Mat map;
        int _patch_size;
};

#endif
