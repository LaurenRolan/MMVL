#ifndef hl_h
#define hl_h

#include <vector>
#include <opencv2/core.hpp>
#include <torch/script.h>
#include "localization.h"

//Pixels between each particle/grid
#define STEP_SIZE 50

class SiamNet;
struct particle;
class Map;

/*
Implementation of histogram localization or grid based localization
*/
class HL: public Localization {
    public:
        HL(Map* map_ptr, SiamNet* model_ptr);

        //Makes the next step in hl
        void next_step(std::array<int, 3> movement, cv::Mat drone_img) override;
        void next_step(at::Tensor drone_cache);

        cv::Mat interpolation();        
        //Gets the amount of particles in x ond y direction
        std::array<int, 2> get_particles_dim();
    private:
        //Cache for all the embeddings
        std::vector<std::array<float, 16384>> cache;

        void make_grid();

        //Makes a sensor update with the cache
        void sensor_update_cache(at::Tensor drone_cache);

        //Normalize the values of particles based on the sum_weight
        void normalize();
};

#endif
