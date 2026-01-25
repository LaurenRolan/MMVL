#ifndef mcl_h
#define mcl_h

#include <vector>
#include <opencv2/core.hpp>

#include "localization.h"

class SiamNet;
struct particle;
class Map;
class Cache;

/*
Impletmentation of Monte Carlo Localization
*/
class MCL: public Localization {
    public:
        MCL(int num_particles, Map* map_ptr, SiamNet* model_ptr);

        //Makes the next step in mcl
        void next_step(std::array<int, 3> movement, cv::Mat drone_img) override;
    protected:
        //Resample the particles
        void resample();

        //Update the position of the particles
        void motion_update(int x, int y);

        //Update the position of the particles
        void motion_update(int x, int y, int rot);

        //Adds noise to the particles
        void add_gaussian();
        
};

#endif
