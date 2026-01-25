#ifndef hl_mcl_h
#define hl_mcl_h

#include "mcl.h"
#include <array>

class HL;

class HL_MCL: public MCL {
    public:
        HL_MCL(int num_particles, Map* map_ptr, SiamNet* model_ptr, HL* hl_ptr);
        void next_step(std::array<int, 3> movement, cv::Mat drone_img);
        cv::Mat interpolation();
    private:
        HL* hl;
        
        void sensor_update(cv::Mat drone_img, int rot);

};

#endif


