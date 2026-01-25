#ifndef mmvl_h
#define mmvl_h

#include "mcl.h"
#include <array>

class HL;

class MMVL: public MCL {
    public:
        MMVL(int num_particles, Map* map_ptr, SiamNet* model_ptr, std::vector<HL*> HL_list);
        void next_step(std::array<int, 3> movement, cv::Mat drone_img);
        cv::Mat interpolation();

    private:
        std::vector<HL*> HL_list;
        int number_maps;
                
        void sensor_update(cv::Mat drone_img, int rot);
};

#endif