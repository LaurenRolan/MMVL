#ifndef vo_motion_h
#define vo_motion_h

#include <fstream>
#include <array>

#define COMPASS_RANGE 5

/*
Class to simulate the motion module. Could be odometry, but in this case it is from the ground truth
*/
class VO_Motion {
    public:
        VO_Motion(std::string const path);
        std::array<int, 3> next_absolute_update();

        //Reads the next line in the file and calculate the motion
        std::array<int, 3> next_update();
    private:
        std::ifstream file;
        std::ifstream traj_file;

        float rot = 0;
        float scale = 0.5;

        //The deliminator in the ground truth file
        std::string delim = " ";

        std::array<int, 3> prev_update = {0, 0, 0};
        
        std::array<float, 4> read_translation();
        float read_rotation();
};

#endif