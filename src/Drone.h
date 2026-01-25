#ifndef Drone_h
#define Drone_h

#include "opencv2/core.hpp"
#include <fstream>

/*
Class to simulate the data coming from the drone.
Reads both position and image from files
*/
class Drone {
    public:
        Drone(std::string path, std::string truth_path);
        
        //Updates the next image and position, returns this image
        cv::Mat get_image();

        //Gets the position of the drone based on the ground truth
        std::array<int, 3> get_pos();
    private:
        std::string _path;
        int idx = 0;

        //The deliminator in the ground truth file
        std::string delim = "\t";
        std::ifstream file;
        std::array<int, 3> pos = {0, 0, 0};
        
        //Update the position of the drone
        void read_next_line();

        //Makes the drone image square
        cv::Mat square_image(cv::Mat img);
        
};

#endif