#ifndef utilities_h
#define utilities_h

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <string>
#include <cmath>

/*
Contains some methods used in different files.
*/


//Generate a float from a gaussion distribution
float get_gaussian_random_number(double mean, double var);

//Generate a random float between 0-1
float random_number();

//Generate a random float between min and max
float random_number(float min, float max);

//Display a image using cv::imshow
void display_img(cv::Mat img, std::string name, cv::Size size = cv::Size(600, 600));

//Calculates the euclidian distance between two points
int distance_between(std::array<int, 2> pos_a, std::array<int, 2> pos_b);

#endif
