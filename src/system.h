#ifndef system_h
#define system_h

#include <string>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

int run(std::string path, std::vector<std::string> map_names, int runs);

#endif