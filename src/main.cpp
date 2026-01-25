#include <iostream>
#include <string>
#include "system.h"
#include <array>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

int main(int argc, char* argv[]) {

    std::string path;
    std::string flight;
    if (argc > 3) {
        path = argv[1];
        std::cout << "Path to dataset is: " << path << std::endl;
    } else {
        std::cout << "Incorrect usage. Need ./main <path_to_dataset> <number_of_runs> <maps>" << std::endl;
        return 0;
    }

    std::cout << "Number of runs: " << argv[2] << std::endl;
    int num = std::stoi(argv[2]);

    std::cout << "Number of maps: " << argc - 3 << std::endl;
    std::vector<std::string> maps;
    for (int i = 3; i < argc; i++) {
        maps.push_back(argv[i]);
    }

    int res = run(path, maps, num);
    
    if (res != 0) {
        std::cerr << "System ended unexpectedly" << std::endl;
        return 1;
    }

    return 0;
}

