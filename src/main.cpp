#include <iostream>
#include <string>
#include "system.h"
#include <array>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

int main(int argc, char* argv[]) {

    init_logger();
    std::string path;
    std::string flight;
    if (argc > 3) {
        path = argv[1];
        spdlog::info("Path to dataset is: {}", path);
    } else {
        spdlog::error("Incorrect usage. Need ./main <path_to_dataset> <number_of_runs> <maps>");
        return 1;
    }

    spdlog::info("Number of runs: {}", argv[2]);
    int num = std::stoi(argv[2]);

    spdlog::info("Number of maps: {}", argc - 3);
    std::vector<std::string> maps;
    for (int i = 3; i < argc; i++) {
        maps.push_back(argv[i]);
    }
    spdlog::info("Maps used: {}", fmt::join(maps, ", "));

    spdlog::info("Starting the system...");
    int res = run(path, maps, num);
    
    if (res != 0) {
        spdlog::error("System ended with error code: {}", res);
        return 1;
    }

    spdlog::info("System ended successfully.");
    return 0;
}

