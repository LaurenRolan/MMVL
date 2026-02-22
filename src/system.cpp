#include <cstdio>
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <chrono>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#include "system.h"
#include "vo_motion.h"
#include "Drone.h"
#include "SiamNet.h"
#include "utilities.h"
#include "map.h"
#include "localization.h"
#include "mcl.h"
#include "hl.h"
#include "mmvl.h"


#include "cache.h"


int run(std::string path, std::vector<std::string> map_names, int runs) {
 
    Map map_obj1 = Map(path + "/Maps/" + map_names[0] + ".png", 275);

    FCCompare model1 = FCCompare("../models/aug4Run_resnet34_30norm_siam.pt", "../models/aug4Run_resnet34_30norm_combined.pt", false, true);
    
    std::vector<std::unique_ptr<Map>> maps;
    std::vector<std::unique_ptr<HL>> HL_storage;
    std::vector<HL*> HL_list;
    
    for (size_t i = 0; i < map_names.size(); i++) {
        maps.push_back(std::make_unique<Map>(
            path + "/Maps/" + map_names[i] + ".png", 275
        ));

        HL_storage.push_back(std::make_unique<HL>(
            maps.back().get(), &model1
        ));

        HL_list.push_back(HL_storage.back().get());
    }

    for (int n = 0; n < runs; n++) {

        MMVL mmvl = MMVL(10000, &map_obj1, &model1, HL_list);

        Drone d = Drone(path + "/Flight/", path + "/traj.txt");

        VO_Motion motion = VO_Motion(path);

        std::ofstream file("result.txt", std::ios::app);

        if (!file.is_open()) {
            spdlog::error("Output file {} did not open correctly", "result.txt");
            return 1;
        }

        std::ifstream traj_file(path + "/traj.txt");
        if (!traj_file.is_open()) {
            spdlog::error("Cant open the traj file to count the lenght of the run");
            return 1;
        }

        std::string line;
        int lenght = 0;

        while (std::getline(traj_file, line)) {
            lenght++;
        }

        int cnt = 1;

        for (int i = 0; i < 1; i++) {
            motion.next_update();    
            cv::Mat drone_img = d.get_image();
            cnt ++;
        }
        
        spdlog::info("Starting run {} of {}", n + 1, runs);
        std::array<int, 3> vo_movement = motion.next_update();
        spdlog::debug("Initial VO movement: [{}, {}, {}]", vo_movement[0], vo_movement[1], vo_movement[2]);
        cv::Mat drone_img = d.get_image();
        spdlog::debug("Initial drone image size: {}x{}", drone_img.cols, drone_img.rows);
        mmvl.next_step(vo_movement, drone_img);
        spdlog::debug("Initial average position: [{:d}, {:d}]", mmvl.get_average_position()[0], mmvl.get_average_position()[1]);

        bool done = false;

        while (!done) {
            auto start = std::chrono::high_resolution_clock::now();
            std::array<int, 3> vo_movement = motion.next_update();
            spdlog::debug("VO movement: [{}, {}, {}]", vo_movement[0], vo_movement[1], vo_movement[2]);

            cv::Mat drone_img = d.get_image();
            mmvl.next_step(vo_movement, drone_img);
            spdlog::debug("MMVL step completed");

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

            spdlog::info("Step: {}, Time: {} ms, Error (blue): {}", cnt, duration.count(), distance_between({d.get_pos()[0], d.get_pos()[1]}, mmvl.get_average_position()));

            file << distance_between({d.get_pos()[0], d.get_pos()[1]}, mmvl.get_average_position()) << ",";

            cnt ++;
            if (cnt >= lenght) {
                done = true;
            }
        }

        file << "\n";
        file.close();
    }

    //video_writer.release();
    cv::destroyAllWindows();
    return 0;
}

