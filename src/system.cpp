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

    FCCompare model1 = FCCompare("../../models/aug4Run_resnet34_30norm_siam.pt", "../../models/aug4Run_resnet34_30norm_combined.pt", false, true);
    
    std::vector<HL*> HL_list;
    for (int i = 0; i < map_names.size(); i++){
        Map map = Map(path + "/Maps/" + map_names[i] + ".png", 275);
        HL hl = HL(&map, &model1);
        HL_list.push_back(&hl);
    }

    for (int n = 0; n < runs; n++) {

        MMVL mmvl = MMVL(10000, &map_obj1, &model1, HL_list);

        Drone d = Drone(path + "/Flight/", path + "/traj.txt");

        VO_Motion motion = VO_Motion(path);

        std::ofstream file("result.txt", std::ios::app);

        if (!file.is_open()) {
            std::cerr << "Output file did not open correctly" << std::endl;
            return 1;
        }

        std::ifstream traj_file(path + "/traj.txt");
        if (!traj_file.is_open()) {
            std::cerr << "Cant open the traj file to count the lenght of the run" << std::endl;
            return 1;
        }

        std::string line;
        int lenght = 0;

        while (std::getline(traj_file, line)) {
            lenght++;
        }
   
        cv::waitKey(10);
        int cnt = 1;

        for (int i = 0; i < 1; i++) {
            motion.next_update();    
            cv::Mat drone_img = d.get_image();
            cnt ++;
        }
        
        std::cout << "starting..." << std::endl;
        std::array<int, 3> vo_movement = motion.next_update();
        cv::Mat drone_img = d.get_image();
        mmvl.next_step(vo_movement, drone_img);

        cv::waitKey(10);

        bool done = false;

        while (!done) {
            auto start = std::chrono::high_resolution_clock::now();
            std::array<int, 3> vo_movement = motion.next_update();

            cv::Mat drone_img = d.get_image();
            mmvl.next_step(vo_movement, drone_img);

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

            cv::Mat map_img =  map_obj1.visualize(mmvl.get_particles(), d.get_pos());
            cv::resize(map_img, map_img, cv::Size(1000, 1000));
            map_img.convertTo(map_img, CV_32FC3);
            cv::Mat weight_img = mmvl.interpolation() * 800;
            cv::resize(weight_img, weight_img, cv::Size(1000, 1000));
            cv::cvtColor(weight_img, weight_img, cv::COLOR_GRAY2BGR);

            cv::Mat combined;
            cv::hconcat(map_img, weight_img, combined);
            display_img(combined, "combined", cv::Size(1000, 500));
 
            std::cout << "step: " << cnt << "\t\tTime: " << duration.count() << std::endl;
            std::cout << "Error (blue): " << distance_between({d.get_pos()[0], d.get_pos()[1]}, mmvl.get_average_position()) << std::endl;

            file << distance_between({d.get_pos()[0], d.get_pos()[1]}, mmvl.get_average_position()) << ",";

            cnt ++;
            if (cnt >= lenght) {
                done = true;
            }

            std::cout << std::endl;
            
            cv::waitKey(100);
        }

        file << "\n";
        file.close();
    }

    //video_writer.release();
    cv::destroyAllWindows();
    return 0;
}

