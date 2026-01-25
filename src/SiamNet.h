#ifndef siamnet_h
#define siamnet_h

#include <cmath>
#include <torch/script.h>
#include <vector>
#include "opencv2/core.hpp"



/*
Abstract class to define how a model, with both caching and comparing must look
*/
class SiamNet {
    public:
        virtual float compare(cv::Mat drone_img, cv::Mat sat_img) = 0;
        virtual float compare(at::Tensor drone_cache, cv::Mat sat_img) = 0;
        virtual float compare(at::Tensor drone_cache, std::vector<float> sat_cache) = 0;
        virtual float compare(at::Tensor drone_cache, at::Tensor sat_cache) = 0;
        virtual float compare(at::Tensor drone_cache, std::array<float, 16384> sat_cache) = 0; //16384
        virtual at::Tensor cache_img(cv::Mat img) = 0;

        
    protected:
        cv::Mat square_image(cv::Mat img);
        std::vector<torch::jit::IValue> cv_to_model_input(cv::Mat img);
        torch::Tensor mean = torch::tensor({0.4521, 0.4413, 0.4286}).unsqueeze(1).unsqueeze(2);
        torch::Tensor std = torch::tensor({0.2333, 0.2025, 0.1937}).unsqueeze(1).unsqueeze(2);
        torch::Tensor normalize(torch::Tensor img);
        bool _scale = false;
        bool _normalize = false;
};

/*
Models using FC layers to compare can use this
*/
class FCCompare: public SiamNet {
    public:
        FCCompare(std::string siam_path, std::string combined_path);
        FCCompare(std::string siam_path, std::string combined_path, bool scale, bool normalize);
        float compare(cv::Mat drone_img, cv::Mat sat_img) override;
        float compare(at::Tensor drone_cache, cv::Mat sat_img) override;
        float compare(at::Tensor drone_cache, std::vector<float> sat_cache) override;
        float compare(at::Tensor drone_cache, at::Tensor sat_cache) override;
        float compare(at::Tensor drone_cache, std::array<float, 16384> sat_cache);
        at::Tensor cache_img(cv::Mat img) override;
    private:
        //Embedding model
        torch::jit::script::Module siam_model;

        //Comparing model 
        torch::jit::script::Module combined_model;
};

#endif
