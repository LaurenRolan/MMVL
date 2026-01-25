#include "SiamNet.h"

#include <string>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <vector>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"

#include <cmath>

std::vector<torch::jit::IValue> SiamNet::cv_to_model_input(cv::Mat img) {
    torch::NoGradGuard no_grad;
    if (_scale) img = img * 255;
    square_image(img);
    cv::resize(img, img, cv::Size(224, 224));
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    
    at::Tensor tensor = torch::from_blob(img.data, {img.rows, img.cols, 3});
    tensor = tensor.toType(c10::kFloat);
    tensor = tensor.permute({ 2, 0, 1 });
    tensor.unsqueeze_(0);

    if (_normalize) tensor = normalize(tensor);

    std::vector<torch::jit::IValue> img_input = std::vector<torch::jit::IValue>{tensor};
    return img_input;
}

cv::Mat SiamNet::square_image(cv::Mat img) {
    if (img.cols == img.rows) {
        return img;
    }

    int dim = std::min(img.cols, img.rows);
    int center_x = img.cols / 2;
    int center_y = img.rows / 2;

    return cv::Mat(img, cv::Rect(center_x - static_cast<int>(dim/2), center_y - static_cast<int>(dim/2), dim, dim));
}

torch::Tensor SiamNet::normalize(torch::Tensor img) {
    return img.sub(mean).div(std);
}

FCCompare::FCCompare(std::string siam_path, std::string combined_path) {
    try {
        siam_model = torch::jit::load(siam_path);
        siam_model.eval();
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model: " << siam_path << std::endl;
        exit(1);
    }

    try {
        combined_model = torch::jit::load(combined_path);
        combined_model.eval();
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model: " << combined_path << std::endl;
        exit(1);
    }
}

FCCompare::FCCompare(std::string siam_path, std::string combined_path, bool scale, bool normalize) {
    _scale = scale;
    _normalize = normalize;

    try {
        siam_model = torch::jit::load(siam_path);
        siam_model.eval();
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model: " << siam_path << std::endl;
        exit(1);
    }

    try {
        combined_model = torch::jit::load(combined_path);
        combined_model.eval();
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model: " << combined_path << std::endl;
        exit(1);
    }
}

at::Tensor FCCompare::cache_img(cv::Mat img) {
    torch::NoGradGuard no_grad;

    if (_scale) img = img * 255;
    
    
    img = SiamNet::square_image(img);
    cv::resize(img, img, cv::Size(224, 224));
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    at::Tensor tensor = torch::from_blob(img.data, {img.rows, img.cols, 3});
    tensor = tensor.toType(c10::kFloat);
    tensor = tensor.permute({ 2, 0, 1 });
    
    tensor.unsqueeze_(0);
    
    if (_normalize) tensor = normalize(tensor);
    

    std::vector<torch::jit::IValue> img_input = std::vector<torch::jit::IValue>{tensor};
    return siam_model.forward(img_input).toTensor();
}

float FCCompare::compare(cv::Mat drone_img, cv::Mat sat_img) {
    torch::NoGradGuard no_grad;
    if (_scale) drone_img = drone_img * 255;
    square_image(drone_img);
    cv::resize(drone_img, drone_img, cv::Size(224, 224));
    cv::cvtColor(drone_img, drone_img, cv::COLOR_BGR2RGB);

    if (_scale) sat_img = sat_img * 255;
    square_image(sat_img);
    cv::resize(sat_img, sat_img, cv::Size(224, 224));
    cv::cvtColor(sat_img, sat_img, cv::COLOR_BGR2RGB);

    at::Tensor drone_tensor = torch::from_blob(drone_img.data, {drone_img.rows, drone_img.cols, 3});
    at::Tensor sat_tensor = torch::from_blob(sat_img.data, {sat_img.rows, sat_img.cols, 3});

    drone_tensor = drone_tensor.toType(c10::kFloat);
    sat_tensor = sat_tensor.toType(c10::kFloat);

    drone_tensor = drone_tensor.permute({ 2, 0, 1 });
    sat_tensor = sat_tensor.permute({ 2, 0, 1 });

    drone_tensor.unsqueeze_(0);
    sat_tensor.unsqueeze_(0);

    if (_normalize) drone_tensor = normalize(drone_tensor);
    if (_normalize) sat_tensor = normalize(sat_tensor);


    std::vector<torch::jit::IValue> drone_input = std::vector<torch::jit::IValue>{drone_tensor};
    std::vector<torch::jit::IValue> sat_input = std::vector<torch::jit::IValue>{sat_tensor};

    at::Tensor output1 = siam_model.forward(drone_input).toTensor();
    at::Tensor output2 = siam_model.forward(sat_input).toTensor();

    at::Tensor sub_tensor = torch::abs(torch::sub(output1, output2));

    std::vector<torch::jit::IValue> comb_input = std::vector<torch::jit::IValue>{sub_tensor};
    at::Tensor output = combined_model.forward(comb_input).toTensor();

    //std::cout << output.item<float>() << std::endl;

    return output.item<float>();
}

float FCCompare::compare(at::Tensor drone_cache, cv::Mat sat_img) {
    torch::NoGradGuard no_grad;
    //std::vector<torch::jit::IValue> sat_input = cv_to_model_input(sat_img);
    if (_scale) sat_img = sat_img * 255;
    square_image(sat_img);
    cv::resize(sat_img, sat_img, cv::Size(224, 224));
    cv::cvtColor(sat_img, sat_img, cv::COLOR_BGR2RGB);

    at::Tensor sat_tensor = torch::from_blob(sat_img.data, {sat_img.rows, sat_img.cols, 3});

    sat_tensor = sat_tensor.toType(c10::kFloat);

    sat_tensor = sat_tensor.permute({ 2, 0, 1 });

    sat_tensor.unsqueeze_(0);

    if (_normalize) sat_tensor = normalize(sat_tensor);

    std::vector<torch::jit::IValue> sat_input = std::vector<torch::jit::IValue>{sat_tensor};

    at::Tensor output2 = siam_model.forward(sat_input).toTensor();
    
    at::Tensor sub_tensor = torch::cat({drone_cache, output2}, 1);
    // at::Tensor sub_tensor = torch::abs(torch::sub(drone_cache, output2));

    std::vector<torch::jit::IValue> comb_input = std::vector<torch::jit::IValue>{sub_tensor};
    at::Tensor output = combined_model.forward(comb_input).toTensor();

    return output.item<float>();
}

float FCCompare::compare(at::Tensor drone_cache, std::vector<float> sat_cache) {
    torch::NoGradGuard no_grad;
    //std::vector<torch::jit::IValue> sat_input = cv_to_model_input(sat_img);


    at::Tensor sat_tensor = torch::from_blob(sat_cache.data(), {1, static_cast<int64_t>(sat_cache.size())});

    // at::Tensor sub_tensor = torch::abs(torch::sub(drone_cache, sat_tensor));
    at::Tensor sub_tensor = torch::cat({drone_cache, sat_tensor}, 1);


    std::vector<torch::jit::IValue> comb_input = std::vector<torch::jit::IValue>{sub_tensor};
    at::Tensor output = combined_model.forward(comb_input).toTensor();

    return output.item<float>();
}


float FCCompare::compare(at::Tensor drone_cache, std::array<float, 16384> sat_cache) {
    torch::NoGradGuard no_grad;
    at::Tensor sat_tensor = torch::zeros({16384});
    memcpy(sat_tensor.data_ptr<float>(), sat_cache.data(), 16384*sizeof(float));

    // at::Tensor sub_tensor = torch::abs(torch::sub(drone_cache, sat_tensor));


    sat_tensor = sat_tensor.squeeze(-1);
    sat_tensor = sat_tensor.unsqueeze(0);
    // std::cout << sat_tensor.sizes()[0] << ", " << sat_tensor.sizes()[1] << ", " << sat_tensor.sizes()[2] << std::endl;
    // std::cout << drone_cache.sizes()[0] << ", " << drone_cache.sizes()[1] << ", " << drone_cache.sizes()[2] << std::endl;
    
    at::Tensor sub_tensor = torch::cat({drone_cache, sat_tensor}, 1);


    std::vector<torch::jit::IValue> comb_input = std::vector<torch::jit::IValue>{sub_tensor};
    at::Tensor output = combined_model.forward(comb_input).toTensor();

    return output.item<float>();
}

float FCCompare::compare(at::Tensor drone_cache, at::Tensor sat_cache) {
    torch::NoGradGuard no_grad;
    at::Tensor sub_tensor = torch::abs(torch::sub(drone_cache, sat_cache));
    // at::Tensor sub_tensor = torch::cat({drone_cache, sat_cache}, 1);


    std::vector<torch::jit::IValue> comb_input = std::vector<torch::jit::IValue>{sub_tensor};
    at::Tensor output = combined_model.forward(comb_input).toTensor();

    return output.item<float>();
}