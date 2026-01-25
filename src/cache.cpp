#include "cache.h"
#include <torch/script.h>
#include <vector>
#include <string>

Cache::Cache(std::string path, int map_width, int map_height) {
    std::ifstream file(path, std::ios::binary);

    if (!file.is_open()) {
        std::cerr << "Could not open caching file at " << path << std::endl;
        exit(1);
    }

    std::vector<std::vector<std::vector<float>>> c((map_width / 5) - 400, std::vector<std::vector<float>>((map_height / 5) - 400, std::vector<float>(512)));
    cache = c;

    for (int i = 0; i < (map_width / 5) - 400; i++) {
        for (int k = 0; k < (map_height / 5) - 400; k++) {
            file.read(reinterpret_cast<char*>(cache[i][k].data()), 512*sizeof(float));
        }
    }

    std::cout << "Caching file is done loading. The cache is in the shape " << cache.size() << ", " << cache[0].size() << ", " << cache[0][0].size() << std::endl;
}

std::vector<float> Cache::get_embedding(int x, int y) {
    std::cout << (x-200) / 5 << ", " << (y-200) / 5 << std::endl;
    return cache[(x-200) / 5][(y-200) / 5];
}