#ifndef cache_h
#define cache_h

#include <vector>
#include <string>

class Cache {
    public:
        Cache(std::string path, int map_width, int map_height);
        std::vector<float> get_embedding(int x, int y);
    private:
        std::vector<std::vector<std::vector<float>>> cache;
};

#endif 
