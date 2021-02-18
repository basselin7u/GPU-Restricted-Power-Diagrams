#pragma once

#include <functional>
#include <map>
#include <string>
#include <vector>

namespace Utils {
    bool loadSeedsFile(const char* filename, std::vector<float>& xyzw, bool is4DFile=false, bool normalize=true);
    bool readBinarySeedsFile(const char* filename, std::vector<float>& xyzw);
    bool saveBinarySeedsFile(const char* filename, std::vector<float> const &xyzw);

    void dropXYZFile(std::vector<float> const& pts, bool pts4D, const char *filename, std::function<bool(size_t index)> validate);
    void dropXYZGeogram(std::vector<float> const& pts, bool pts4D, const char *filename, std::function<bool(size_t index)> validate,
                        std::map<char const *, std::function<float(size_t index)> > const &attrMapValue);
    bool readXYZGeogram(const char *filename, int &numVertices, std::map<std::string, std::vector<float> > &vertexData);
}
