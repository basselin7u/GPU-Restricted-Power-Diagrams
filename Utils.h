#pragma once

#include <functional>
#include <map>
#include <string>
#include <vector>

#include "cvoro_config.h"

namespace Utils {
    bool loadSeedsFile(const char* filename, std::vector<real>& xyzw, bool is4DFile=false, bool normalize=true);
    bool readBinarySeedsFile(const char* filename, std::vector<real>& xyzw);
    bool saveBinarySeedsFile(const char* filename, std::vector<real> const &xyzw);

    void dropXYZFile(std::vector<real> const& pts, bool pts4D, const char *filename, std::function<bool(size_t index)> validate);
    void dropXYZGeogram(std::vector<real> const& pts, bool pts4D, const char *filename, std::function<bool(size_t index)> validate,
                        std::map<char const *, std::function<real(size_t index)> > const &attrMapValue);
    bool readXYZGeogram(const char *filename, int &numVertices, std::map<std::string, std::vector<real> > &vertexData);
}
