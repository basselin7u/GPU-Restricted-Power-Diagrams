#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>

#include "Utils.h"

namespace Utils {
  bool loadResFile(std::string const &filename, std::vector<double>& xyzw) {
    std::ifstream in;
    in.open(filename, std::ifstream::in);
    if (in.fail()) return false;
    std::string line;
    int npts = 0;
    bool firstline = true;
    double x,y,z,w;
    while (!in.eof()) {
        std::getline(in, line);
        if (!line.length()) continue;
        std::istringstream iss(line.c_str());
        if (firstline) {
            iss >> npts;
            xyzw.reserve(4*size_t(npts));
            firstline = false;
        } else {
	    iss >> x >> y >> z >> w;
            xyzw.push_back(x);
            xyzw.push_back(y);
            xyzw.push_back(z);
            xyzw.push_back(w);
        }
    }
    assert(int(xyzw.size()) == npts*4);
    in.close();
    return true;
}

}

