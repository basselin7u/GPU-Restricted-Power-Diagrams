#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>

#include "Utils.h"

namespace Utils {
static void getBoundingBox(const std::vector<real>& xyzw, real& xmin, real& ymin, real& zmin, real& xmax, real& ymax, real& zmax) {
    size_t nb_v = xyzw.size()/4;
    xmin = xmax = xyzw[0];
    ymin = ymax = xyzw[1];
    zmin = zmax = xyzw[2];
    for(size_t i=1; i<nb_v; ++i) {
        xmin = fmin(xmin, xyzw[4*i]);
        ymin = fmin(ymin, xyzw[4*i+1]);
        zmin = fmin(zmin, xyzw[4*i+2]);
        xmax =fmax(xmax, xyzw[4*i]);
        ymax = fmax(ymax, xyzw[4*i+1]);
        zmax = fmax(zmax, xyzw[4*i+2]);
    }
    real d = xmax-xmin;
    d = fmax(d, ymax-ymin);
    d = fmax(d, zmax-zmin);
    d = 0.001f*d;
    xmin -= d;
    ymin -= d;
    zmin -= d;
    xmax += d;
    ymax += d;
    zmax += d;
}

 bool loadSeedsFile(const char* filename, std::vector<real>& xyzw, bool is4DFile, bool normalize) {
    std::ifstream in;
    in.open(filename, std::ifstream::in);
    if (in.fail()) return false;
    std::string line;
    int npts = 0;
    bool firstline = true;
    real x,y,z,w=0;
    while (!in.eof()) {
        std::getline(in, line);
        if (!line.length()) continue;
        std::istringstream iss(line.c_str());
        if (firstline) {
            iss >> npts;
            xyzw.reserve(4*size_t(npts));
            firstline = false;
        } else {
            iss >> x >> y >> z;
            if (is4DFile) iss >> w;
            xyzw.push_back(x);
            xyzw.push_back(y);
            xyzw.push_back(z);
            xyzw.push_back(w);
        }
    }
    assert(int(xyzw.size()) == npts*4);
    in.close();

    if (normalize) { // normalize point cloud between [0,1000]^3
        real xmin,ymin,zmin,xmax,ymax,zmax;
        getBoundingBox(xyzw, xmin, ymin, zmin, xmax, ymax, zmax);

        real maxside = fmax(fmax(xmax-xmin, ymax-ymin), zmax-zmin);
#pragma omp parallel for
        for (int i=0; i<int(xyzw.size())/4; i++) {
            xyzw[size_t(i)*4+0] = 1000.f*(xyzw[size_t(i)*4+0]-xmin)/maxside;
            xyzw[size_t(i)*4+1] = 1000.f*(xyzw[size_t(i)*4+1]-ymin)/maxside;
            xyzw[size_t(i)*4+2] = 1000.f*(xyzw[size_t(i)*4+2]-zmin)/maxside;
        }
        getBoundingBox(xyzw, xmin, ymin, zmin, xmax, ymax, zmax);
        std::cerr << "loadSeedsFile[test_voronoi.cpp]: bbox=[" << xmin << ":" << xmax << "], [" << ymin << ":" << ymax << "], [" << zmin << ":" << zmax << "]" << std::endl;
    }
    return true;
}

bool readBinarySeedsFile(const char* filename, std::vector<real>& xyzw) {
    std::ifstream file(filename, std::ifstream::binary);
    if (!file.is_open()) {
        std::cerr << "readBinarySeedsFile[test_voronoi.cpp]: can not open " << filename << "\n";
        return false;
    }
    size_t N;
    file.read(reinterpret_cast<char *>(&N), sizeof(N));
    if (N==0) return false;
    xyzw.resize(4*N);
    file.read(reinterpret_cast<char *>(xyzw.data()), long(4*sizeof(real)*N));
    if (!file.good()) {
        std::cerr << "readBinarySeedsFile[test_voronoi.cpp]: problem when reading " << filename << "\n";
        return false;
    }
    return true;
}

bool saveBinarySeedsFile(const char* filename, std::vector<real> const &xyzw) {
    std::ofstream file(filename, std::ifstream::trunc | std::ifstream::binary);
    if (!file.is_open()) {
        std::cerr << "saveBinarySeedsFile[test_voronoi.cpp]: can not create " << filename << "\n";
        return false;
    }
    size_t N=xyzw.size()/4;
    file.write(reinterpret_cast<char *>(&N), sizeof(size_t));
    file.write(reinterpret_cast<char const *>(xyzw.data()), long(4*sizeof(real)*N));
    if (!file.good()) {
        std::cerr << "saveBinarySeedsFile[test_voronoi.cpp]: problem when saving " << filename << "\n";
        return false;
    }
    return true;
}

void dropXYZFile(std::vector<real> const &pts, bool pts4D, const char *filename, std::function<bool(size_t index)> validate) {
    std::fstream file;
    file.open(filename, std::ios_base::out);
    // FOR(i, pts.size() / 4) std::cerr << pts[4 * i + 3]<<" \n";
    int k = 0;
    for (size_t i = 0; i < pts.size() / 4; i++) if (validate(i)) k++;
    file << k << std::endl;
    for (size_t i = 0; i < pts.size() / 4; i++){
        if (!validate(i)) continue;
        if (pts4D) {
            for (size_t j = 0; j < 3; j++) {
                if (pts[4 * i + j] < 0 || pts[4 * i + j]>1000 * pts[4 * i + 3]) {
                    std::cerr << "dropXYZFile[test_voronoi.cpp]: POINTS=" << pts[4 * i] / pts[4 * i + 3] << "  " << pts[4 * i + 1] / pts[4 * i + 3] << "  " << pts[4 * i + 2] / pts[4 * i + 3] << ":" << pts[4 * i + 3] << std::endl;
                    break;
                }
            }
            file << pts[4 * i] / pts[4 * i + 3] << "  " << pts[4 * i + 1] / pts[4 * i + 3] << "  " << pts[4 * i + 2] / pts[4 * i + 3] << std::endl;
        }
        else {
            for (size_t j = 0; j < 3; j++) {
                if (pts[4 * i + j] < 0 || pts[4 * i + j]>1000) {
                    std::cerr << "dropXYZFile[test_voronoi.cpp]: POINTS=" << pts[4 * i] << "  " << pts[4 * i + 1] << "  " << pts[4 * i + 2] << std::endl;
                    break;
                }
            }
            file << pts[4 * i] << "  " << pts[4 * i + 1] << "  " << pts[4 * i + 2] << std::endl;
        }
    }
    file.close();
}


#define ENDOFLINE "\n" 
void dropXYZGeogram(std::vector<real> const &pts, bool pts4D, const char *filename, std::function<bool(size_t index)> validate,
                           std::map<char const *, std::function<real(size_t index)> > const &attrMapValue) {
    int k = 0;
    for (size_t i = 0; i < pts.size() / 4; i++) if (validate(i)) k++;
    std::fstream file;
    file.open(filename, std::ios_base::out | std::ios_base::binary);
    file << "[HEAD]" << ENDOFLINE << "\"GEOGRAM\"" << ENDOFLINE << "\"1.0\"" << ENDOFLINE;
    file << "[ATTS]\n\"GEO::Mesh::vertices\"" << ENDOFLINE;
    file << k<< ENDOFLINE;

    file << "[ATTR]" << ENDOFLINE << "\"GEO::Mesh::vertices\"" << ENDOFLINE << "\"point\"" << ENDOFLINE << "\"double\"" << ENDOFLINE << "8" << ENDOFLINE << "3" << ENDOFLINE;
    for (size_t i = 0; i < pts.size() / 4; i++) {
        if (!validate(i)) continue;
        if (pts4D)
            for (size_t d=0 ; d < 3; ++d) file << pts[4 * i + d] / pts[4 * i + 3] << ENDOFLINE;
        else 
            for (size_t d=0 ; d < 3; ++d) file << pts[4 * i + d] << ENDOFLINE;
   }
    for (auto &it : attrMapValue) {
        file << "[ATTR]" << ENDOFLINE << "\"GEO::Mesh::vertices\"" << ENDOFLINE << "\"" << it.first << "\"" << ENDOFLINE << "\"double\"" << ENDOFLINE << "8" << ENDOFLINE << "1" << ENDOFLINE;
        for (size_t i = 0; i < pts.size() / 4; i++) if (validate(i)) file << it.second(i) << ENDOFLINE;
    }
    file.close();
}

bool readXYZGeogram(const char *filename, int &numVertices, std::map<std::string, std::vector<real> > &vertexData)
{
    std::ifstream in;
    in.open(filename, std::ifstream::in);
    if (in.fail()) {
        std::cerr << "Utils::readXYZGeogram: can not open " << filename << "\n";
        return false;
    }
    std::string line;
    char const *(expectedHeader)[]=
    {
     "[HEAD]",  "\"GEOGRAM\"", "\"1.0\"", "[ATTS]",  "\"GEO::Mesh::vertices\""
    };
    for (auto const &wh : expectedHeader) {
        std::string str;
        in >> str;
        if (str!=wh) {
            std::cerr << "Utils::readXYZGeogram[" << filename << "]: " << str << "!=" <<wh << "\n";
            return false;
        }
    }
    in >> numVertices;
    if (!in.good() || numVertices<=0) {
        std::cerr << "Utils::readXYZGeogram[" << filename << "]: can not read the number of vertices \n";
        return false;
    }
    while(!in.eof()) {
        std::string name;
        int numData;
        for (int wh=0; wh < 6; ++wh) {
            std::string str;
            int val;
            if (wh<4) {
                in >> str;
                if (str.empty() && in.eof() && wh==0)
                    return true;
            }
            else
                in >> val;
            if (!in.good()) {
                std::cerr << "Utils::readXYZGeogram[" << filename << "]: oops something is bad when looking for ATTR \n";
                return false;
            }
            char const *(expected[])={"[ATTR]", "\"GEO::Mesh::vertices\"", nullptr, "\"double\"", nullptr, nullptr};
            if (expected[wh] && str!=expected[wh]) {
                std::cerr << "Utils::readXYZGeogram[" << filename << "]: unexpected tag=" << str << "!=" << expected[wh] << "\n";
                return false;
            }
            else if (wh==2)
                name=str.substr(1,str.length()-2); // remove the ""
            else if (wh==5)
                numData=val;
            else if (wh==4 && val!=8) {
                std::cerr << "Utils::readXYZGeogram[" << filename << "]: unexpected data size=" << val << "\n";
                return false;
            }
        }
        if (vertexData.find(name)!=vertexData.end()) {
            std::cerr << "Utils::readXYZGeogram[" << filename << "]: oops, a attribute with name: " << name << "already exists\n";
            return false;
        }
        vertexData[name]=std::vector<real>();
        auto &values=vertexData.find(name)->second;
        values.resize(size_t(numData*numVertices));
        for (auto &v : values) {
            in >> v;
            if (!in.good()) {
                std::cerr << "Utils::readXYZGeogram[" << filename << "]: can not read some values \n";
                return false;
            }
        }
    }
    return true;
}

}

