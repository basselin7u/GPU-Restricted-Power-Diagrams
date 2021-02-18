#include <algorithm>
#include <array>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>

#include "basic.h"

#include "Grid.h"

inline double det2x2(double a11, double a12, double a21, double a22) {
    return a11*a22 - a12*a21;
}
inline double det3x3(double a11, double a12, double a13, double a21, double a22, double a23, double a31, double a32, double a33) {
    return a11*det2x2(a22, a23, a32, a33) - a21*det2x2(a12, a13, a32, a33) + a31*det2x2(a12, a13, a22, a23);
}
inline double getTetVolume(std::vector<real> const pts, size_t i0, size_t i1, size_t i2) {
    return -det3x3((double) pts[4*i0], (double) pts[4*i0+1], (double) pts[4*i0+2],
                   (double) pts[4*i1], (double) pts[4*i1+1], (double) pts[4*i1+2],
                   (double) pts[4*i2], (double) pts[4*i2+1], (double) pts[4*i2+2])/6;
}
inline double getTriArea2(std::vector<real> const pts, size_t i0, size_t i1, size_t i2) {
    double i1i0[]={double(pts[4*i1]-pts[4*i0]), double(pts[4*i1+1]-pts[4*i0+1]), double(pts[4*i1+2]-pts[4*i0+2])};
    double i2i0[]={double(pts[4*i2]-pts[4*i0]), double(pts[4*i2+1]-pts[4*i0+1]), double(pts[4*i2+2]-pts[4*i0+2])};
    double cross[]={i1i0[1]*i2i0[2]-i1i0[2]*i2i0[1], i1i0[2]*i2i0[0]-i1i0[0]*i2i0[2], i1i0[0]*i2i0[1]-i1i0[1]*i2i0[0]};
    return (cross[0]*cross[0]+cross[1]*cross[1]+cross[2]*cross[2])/4;
}

double Grid::getVolume() const {
    double volume=0;
    size_t numTriangles=m_triangles.size()/3;
    for (size_t i=0; i<numTriangles; ++i) volume += double(getTetVolume(m_points,size_t(m_triangles[3*i]),size_t(m_triangles[3*i+1]),size_t(m_triangles[3*i+2])));
    return volume;
}

inline std::array<double,4> getBarycenter(std::vector<real> const &pts, size_t i0, size_t i1, size_t i2) {
    double vol_4=getTetVolume(pts,i0,i1,i2)/4;
    return {vol_4*((double)pts[4*i0]+(double)pts[4*i1]+(double)pts[4*i2]),
            vol_4*((double)pts[4*i0+1]+(double)pts[4*i1+1]+(double)pts[4*i2+1]),
            vol_4*((double)pts[4*i0+2]+(double)pts[4*i1+2]+(double)pts[4*i2+2]),
            4*vol_4 };
}

inline std::array<double,4> getBarycenter(std::vector<real> const &pts, std::vector<int> const &triangles) {
    std::array<double,4> bary={0,0,0,0};
    size_t numTriangles=triangles.size()/3;
    for (size_t i=0; i<numTriangles; ++i) {
        auto res=getBarycenter(pts,size_t(triangles[3*i]),size_t(triangles[3*i+1]),size_t(triangles[3*i+2]));
        for (size_t j=0; j<4; ++j) bary[j]+=res[j];
    }
    return bary;
}

void Grid::swapTrianglesOrientation() {
    size_t numTriangles=m_triangles.size()/3;
    for (size_t i=0; i<numTriangles; ++i) std::swap(m_triangles[3*i],m_triangles[3*i+1]);
}

bool Grid::valid() const
{
    size_t numVoxels=size_t(m_size*m_size*m_size);
    if (m_inDomain.size()!=numVoxels) {
        if (m_debug) std::cerr << "Grid::valid: unexpected inDomain size: " << m_inDomain.size() << "!=" << numVoxels << "\n";
        return false;
    }
    for (auto c : m_inDomain) {
        if (c & 0xfc) {
            if (m_debug) std::cerr << "Grid::valid: unexpected inDomain value: " << int(c) << "\n";
            return false;
        }
    }
    size_t numPoints=m_points.size();
    if (numPoints%4) {
        if (m_debug) std::cerr << "Grid::valid: unexpected m_points size: " << numPoints << "\n";
        return false;
    }
    numPoints/=4;
    size_t numTriangles=m_triangles.size();
    if (numTriangles%3) {
        if (m_debug) std::cerr << "Grid::valid: unexpected m_triangles size: " << numTriangles << "\n";
        return false;
    }
    numTriangles/=3;
    for (auto p : m_triangles) {
        if (p<0 || p >= int(numPoints)) { 
            if (m_debug) std::cerr << "Grid::valid: invalid point id in m_triangles : " << p << "\n";
            return false;
        }
    }
    size_t numIndirections=m_trianglesList.size();
    for (auto t : m_trianglesList) {
        if (t < 0 || t >= int(numTriangles)) {
            if (m_debug) std::cerr << "Grid::valid: invalid triangle id in m_trianglesList : " << t << "\n";
            return false;
        }
    }
    size_t numOffsets=m_offsets.size();
    if (numOffsets<=numVoxels) {
        if (m_debug) std::cerr << "Grid::valid: unexpected offsets size: " << numOffsets << "<=" << numVoxels << "\n";
        return false;
    }
    for (auto o : m_offsets) {
        if (o < 0 || o > int(numIndirections)) {
            if (m_debug) std::cerr << "Grid::valid: invalid offset in m_offsets : " << o << "\n";
            return false;
        }
    }

    // check orientation
    double volume=getVolume();
    if (getVolume()<=0) {
        if (m_debug) std::cerr << "Grid::valid: bad volume: " << volume << std::endl;
        return false;
    }

    // check that no triangles are empty
    real const epsilon=0.01f;
    for (size_t i=0; i<numTriangles; ++i) {
        real area2=real(getTriArea2(m_points, size_t(m_triangles[3*i]), size_t(m_triangles[3*i+1]), size_t(m_triangles[3*i+2])));
        if (area2<epsilon*epsilon) {
            if (m_debug) std::cerr << "Grid::valid: bad squared area for triangle " << i << ": " << area2 << std::endl;
            return false;
        }
    }

    // check that for each e_{ij}, there exist a unique e_{ji}
    std::set<std::pair<int,int> > edges;
    for (size_t i=0; i<numTriangles; ++i) {
        for (size_t j=0; j<3; ++j) {
            auto edge=std::make_pair(m_triangles[3*i+j],m_triangles[3*i+(j+1)%3]);
            if (edges.find(edge)!=edges.end()) {
                if (m_debug) std::cerr << "Grid::valid: edge " << m_triangles[3*i+j] << "x" << m_triangles[3*i+(j+1)%3] << "already find\n";
                return false;
            }
            edges.insert(edge);
        }
    }
    for (auto const &e : edges) {
        auto opp=std::make_pair(e.second, e.first);
        if (edges.find(opp)==edges.end()) {
            if (m_debug) std::cerr << "Grid::valid: can not find edge " << e.second << "x" << e.first << "\n";
            return false;
        }
    }

    // check that no cell has 8 inDomain data
    std::set<size_t> inDomainBorders;
    for (size_t i=0; i<m_inDomain.size(); ++i) {
        if (m_inDomain[i]==1) inDomainBorders.insert(i);
    }
    for (auto p : inDomainBorders) {
        int find=0;
        for (int x=0; x<=1; x+=1) {
            for (int y=0; y<=1; y+=1) {
                for (int z=0; z<=1; z+=1) {
                    auto it=inDomainBorders.find(x+y*m_size+z*m_size*m_size);
                    if (it!=inDomainBorders.end()) ++find;
                }
            }
        }
        if (find!=8) continue;
        if ((p%m_size)==m_size-1) continue; // assume x+1 is outside
        if (((p/m_size)%m_size)==m_size-1) continue; // assume y+1 is outside
        if (((p/m_size/m_size)%m_size)==m_size-1) continue; // assume z+1 is outside
        if (m_debug) std::cerr << "Grid::valid: cell (" << (p%m_size) << "," << ((p/m_size)%m_size) << "," << ((p/m_size/m_size)%m_size) << "): contained only border vertices\n";
        return false;
    }

    // check that the triangles list are ordered
    for (size_t o=0; o+1<numOffsets; ++o) {
        if (m_offsets[o]>m_offsets[o+1]) {
            if (m_debug) std::cerr << "Grid::valid: bad ordering for offset " << o << "\n";
            return false;
        }
        for (size_t t=size_t(m_offsets[o]); t+1<size_t(m_offsets[o+1]); ++t) {
            if (m_trianglesList[t]>=m_trianglesList[t+1]) {
                if (m_debug) std::cerr << "Grid::valid: triangles list is not ordered\n";
                return false;
            }
        }
    }
    return true;
}

bool Grid::load(char const * filename)
{
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Grid::load: can not open " << filename << "\n";
        return false;
    }
    file >> m_size;
    m_voxelSize = 1000 / real(m_size);
    size_t numVoxels=size_t(m_size*m_size*m_size);
    m_inDomain.resize(numVoxels);
    for (size_t i=0; i<numVoxels; ++i) {
        file >> m_inDomain[i];
        if (m_inDomain[i]& 0xfc) {
            std::cerr << "Grid::load[" << filename << "]: unexpected value in domain\n";
            return false;
        }
    }
    if (!file.good()) {
        std::cerr << "Grid::load[" << filename << "]: can not read the in domain vector\n";
        return false;
    }
    int numPoints;
    file >> numPoints;
    m_points.resize(4*size_t(numPoints));
    for (size_t i=0; i<size_t(numPoints); ++i) {
        real coord;
        for(size_t j=0; j<3; ++j) { file >> coord; m_points[4*i+j]=coord; }
        m_points[4*i+3]=0;
    }
    if (!file.good()) {
        std::cerr << "Grid::load[" << filename << "]: can not read the points\n";
        return false;
    }
    int numTriangles;
    file >> numTriangles;
    m_triangles.resize(size_t(numTriangles));
    for (auto &t : m_triangles) {
        file >>t;
        if (t<0 || t>= numPoints) {
            std::cerr << "Grid::load[" << filename << "]: bad triangle point " << t << "\n";
            return false;
        }
    }
    if (!file.good()) {
        std::cerr << "Grid::load[" << filename << "]: can not read the triplets\n";
        return false;
    }
    numTriangles/=3;

    int numIndirections;
    file >> numIndirections;
    m_trianglesList.resize(size_t(numIndirections));
    for (auto &ind : m_trianglesList) {
        file >> ind;
        if (ind < 0 || ind >= numTriangles) {
            std::cerr << "Grid::load[" << filename << "]: bad triangle id " <<  ind << "\n";
            return false;
        }
    }
    if (!file.good()) {
        std::cerr << "Grid::load[" << filename << "]: can not read the triangles indirections\n";
        return false;
    }

    int tmp;
    file >> tmp;
    if (tmp<=int(numVoxels)) {
        std::cerr << "Grid::load[" << filename << "]: the number of offsets seems bad\n";
        return false;
    }
    m_offsets.resize(size_t(tmp));
    for (auto &off : m_offsets) {
        file >> off;
        if (off < 0 || off > numIndirections) {
            std::cerr << "Grid::load[" << filename << "]: bad offset " <<  off << "\n";
            return false;
        }
    }
    if (!file.good()) {
        std::cerr << "Grid::load[" << filename << "]: can not read the offsets\n";
        return false;
    }

    double volume=getVolume();
    if (volume<0) {
        if (m_debug) std::cerr << "Grid::load[" << filename << "]: swap orientation\n";
        swapTrianglesOrientation();
        volume *= -1;
    }
    if (m_debug) std::cerr << "Grid::load: volume=" << std::setprecision(20) << volume << std::endl;

    return true;
}

void Grid::printInvariants()
{
    auto bary=getBarycenter(m_points, m_triangles);
    std::cerr << "Grid::load: volume=" << std::setprecision(20) << bary[3] << std::endl;
    std::cerr << "Grid::load: bary=" << std::setprecision(20) << bary[0]/bary[3] << "x"
              << std::setprecision(20) << bary[1]/bary[3] << "x"
              << std::setprecision(20) << bary[2]/bary[3] << std::endl;
 }

bool Grid::save(char const * filename) const
{
    std::ofstream file(filename, std::ofstream::trunc);
    if (!file.is_open()) {
        std::cerr << "Grid::save: can not open " << filename << "\n";
        return false;
    }

#ifdef USE_DOUBLE
    file.precision(20);
#endif
    file << m_size << "\n";
    for (auto const &inDomain : m_inDomain) file << inDomain << " ";
    file << "\n";
    if (!file.good()) {
        if (m_debug) std::cerr << "Grid::save[" << filename << "]: can not write the in domain vector\n";
        return false;
    }

    file << int(m_points.size()/4) << "\n";
    for (size_t i=0; i<m_points.size()/4; ++i)
        file << m_points[4*i] << " " << m_points[4*i+1] << " "<< m_points[4*i+2] << " " ;
    file << "\n";
    if (!file.good()) {
        if (m_debug) std::cerr << "Grid::save[" << filename << "]: can not write the points\n";
        return false;
    }

    file << int(m_triangles.size()) << "\n";
    for (auto t : m_triangles) file << t << " ";
    file << "\n";
    if (!file.good()) {
        if (m_debug) std::cerr << "Grid::save[" << filename << "]: can not write the triplets\n";
        return false;
    }

    file << int(m_trianglesList.size()) << "\n";
    for (auto t : m_trianglesList) file << t << " ";
    file << "\n";
    if (!file.good()) {
        if (m_debug) std::cerr << "Grid::save[" << filename << "]: can not write the triangles indirections\n";
        return false;
    }

    file << int(m_offsets.size()) << "\n";
    for (auto t : m_offsets) file << t << " ";
    file << "\n";

    if (!file.good()) {
        if (m_debug) std::cerr << "Grid::save[" << filename << "]: can not write the offsets\n";
        return false;
    }
    
    return true;
}
