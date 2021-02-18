#pragma once

#include <vector>

#include "cvoro_config.h"

/** local structure uses to store a volume defined by triangles in a
    grid with m_size x m_size x m_size voxels which defines the volume [0..1000]^3

    \note each voxel (i,j,k) is stored in position vId=i+j*m_size+k*m_size*m_size
          with 0<=i<m_size, 0<=j<m_size, 0<=k<m_size
          the first corner of this voxel is (i*m_voxelSize, j*m_voxelSize, k*m_voxelSize)
*/
struct Grid {
    /// constructor
    explicit Grid(bool debug=false)
        : m_size(0)
        , m_voxelSize(0)
        , m_debug(debug)
    {
    }
    /// checks if the data are valid or not: consistent, volume>0 and no empty triangles
    bool valid() const;
    /// tries to load a file data: internal format
    bool load(char const * filename);
    /// tries to save a file data: internal format
    bool save(char const * filename) const;

    /** returns the surface volume

     \note VoroAlgo assume that the triangles are consistently
           oriented and this volume is greater than 0*/
    double getVolume() const;
    /// change all surface' triangles orientation
    void swapTrianglesOrientation();
    /** prints the volume invariants: the volume and its barycenter to std::cerr */
    void printInvariants();

    /// the numbers of voxel in one dimension
    int m_size;
    /// 1000/m_size
    real m_voxelSize;
    
    /** list of size*size*size char: for each first corner of a voxel
        a flag to know if this corner is outside/on the border/inside
        of the volume with the convention:
            0: outside, 1: on border, 2: inside the volume */
    std::vector<char> m_inDomain;
    /// the vertices points of the volume: sequence of xyzw(with w=0)
    std::vector<real> m_points;
    /// the triangles which define the surface: 3 points id by triangles
    std::vector<int> m_triangles;
    /** a list of triangles id list */
    std::vector<int> m_trianglesList;
    /** a list of offsets by voxel + one offset: m_trianglesList.size()

        For each voxel vId, the triangles id which intersect the voxel are:
          + m_trianglesList[m_offsets[vId]],
          + m_trianglesList[m_offsets[vId]+1],
          ...
          + m_trianglesList[m_offsets[vId+1]-1]

        \note we assume that these lists are sorted in increasing order
          ie. m_trianglesList[m_offsets[vId]] < m_trianglesList[m_offsets[vId]+1] < ...
     */
    std::vector<int> m_offsets;

protected:
    bool m_debug;
};

