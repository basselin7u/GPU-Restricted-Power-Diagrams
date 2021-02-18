#pragma once

#include <iostream>
#include <memory>
#include <vector>

#include "openCL.h"

class KNearests {
public:
	KNearests(std::shared_ptr<OpenCLContext> context, cl_mem points, int numpoints, bool debug=false);
	~KNearests();
	bool buildKnearests(int K, cl_mem ids, int numIds, int id_offset, double &totalTime);
    // return the number of stored points
    int getNumPoints() const {
        return allocated_points;
    }
	// return the stored points
	cl_mem const &getPoints() const {
		return gpu_stored_points;
	}
	// return the permutation : perm[pointId]=>originalPointId
	cl_mem const &getPermutation() const {
		return gpu_permutation;
	}
	// knn, allocated_points * KN
	cl_mem const &getNearests() const {
		return nearest_knearests;
	}
	void printStats(std::ostream &output) const;
protected:
	void freeNearests();
    
protected:
    std::shared_ptr<OpenCLContext> m_context;
        
	int allocated_points;        // number of input points
	int dim;        // grid resolution
	cl_mem gpu_stored_points;     // input points sorted, numpoints
    cl_mem gpu_permutation; // permutation: perm[storedId]=>original point

	int num_cell_offsets;
	cl_mem gpu_cell_offsets; // cell offsets (sorted by rings), Nmax*Nmax*Nmax*Nmax
	cl_mem gpu_cell_offset_dists;  // stores min dist to the cells in the rings
	cl_mem gpu_counters; // counters per cell,   dim*dim*dim (+1)

	cl_mem nearest_knearests;   // knn, allocated_points * KN

	bool m_debug; // if true, print message
};


