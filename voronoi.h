#pragma once

#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "cvoro_config.h"
#include "openCL.h"

struct Grid;
class OpenCLContext;

/* The first phase of the algorithm is launched when you give the user
   seeds: ie. it creates an accelerating structure to find the K
   nearest neighbours of a seed, this structure reorders the user
   seeds and stores the permutated seeds labelled inputPoints in the
   following.
   
   The second phase begins when you call launch:
   - it computes the voronoi cells, ... 
   - it retrieves the usefull cells if you have called setGrid to define
     a volume and m_keepDist!=-1,
   - finally, it calls saveData to let the users retrieved its data
     and clean the memory.

   In saveData, you can retrieve the output results and/or the permutated
   seeds stored in the accelating structures and two permutations:
   - the permutation to retrieve the inputPoints (permutated user seeds)
     by using getPermutationToInputPoints ; this permutation will be
     empty when the permutation is identity
   - the permutation to retrieve the original position by using
     getPermutationToUsers
     
   More precisely, if we see the proccessing as:
     userSeeds(set by setInputPoints)
          ->inputPoints(stored in the accelerating structure)
          ->output results
   for a value output[i], you can retrieve:
   - the input seeds by using inputPoints[permToInputPoints[i]]
        when the getPermutationToInputPoints is defined or
        directly inputPoints[i] when this permutation is empty
   - some auxilliary data by using myData[perToUser[i]]
   
 Note: concerning the output vertices given by getBarycenters, there
   are given as xyzw with w=volume of the voronoi cell and (x/w,y/w,z/w)
   are the barycenter ; excepted:
   - vertex with .w==NO_CELL_LIMITS which corresponds to a border seed
   - or vertex with .w<m_keepVolume which corresponds to an empty cell
*/
#define NO_CELL_LIMITS -100000

struct Grid;
struct VoroAlgoPrivateData;
struct VoroAlgoComputeData;

/// class to store computation stats
struct VoroAlgoStat {
    /// constructor
    explicit VoroAlgoStat(std::string name)
	: m_name(name)
	, m_num(0), m_numDone(0)
	, m_idToHistoMap()
    {
        for (auto &d : m_times) d=0;
    }
    /// utility function to print this statistic in python norm
    void pythonPrint(std::ostream &output, std::string prefix="") const
    {
        output << "{\"name\" : \"" << prefix << m_name << "\", ";
        char const *(wh[])={"knn", "CC", "time"};
        for (int i=0; i<3; ++i) output << "\"" << wh[i] << "\":" << m_times[i] << ", ";
        for (auto const &it : m_idToHistoMap) {
            output << "\"counter" << it.first << "\":[";
            for (size_t i=0; i<it.second.size(); ++i) {
                output << it.second[i];
                if (i+1!=it.second.size()) output << ",";
            }
            output << "], ";
        }
        if (m_numDone) output << "\"nbdone\":" << m_numDone << ",";
        output << "\"nbcells\":" << m_num;
        output << "}";
    }
    /// basic name
    std::string m_name;
    /// num id
    int m_num;
    /// num of done cells
    int m_numDone;
    /// the different times: 0 : knn, 1: cVoro, 2: total
    double m_times[3];
    /** the different counters:
        0 : {maxK, maxP, maxV, maxT, maxH},
        1 : histoK,
        2 : histoP,
        3 : histoV,
        4 : histoT,
        10 : {vertex_overflow, plane_overflow, ...} (see Status.h)*/
    std::map<int, std::vector<int> > m_idToHistoMap;
};

class VoroAlgo {
public:
    /** constructor given a initialized openCL context and a flag to
        know if we print or not debug message.

     \note the cl_context and cl_device_id must have been initialised with 
        setContextAndDevice or createContext. A cl_command_queue will be created 
        if none exists. */
    explicit VoroAlgo(std::shared_ptr<OpenCLContext> context, bool debug=false);
    /// destructor
    ~VoroAlgo();
    VoroAlgo(VoroAlgo const&)=delete;
    VoroAlgo &operator=(VoroAlgo const&)=delete;

    /** launches the computation: you must call setInputPoints (and if
        you have a grid setGrid) before.

        \note when the simulation is finished, saveData is called ;
               then all computed data are removed */
    bool launch(std::function<bool(VoroAlgo const &)> saveData, std::vector<VoroAlgoStat> &stats);

    /** sets the grid (if you launch multiple simulation, this needs only
        to be set before the first iteration).
        
        \note if no grid is set, the computation will be done on [0,1000]^3
              if a grid is set, the computation will be done on the intersection of the grid and [-0.1,1000.1]^3
    */
    void setGrid(Grid const &grid);

    //
    // INPUT ( getter are only accessible in saveData)
    //

    /** sets the input points to be used in a simulation in openCL: numSeeds x 4 real xyzw
        these points must be in [0x1000]^3 domains.

        \note: seeds can be released when this function returns */
    void setInputPoints(cl_mem seeds, int numSeeds);
    //// sets the input points to be used in a simulation in cpp : 4 real .xyzw by seeds
    void setInputPoints(std::vector<real> const &seeds);
    //// returns the number of input points: numInput
    int getNumInputPoints() const;
    /// returns the input points (after permutation) : numInput x real4
    cl_mem getInputPointsCL() const;
    /// returns the input points (after permutation) : numInput x real4
    std::vector<real> getInputPoints() const;
    
    //
    // OUTPUT (getter are only accessible in saveData)
    //
    
    /// returns the number of output points: num
    int getNumOutputPoints() const;

    /// returns the output barycenters : num x real4
    cl_mem getBarycentersCL() const;
    /// returns the output hessian : num x (m_H * real[m_HC])
    cl_mem getHessiansCL() const;
    /// returns the output barycenters : num x (M_H * int)
    cl_mem getHessianIdsCL() const;
    /** returns the permutation to retrieve the input points positions from output pos
        \note it returns nullptr if this permutation is identity
     */
    cl_mem getPermutationToInputPointsCL() const;
    /** returns the permutation to retrieve original positions from output pos. */
    cl_mem getPermutationToUserCL() const;
    /** returns the output barycenters : num x real4 as bary.xyzw.
        \note bary.w is the volume of the cell */
    std::vector<real> getBarycenters() const;
    /** returns the output hessian :
        - if m_HC=1 num x (M_H * area) of the surface
        - else num x (M_H * bary.xyzw) the barycenter of the surfaces (and bary.w is its area).
     */
    std::vector<real> getHessians() const;
    /// returns the output barycenters : num x (M_H * int)
    std::vector<int> getHessianIds() const;
    /** returns the permutation to retrieve input points positions from output pos
        \note it returns an empty vector if this permutation is identity
     */
    std::vector<int> getPermutationToInputPoints() const;
    /** returns the permutation to retrieve original positions from output pos. */
    std::vector<int> getPermutationToUser() const;

    //
    // access to context
    //

    /// returns the openCL context
    std::shared_ptr<OpenCLContext> getContext() const;
    /// returns the current cl_context
    cl_context getCLContext() const;
    /// returns the current queue
    cl_command_queue getCLQueue() const;

    /// maximum number of K in the first computation's step, it will be multiplied by 1.5 in the second computation's step, ..
    int m_K;
    /// maximum number of planes in the first computation's step, it will be multiplied by 1.5 in the second computation's step, ...
    int m_P;
    /// maximum number of triangles used when intersecting a grid
    int m_T;
    /// maximum number of vertex in the first computation's step, it will be multiplied by 1.5 in the second computation's step, ..
    int m_V;
    /// percent of plane equations stored in global memory (instead of in local memory)
    real m_PGlobalPercent;
    /// maximum number of hessian by seed to be stored
    int m_H;
    /// number of hessian component: 1:area, 4:barycenter with area
    int m_HC;
    
    /// maximum number of neighbours computed in a batch
    int m_maxKComputed;
#ifndef USE_DOUBLE
    /// define the half plane in plane algorithm in first computation's step and at the end (0: real, 1: real with check, 2: double)
    int m_vhsAlgo[2];
#endif
    /// defines the seeds which are stored in the output: 0: used, 1: one bordering, -1: all
    int m_keepDistance;
    /** the volume which is used to decide which cells are valid, ie. the cells such that
        volume(cell)>=m_keepVolume.

        \note as the volume computation is computed as a sum of reals,
             4e-5*minVoronoiCellVolume seems to give good result, but 
             it can give rare false positives. So is probably
             better to choose 1e-4*minVoronoiCellVolume
     */
    real m_keepVolume;

    /// enables CL profiling for stats
    bool m_enableCLProfiling;

    /// enables debug counters to build K's historic
    bool m_enableDCountersHistoK;
    /// enables debug counters to build P's historic
    bool m_enableDCountersHistoP;
    /// enables debug counters to build T's historic
    bool m_enableDCountersHistoT;
    /// enables debug counters to build V's historic
    bool m_enableDCountersHistoV;
    /// enables debug counters to find maximum of K,P,V,T,H
    bool m_enableDCountersMax;
    /// enables debug counters to store status errors (see Status.h)
    bool m_enableDCountersStatus;
    
protected:
    void initCounters();
    
    bool m_debug;

private:
    std::shared_ptr<VoroAlgoPrivateData> m_data;
    std::shared_ptr<VoroAlgoComputeData> m_computeData;
};


