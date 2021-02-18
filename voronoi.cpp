#include <float.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "basic.h"
#include "cvoro_config.h"
#include "Grid.h"
#include "knearests.h"
#include "openCL.h"
#include "Status.h"
#include "StopWatch.h"
#include "voronoi.h"

static char const *s_defineReal={
#ifndef USE_DOUBLE
    "#define real float\n"
    "#define real4 float4\n"
#else
    "#if cl_khr_fp64\n"
    "#    pragma OPENCL EXTENSION cl_khr_fp64: enable\n"
    "#else\n"
    "#  error Missing double precision extension\n"
    "#endif\n"
    "#define real double\n"
    "#define real4 double4\n"
#endif
};

/////////////////////////////////////////////////////////////
// grid accessor
/////////////////////////////////////////////////////////////
struct GridAccess {
    GridAccess();
    void init(std::shared_ptr<OpenCLContext> context, Grid const &grid);
    ~GridAccess();

    int m_size; // number of cell by dimension
    real m_voxelSize; // 1000/m_size
    bool m_hasVerticesOnBorders;
    cl_mem m_inDomain; // size*size*size: list of corner  0: outside, 1: on border, 2: inside a tet
    cl_mem m_points; // the triangle points (real4)
    cl_mem m_triangles; // 3 points id by triangles
    cl_mem m_trianglesList; // a list of triangle id
    cl_mem m_offsets; // a list of offset, one by cell + 1 (last has value m_trianglesList.size()
};

GridAccess::GridAccess()
    : m_size(0)
    , m_voxelSize(0)
    , m_hasVerticesOnBorders(false)
    , m_inDomain(nullptr)
    , m_points(nullptr)
    , m_triangles(nullptr)
    , m_trianglesList(nullptr)
    , m_offsets(nullptr)
{
}

GridAccess::~GridAccess()
{
    if (m_inDomain) clReleaseMemObject(m_inDomain);
    if (m_points) clReleaseMemObject(m_points);
    if (m_triangles) clReleaseMemObject(m_triangles);
    if (m_trianglesList) clReleaseMemObject(m_trianglesList);
    if (m_offsets) clReleaseMemObject(m_offsets);
}

void GridAccess::init(std::shared_ptr<OpenCLContext> context, Grid const &grid)
{
    m_size=grid.m_size;
    m_voxelSize=grid.m_voxelSize;
    m_inDomain = clCreateBuffer(context->getContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                sizeof(char) * grid.m_inDomain.size(), const_cast<char *>(grid.m_inDomain.data()),
                                nullptr);
    m_points = clCreateBuffer(context->getContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                              sizeof(real) * grid.m_points.size(), const_cast<real *>(grid.m_points.data()),
                              nullptr);
    m_triangles = clCreateBuffer(context->getContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 sizeof(int) * grid.m_triangles.size(), const_cast<int *>(grid.m_triangles.data()),
                                 nullptr);
    m_trianglesList = clCreateBuffer(context->getContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     sizeof(int) * grid.m_trianglesList.size(), const_cast<int *>(grid.m_trianglesList.data()),
                                     nullptr);
    m_offsets = clCreateBuffer(context->getContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               sizeof(int) * grid.m_offsets.size(), const_cast<int *>(grid.m_offsets.data()),
                               nullptr);

    // check if some vertices are on the border
    OpenCLMemory memory;
    cl_mem counters = clCreateBuffer(context->getContext(), CL_MEM_READ_WRITE, sizeof(int), nullptr, nullptr);
    memory.retain(counters);
    auto queue=context->getQueue();
    int zero=0;
    cl_event fillEvent;
    clEnqueueFillBuffer(queue, counters, &zero, sizeof(int), 0, sizeof(int), 0, nullptr, &fillEvent);
    size_t globalWorkSize[1];
    size_t localWorkSize[1];
    int numInDomain=int(grid.m_inDomain.size());
    if (!context->createKernel(numInDomain, "InDomainCheck", "check",
                               []() {
                                   char const *count= {
                                       "__kernel void check(__global char const *inDomain, int num, volatile __global int *counter)\n"
                                       "{\n"
                                       "  int gid = get_global_id(0);\n"
                                       "  if (gid>=num) return;\n"
                                       "  if (inDomain[gid]==1) counter[0]=1;\n"
                                       "}\n"
                                   };
                                   return count;
                               }
                               , memory, globalWorkSize, localWorkSize))
        return;

    auto errNum = clSetKernelArg(memory.kernel, 0, sizeof(cl_mem), &m_inDomain);
    errNum |= clSetKernelArg(memory.kernel, 1, sizeof(int), &numInDomain);
    errNum |= clSetKernelArg(memory.kernel, 2, sizeof(cl_mem), &counters);
    if (errNum != CL_SUCCESS) {
        std::cerr << "GridAccess::init: Error setting kernel arguments." << std::endl;
        return;
    }
    // Queue the kernel up for execution across the array
    cl_event programEvent;
    errNum = clEnqueueNDRangeKernel(queue, memory.kernel, 1, nullptr,
                                    globalWorkSize, localWorkSize,
                                    1, &fillEvent, &programEvent);
    if (errNum != CL_SUCCESS) {
        std::cerr << "GridAccess::init: Error queuing kernel for execution." << std::endl;
        return;
    }
    int counterInDomain;
    clEnqueueReadBuffer(queue, counters, CL_TRUE,
                        0, sizeof(int), &counterInDomain, 1, &programEvent, nullptr);
    m_hasVerticesOnBorders=counterInDomain>0;
}

/////////////////////////////////////////////////////////////
// basic
/////////////////////////////////////////////////////////////

static std::string const &getStatusCode()
{
    static std::string s_statusAlgo;
    if (s_statusAlgo.empty()) {
        std::ifstream kernelFile(CVORO_STATUS_FILE, std::ios::in); // changeme
        if (!kernelFile.is_open()) {
            std::cerr << "getStatusCode[voronoi.cpp]: Failed to open file for reading: " << CVORO_STATUS_FILE  << std::endl;
            return s_statusAlgo;
        }
        std::stringstream s;
        s << kernelFile.rdbuf();
        s_statusAlgo = s.str();
    }
    return s_statusAlgo;
}

static char const *(StatusStr[STATUS_NUM]) = {
	"vertex_overflow","plane_overflow","inconsistent_boundary","security_radius_not_reached","success",
    "needs_exact_predicates", "empty_cell", "find_another_beginning_vertex", "triangle_overflow", "seed_on_border",
    "hessian_overflow"
};

static void showStatusStats(OpenCLContext const &context, cl_mem cl_status, int numSeeds, bool debug) {
    Stopwatch W("show_status", debug);
    OpenCLMemory memory;
    cl_mem counters = clCreateBuffer(context.getContext(), CL_MEM_READ_WRITE, sizeof(int) * STATUS_NUM, nullptr, nullptr);
    memory.retain(counters);
    auto queue=context.getQueue();
    cl_event fillEvent;
    int zero=0;
    clEnqueueFillBuffer(queue, counters, &zero, sizeof(int), 0, sizeof(int) * STATUS_NUM, 0, nullptr, &fillEvent);
    size_t globalWorkSize[1];
    size_t localWorkSize[1];
    if (!context.createKernel(numSeeds, "VoroCount", "count",
                              []() {
                                  char const *count= {
                                                      "__kernel void count(__global Status const *status, int num, volatile __global int *counter)\n"
                                                      "{\n"
                                                      "  int gid = get_global_id(0);\n"
                                                      "  if (gid>=num || status[gid]==success) return;\n"
                                                      "  atomic_add(&counter[status[gid]],1);\n"
                                                      "}\n"
                                  };
                                  std::string program(getStatusCode());
                                  program += count;
                                  return program;
                              }
                              , memory, globalWorkSize, localWorkSize))
        return;

    auto errNum = clSetKernelArg(memory.kernel, 0, sizeof(cl_mem), &cl_status);
    errNum |= clSetKernelArg(memory.kernel, 1, sizeof(int), &numSeeds);
    errNum |= clSetKernelArg(memory.kernel, 2, sizeof(cl_mem), &counters);
    if (errNum != CL_SUCCESS) {
        std::cerr << "showStatusStats[voronoi.cpp]: Error setting kernel arguments." << std::endl;
        return;
    }
    // Queue the kernel up for execution across the array
    cl_event programEvent;
    errNum = clEnqueueNDRangeKernel(queue, memory.kernel, 1, nullptr,
                                    globalWorkSize, localWorkSize,
                                    1, &fillEvent, &programEvent);
    if (errNum != CL_SUCCESS) {
        std::cerr << "showStatusStats[voronoi.cpp]: Error queuing kernel for execution." << std::endl;
        return;
    }

	int nb_status[STATUS_NUM];
    clEnqueueReadBuffer(queue, counters, CL_TRUE,
                        0, sizeof(int) * STATUS_NUM, &nb_status[0], 1, &programEvent, nullptr);
    int errors=0;
    FOR(r, STATUS_NUM) errors+=nb_status[r];
    if (errors==0) return;
    nb_status[success]=numSeeds-errors;
	std::cerr << "---------Summary of success/failure------------\n";
	FOR(r, STATUS_NUM) { if (nb_status[r]) std::cerr << " " << StatusStr[r] << "   " << nb_status[r] << "\n"; }
	std::cerr << " " << StatusStr[success] << "   " << nb_status[success] << " /  " << numSeeds << "\n";
}

//------------------------------------------------------------
//    VoroAlgoPrivateData
//------------------------------------------------------------
struct VoroAlgoPrivateData {
    explicit VoroAlgoPrivateData(std::shared_ptr<OpenCLContext> context)
        : m_context(context)
        , m_grid()

        , m_statusCounter()
          
        , m_memory()
        , m_numDebugCounters()
        , m_debugCounters()
    {
    }
    ~VoroAlgoPrivateData()
    {
        for (auto mem : m_memory) {
            if (mem) clReleaseMemObject(mem);
        }
    }
    // retain a cl_mem object, it will be automatically deleted when this instance is released
    void retain(cl_mem mem) {
        m_memory.insert(mem);
    }
    // release claim on a cl_mem object, the caller is responsible to delete it
    void release(cl_mem mem) {
        m_memory.erase(mem);
    }
    
    void dCounterCreate(int id, int numValues) {
        if (id<0) return;
        if (id>=int(m_debugCounters.size())) {
            m_numDebugCounters.resize(size_t(id+1), 0);
            m_debugCounters.resize(size_t(id+1), nullptr);
        }
        else if (numValues==m_numDebugCounters[size_t(id)])
            return;
        else if (m_debugCounters[size_t(id)]!=nullptr) {
            clReleaseMemObject(m_debugCounters[size_t(id)]);
            release(m_debugCounters[size_t(id)]);
            m_debugCounters[size_t(id)]=nullptr;
        }
        m_numDebugCounters[size_t(id)]=numValues;
        if (numValues<=0) return;
        m_debugCounters[size_t(id)]=clCreateBuffer(m_context->getContext(), CL_MEM_READ_WRITE, sizeof(cl_int)*size_t(numValues), nullptr, nullptr);
        retain(m_debugCounters[size_t(id)]);
    }
    void dCountersAddTo(cl_kernel kernel, unsigned int &n, cl_int &errNum) const {
        for (auto &mem : m_debugCounters) {
            if (mem==nullptr) continue;
            errNum |= clSetKernelArg(kernel, n++, sizeof(cl_mem), &mem);
        }
    }
    void dCountersAddDefine(std::ostream &output) const {
        for (size_t i=0; i<m_debugCounters.size(); ++i)
            if (m_debugCounters[i]) output << "#define DEBUG_COUNTER" << i << "\n";
    }
    void dCountersFillZero(std::vector<cl_event> &events) const {
        for (size_t i=0; i<m_debugCounters.size(); ++i) {
            if (!m_debugCounters[i]) continue;
            cl_event ev;
            cl_int zero=0;
            clEnqueueFillBuffer(m_context->getQueue(), m_debugCounters[i], &zero, sizeof(cl_int), 0, sizeof(cl_int) * size_t(m_numDebugCounters[i]), 0, nullptr, &ev);
            events.push_back(ev);
        }
    }
    void dCountersSave(VoroAlgoStat &stat) const {
        for (size_t i=0; i<m_debugCounters.size(); ++i) {
            if (!m_debugCounters[i]) continue;
            std::vector<int> counters;
            counters.resize(size_t(m_numDebugCounters[i]));
            clEnqueueReadBuffer(m_context->getQueue(), m_debugCounters[i], CL_TRUE,
                                0, sizeof(cl_int) * size_t(m_numDebugCounters[i]), const_cast<int *>(counters.data()), 0, nullptr, nullptr);
            stat.m_idToHistoMap[int(i)]=counters;
        }
    }
    std::string dCountersProgramExtension() const {
        std::stringstream s;
        for (size_t i=0; i<m_debugCounters.size(); ++i) {
            if (m_debugCounters[i]) s << "_D" << i;
        }
        return s.str();
    }
    void sCountersSave(VoroAlgoStat &stat) const {
        if (!m_statusCounter) return;
        std::vector<int> counters;
        counters.resize(STATUS_NUM);
        clEnqueueReadBuffer(m_context->getQueue(), m_statusCounter, CL_TRUE,
                            0, sizeof(cl_int) * STATUS_NUM, const_cast<int *>(counters.data()), 0, nullptr, nullptr);
        stat.m_idToHistoMap[10]=counters;
    }
    void sCounterCreate(bool set) {
        if (!set) {
            clReleaseMemObject(m_statusCounter);
            release(m_statusCounter);
            m_statusCounter=nullptr;
            return;
        }
        if (m_statusCounter) return;
        m_statusCounter=clCreateBuffer(m_context->getContext(), CL_MEM_READ_WRITE, sizeof(cl_int)*STATUS_NUM, nullptr, nullptr);
        retain(m_statusCounter);
    }

    std::shared_ptr<OpenCLContext> m_context;
    std::shared_ptr<GridAccess> m_grid;

    cl_mem m_statusCounter;
    
private:
    std::set<cl_mem> m_memory;

    std::vector<int> m_numDebugCounters;
    std::vector<cl_mem> m_debugCounters;
};

//------------------------------------------------------------
//    VoroAlgoComputeData
//------------------------------------------------------------
struct VoroAlgoComputeData {
    VoroAlgoComputeData(std::shared_ptr<OpenCLContext> context, bool debug)
        : m_context(context)

        , m_kn()
        , m_maxKComputed(100000)
        , m_K(25)

#ifndef USE_DOUBLE
        , m_vhsAlgo(1)
#endif
        , m_P(35)
        , m_V(40)
        , m_PGlobalPercent(0)
          
        , m_T(0)
        , m_outDomain(false)

        , m_H(0)
        , m_HC(1)

        , m_keepDistance(1)
        , m_keepVolume(1)
          
        , m_numSeeds(0)

        , m_status(nullptr)
        , m_usedCells(nullptr)
          
        , m_numPlanes(0)
        , m_planes(nullptr)

        , m_bary(nullptr)
        , m_hessian(nullptr)
        , m_hessianId(nullptr)

        , m_numIds_cl()

        , m_permutationToUser(nullptr)
        , m_permutationToPoints(nullptr)
          
        , m_debug(debug)
        , m_memory()
    {
        for (auto &num : m_numIds) num=0;
        for (auto &id : m_ids) id=nullptr;
    }
    ~VoroAlgoComputeData()
    {
        for (auto mem : m_memory) {
            if (mem) clReleaseMemObject(mem);
        }
    }
    // create the data
    void initMemory(bool createUsedCells, std::vector<cl_event> &events);
    // create update the planes memory
    void updatePlanes(int num);
    // init basic id
    void initIds();
    // recompute m_numIds[0-1] and ids[0-1]: to find all problematic ids and the localCond ids
    bool recomputeIds(VoroAlgoPrivateData const &data, char const *localCond, char const *progName,
                      VoroAlgoStat *stat=nullptr);
    // find the list of id such that m_usedCells[id]=m_keepDistance
    bool findBordering(cl_mem &ids, int &numIds) const;
    // compress the output using m_usedCells to remove all unused output (ie. with m_usedCells[id]==0)
    void compressResult();
    // launch an iteration
    bool launch(VoroAlgoPrivateData const &data, cl_mem ids, int numIds, std::vector<cl_event> const &events,
                VoroAlgoStat &stat) const;

    // retain a cl_mem object, it will be automatically deleted when this instance is released
    void retain(cl_mem mem) {
        m_memory.insert(mem);
    }
    // release claim on a cl_mem object, the caller is responsible to delete it
    void release(cl_mem mem) {
        m_memory.erase(mem);
    }

    std::shared_ptr<OpenCLContext> m_context;
    std::shared_ptr<KNearests> m_kn;

    int m_maxKComputed;
    int m_K;

#ifndef USE_DOUBLE
    int m_vhsAlgo;
#endif
    int m_P;
    int m_V;
    real m_PGlobalPercent;
    
    int m_T;
    bool m_outDomain;
    
    int m_H;
    int m_HC;

    int m_keepDistance; // 0: used, 1: one bordering, -1: all
    real m_keepVolume;

    int m_numSeeds;
    
    cl_mem m_status;
    cl_mem m_usedCells;

    int m_numPlanes;
    cl_mem m_planes;
  
    cl_mem m_bary;
    cl_mem m_hessian;
    cl_mem m_hessianId;

    int m_numIds[2]; // 0: global, 1: local
    cl_mem m_numIds_cl; // idem on device
    cl_mem m_ids[3]; // 0: global, 1: local, 2: tmp

    cl_mem m_permutationToUser; // permutation: perm[output_id]=>id of points given by the user
    cl_mem m_permutationToPoints; // permutation: perm[output_id]=>id of kn points
    
private:
    bool m_debug;
    std::set<cl_mem> m_memory;
};

void VoroAlgoComputeData::initMemory(bool createUsedCells, std::vector<cl_event> &events)
{
    // create id
    initIds();

    auto context=m_context->getContext();
    // allocate status
    m_status = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(Status)*size_t(m_numSeeds), nullptr, nullptr);
    retain(m_status);

    auto queue=m_context->getQueue();
    if (createUsedCells) {
        m_usedCells = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(uchar)*size_t(m_numSeeds), nullptr, nullptr);
        retain(m_usedCells);
        uchar zero=0;
        cl_event event;
        clEnqueueFillBuffer(queue, m_usedCells, &zero, sizeof(uchar), 0, sizeof(uchar)*size_t(m_numSeeds), 0, nullptr, &event);
        events.push_back(event);
    }
    m_bary = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(real)*4*size_t(m_numSeeds), nullptr, nullptr);
    retain(m_bary);
    if (m_H>0) {
        m_hessian = clCreateBuffer(context, CL_MEM_WRITE_ONLY, m_HC*sizeof(real)*size_t(m_numSeeds)*size_t(m_H), nullptr, nullptr);
        retain(m_hessian);
        m_hessianId = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int)*size_t(m_numSeeds)*size_t(m_H), nullptr, nullptr);
        retain(m_hessianId);
    }
}

void VoroAlgoComputeData::updatePlanes(int num)
{
  if (num<m_numPlanes) return;
  if (m_planes)
    release(m_planes);
  auto context=m_context->getContext();
  m_planes = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(real)*4*size_t(num), nullptr, nullptr);
  m_numPlanes=num;
  retain(m_planes);
}

void VoroAlgoComputeData::compressResult()
{
    Stopwatch W("compress_result", m_debug);
    OpenCLMemory memory;
    auto context=m_context->getContext();
    cl_mem permutation = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * size_t(m_numSeeds), nullptr, nullptr);
    memory.retain(permutation);
    cl_event fillEvent;
    int zero=0;
    cl_mem index = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int), nullptr, nullptr);
    memory.retain(index);
    auto queue=m_context->getQueue();
    clEnqueueFillBuffer(queue, index, &zero, sizeof(int), 0, sizeof(int), 0, nullptr, &fillEvent);
    size_t globalWorkSize[1];
    size_t localWorkSize[1];
    if (!m_context->createKernel(m_numSeeds, "VoroComputePerm", "compute_permutation",
                                 []() {
                                     char const *compute_permutation= {
                                                                       "__kernel void compute_permutation(__global uchar const *usedCells, int num, volatile __global int *counter, __global int *perm)\n"
                                                                       "{\n"
                                                                       "  int gid = get_global_id(0);\n"
                                                                       "  if (gid>=num) return;\n"
                                                                       "  if (usedCells[gid]==0)\n"
                                                                       "    perm[gid]=-1; \n"
                                                                       "  else\n"
                                                                       "    perm[gid]=atomic_add(counter,1);\n"
                                                                       "}\n"
                                     };
                                     return std::string(compute_permutation);
                                 }
                                 , memory, globalWorkSize, localWorkSize))
        return;

    auto errNum = clSetKernelArg(memory.kernel, 0, sizeof(cl_mem), &m_usedCells);
    errNum |= clSetKernelArg(memory.kernel, 1, sizeof(int), &m_numSeeds);
    errNum |= clSetKernelArg(memory.kernel, 2, sizeof(cl_mem), &index);
    errNum |= clSetKernelArg(memory.kernel, 3, sizeof(cl_mem), &permutation);
    if (errNum != CL_SUCCESS) {
        std::cerr << "VoroAlgoComputeData::compressResult: Error setting kernel arguments[compute_permutation]." << std::endl;
        return;
    }
    // Queue the kernel up for execution across the array
    cl_event programEvent;
    errNum = clEnqueueNDRangeKernel(queue, memory.kernel, 1, nullptr,
                                    globalWorkSize, localWorkSize,
                                    1, &fillEvent, &programEvent);
    if (errNum != CL_SUCCESS) {
        std::cerr << "VoroAlgoComputeData::compressResult: Error queuing kernel for execution[compute_permutation]." << std::endl;
        return;
    }

    int num;
    clEnqueueReadBuffer(queue, index, CL_TRUE, 0, sizeof(int), &num, 1, &programEvent, nullptr);
    // --------------------
    auto bary = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(real)*4*size_t(num), nullptr, nullptr);
    retain(bary);
    auto permToPoints = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int)*size_t(num), nullptr, nullptr);
    retain(permToPoints);
    auto permToUser = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int)*size_t(num), nullptr, nullptr);
    retain(permToUser);
    cl_mem hessian=nullptr;
    cl_mem hessianId=nullptr;
    if (m_H>0) {
        hessian = clCreateBuffer(context, CL_MEM_WRITE_ONLY, size_t(m_HC)*sizeof(real)*size_t(m_H)*size_t(num), nullptr, nullptr);
        retain(hessian);
        hessianId = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int)*size_t(m_H)*size_t(num), nullptr, nullptr);
        retain(hessianId);
    }
    std::stringstream s;
    s << "VoroCompactRes" << m_keepDistance << "x" << m_keepVolume;
    if (m_H>0) s << "_H" << m_H << "x" << m_HC;
    std::string programName=s.str();
    if (!m_context->createKernel(m_numSeeds, programName.c_str(), "compact_result",
                                 [this]() {
                                     std::stringstream s1;
                                     char const *compact_result=
                                         {
                                          "__kernel void compact_result\n"
                                          "  (__global real4 const *pts, __global real4 const *seeds, __global uchar const *usedCells, __global int const *perm \n"
                                          "  , int num, __global real4 *newSeeds, __global int *permToPoints\n"
                                          "  , __global int const *origPermToUser, __global int *permToUser\n"
                                          "#ifdef CHESSIAN\n"
                                          "  , __global real const * hessian, __global real *newHessian\n"
                                          "  , __global int const * hessianId, __global int *newHessianId\n"
                                          "#endif\n"
                                          "  )\n"
                                          "{\n"
                                          "  int gid = get_global_id(0);\n"
                                          "  if (gid>=num) return;\n"
                                          "  int pos=perm[gid];\n"
                                          "  if (pos<0) return;\n"
                                          "  permToPoints[pos]=gid;\n"
                                          "  permToUser[pos]=origPermToUser[gid];\n"
                                          "  newSeeds[pos]=seeds[gid]; \n"
                                          "  if (seeds[gid].w < EPSILON_VOLUME) {\n"
                                          "    if (usedCells[gid]==MAX_DIST)\n"
                                          "      newSeeds[pos].w=NO_CELL_LIMITS; \n"
                                          "#if defined(CHESSIAN) && H>0\n"
                                          "    newHessianId[H*pos]=-1;\n"
                                          "#endif\n"
                                          "    return; \n"
                                          "  }\n"
                                          "#ifdef CHESSIAN\n"
                                          "  int w=0, begPos=H*pos;\n"
                                          "  for (int h=0; h<H; ++h) {\n"
                                          "    int hId=hessianId[H*gid+h];\n"
                                          "    if (hId<0) break;\n"
                                          "    if (perm[hId]<0 || seeds[hId].w < EPSILON_VOLUME) { /*printf(\"Arrgh hId=%d[%f]\\n\", hId, hessian[H*gid+h]);*/ continue; }\n" // happens but not frequent
                                          "    newHessianId[begPos+w]=perm[hId];\n"
                                          "#if HC==1\n"
                                          "    newHessian[begPos+w]=hessian[H*gid+h];\n"
                                          "#else\n"
                                          "    for (int c=0; c<4; ++c) newHessian[4*(begPos+w)+c]=hessian[4*(H*gid+h)+c];\n"
                                          "#endif\n"
                                          "    ++w;\n"
                                          "  }"
                                          "  if (w<H) newHessianId[begPos+w]=-1;\n"
                                          "#endif\n"
                                          "}\n"
                                         };
                                     s1 << s_defineReal;
                                     s1 << "#define EPSILON_VOLUME " << m_keepVolume << "\n";
                                     s1 << "#define NO_CELL_LIMITS " << NO_CELL_LIMITS << "\n";
                                     s1 << "#define MAX_DIST " << m_keepDistance << "\n";
                                     if (m_H>0) {
                                         s1 << "#define CHESSIAN\n";
                                         s1 << "#define H " << m_H << "\n";
                                         s1 << "#define HC " << m_HC << "\n";
                                     }
                                     std::string program=s1.str();
                                     program+=compact_result;
                                     return program;
                      }
                      , memory, globalWorkSize, localWorkSize))
        return;

    unsigned int n=0;
    cl_mem originalPts=m_kn->getPoints();
    cl_mem originalPerm=m_kn->getPermutation();
    errNum = clSetKernelArg(memory.kernel, n++, sizeof(cl_mem), &originalPts);
    errNum |= clSetKernelArg(memory.kernel, n++, sizeof(cl_mem), &m_bary);
    errNum |= clSetKernelArg(memory.kernel, n++, sizeof(cl_mem), &m_usedCells);
    errNum |= clSetKernelArg(memory.kernel, n++, sizeof(cl_mem), &permutation);
    errNum |= clSetKernelArg(memory.kernel, n++, sizeof(int), &m_numSeeds);
    errNum |= clSetKernelArg(memory.kernel, n++, sizeof(cl_mem), &bary);
    errNum |= clSetKernelArg(memory.kernel, n++, sizeof(cl_mem), &permToPoints);
    errNum |= clSetKernelArg(memory.kernel, n++, sizeof(cl_mem), &originalPerm);
    errNum |= clSetKernelArg(memory.kernel, n++, sizeof(cl_mem), &permToUser);
    if (m_H>0) {
        errNum |= clSetKernelArg(memory.kernel, n++, sizeof(cl_mem), &m_hessian);
        errNum |= clSetKernelArg(memory.kernel, n++, sizeof(cl_mem), &hessian);
        errNum |= clSetKernelArg(memory.kernel, n++, sizeof(cl_mem), &m_hessianId);
        errNum |= clSetKernelArg(memory.kernel, n++, sizeof(cl_mem), &hessianId);
    }
    if (errNum != CL_SUCCESS) {
        std::cerr << "VoroAlgoComputeData::compressResult: Error setting kernel arguments[compact_result]." << std::endl;
        return;
    }
    errNum = clEnqueueNDRangeKernel(queue, memory.kernel, 1, nullptr,
                                    globalWorkSize, localWorkSize,
                                    0, nullptr, nullptr);
    if (errNum != CL_SUCCESS) {
        std::cerr << "VoroAlgoComputeData::compressResult: Error queuing kernel for execution[compact_result]." << std::endl;
        return;
    }
    m_numSeeds=num;
    std::swap(m_bary, bary);
    std::swap(m_hessian, hessian);
    std::swap(m_hessianId, hessianId);
    std::swap(m_permutationToPoints, permToPoints);
    std::swap(m_permutationToUser, permToUser);
    clFinish(queue);
    cl_mem oldMem[] = {bary, hessian, hessianId, permToPoints, permToUser};
    for (auto mem : oldMem) {
        if (!mem) continue;
        release(mem);
        clReleaseMemObject(mem);
    }
}

bool VoroAlgoComputeData::findBordering(cl_mem &ids, int &numIds) const
{
    Stopwatch W("find_bordering", m_debug);
    OpenCLMemory memory;
    cl_event fillEvent;
    int zero=0;
    cl_mem index = clCreateBuffer(m_context->getContext(), CL_MEM_READ_WRITE, sizeof(int), nullptr, nullptr);
    memory.retain(index);
    auto queue=m_context->getQueue();
    clEnqueueFillBuffer(queue, index, &zero, sizeof(int), 0, sizeof(int), 0, nullptr, &fillEvent);
    size_t globalWorkSize[1];
    size_t localWorkSize[1];
    std::stringstream s;
    s << "VoroFindBordering_" << m_keepVolume;
    if (!m_context->createKernel(m_numSeeds, s.str().c_str(), "find_bordering",
                                 [this]() {
                                     std::stringstream s1;
                                     char const *find_bordering= {
                                                                  "__kernel void find_bordering(__global uchar const *usedCells, __global real4 *pts, int num,\n"
                                                                  "                             __global int *ids, int d, volatile __global int *counter)\n"
                                                                  "{\n"
                                                                  "  int gid = get_global_id(0);\n"
                                                                  "  if (gid>=num) return;\n"
                                                                  "  if (usedCells[gid]+1!=d || pts[gid].w>=EPSILON_VOLUME)\n"
                                                                  "    return; \n"
                                                                  "  ids[atomic_add(counter,1)]=gid;\n"
                                                                  "}\n"
                                     };
                                     s1 << s_defineReal;
                                     s1 << "#define EPSILON_VOLUME " << m_keepVolume << "\n";
                                     std::string program=s1.str();
                                     program+=find_bordering;
                                     return program;
                                 }
                                 , memory, globalWorkSize, localWorkSize))
        return false;

    auto errNum = clSetKernelArg(memory.kernel, 0, sizeof(cl_mem), &m_usedCells);
    errNum |= clSetKernelArg(memory.kernel, 1, sizeof(cl_mem), &m_bary);
    errNum |= clSetKernelArg(memory.kernel, 2, sizeof(int), &m_numSeeds);
    errNum |= clSetKernelArg(memory.kernel, 3, sizeof(cl_mem), &ids);
    errNum |= clSetKernelArg(memory.kernel, 4, sizeof(int), &m_keepDistance);
    errNum |= clSetKernelArg(memory.kernel, 5, sizeof(cl_mem), &index);
    if (errNum != CL_SUCCESS) {
        std::cerr << "VoroAlgoComputeData::findBordering: Error setting kernel arguments." << std::endl;
        return false;
    }
    // Queue the kernel up for execution across the array
    cl_event programEvent;
    errNum = clEnqueueNDRangeKernel(queue, memory.kernel, 1, nullptr,
                                    globalWorkSize, localWorkSize,
                                    1, &fillEvent, &programEvent);
    if (errNum != CL_SUCCESS) {
        std::cerr << "VoroAlgoComputeData::findBordering: Error queuing kernel for execution." << std::endl;
        return false;
    }
    clEnqueueReadBuffer(queue, index, CL_TRUE, 0, sizeof(int), &numIds, 1, &programEvent, nullptr);
    return true;
}

void VoroAlgoComputeData::initIds()
{
    std::vector<int> ids;
    ids.reserve(size_t(m_numSeeds));
    for (int i=0; i<m_numSeeds; ++i) ids.push_back(i);
    auto context=m_context->getContext();
    m_numIds[0]=m_numSeeds;
    m_ids[0]=clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int)*size_t(m_numSeeds), const_cast<int *>(ids.data()), nullptr);
    retain(m_ids[0]);
    m_ids[1]=clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int)*size_t(m_numSeeds), nullptr, nullptr);
    retain(m_ids[1]);
    m_ids[2]=clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int)*size_t(m_numSeeds), nullptr, nullptr);
    retain(m_ids[2]);
    m_numIds_cl=clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int)*2, nullptr, nullptr);
    retain(m_numIds_cl);
}

bool VoroAlgoComputeData::recomputeIds(VoroAlgoPrivateData const &data, char const *localCond, char const *progName, VoroAlgoStat *stat)
{
    if (m_numIds[0]<=0) return true;
    Stopwatch W("recomputeIds", m_debug);
    int mZero=0;
    auto queue=m_context->getQueue();
    std::vector<cl_event> events;
    events.resize(1);
    clEnqueueFillBuffer(queue, m_numIds_cl, &mZero, sizeof(int),
                        0, 2*sizeof(int), 0, nullptr, &events.back());
    bool createCounter=stat && data.m_statusCounter;
    if (createCounter) {
        cl_event ev;
        cl_int zero=0;
        clEnqueueFillBuffer(queue, data.m_statusCounter, &zero, sizeof(cl_int), 0, sizeof(cl_int) * STATUS_NUM, 0, nullptr, &ev);
        events.push_back(ev);
    }
    OpenCLMemory memory;
    size_t globalWorkSize[1];
    size_t localWorkSize[1];
    std::stringstream s;
    s << "voroRecompute" << progName;
    if (createCounter) s << "_S";
    if (!m_context->createKernel(m_numIds[0], s.str().c_str(), "recomputeIds",
                                 [localCond, createCounter]() {
                                  char const *idToLocalIds={
                                      "__kernel void recomputeIds(volatile __global int *counter, int numIds, __global int const *ids, __global int *newIds,\n"
                                      "                           __global int *localIds, __global const Status *status\n"
                                      "#ifdef DEBUG_COUNTER\n"
                                      "                           , volatile __global int *sCounters\n"
                                      "#endif\n"
                                      ")\n"
                                      "{\n"
                                      "  int gid=get_global_id(0);\n"
                                      "  if (gid>=numIds) return; \n"
                                      "  int seed=ids[gid]; \n"
                                      "  Status stat=status[seed]; \n"
                                      "  if (stat==success) return; \n"
                                      "#ifdef DEBUG_COUNTER\n"
                                      "  atomic_add(&sCounters[stat],1);\n"
                                      "#endif\n"
                                      "  int pos = atomic_add(&counter[0],1);\n"
                                      "  newIds[pos] = seed;\n"
                                      "  if (!(COND)) return;\n"
                                      "  pos = atomic_add(&counter[1],1);\n"
                                      "  localIds[pos] = seed;\n"
                                      "}\n"
                                  };
                                  std::stringstream s1;
                                  s1 << "#define COND " << localCond << "\n";
                                  if (createCounter) s1 << "#define DEBUG_COUNTER\n";
                                  std::string program(getStatusCode());
                                  program += s1.str();
                                  program += idToLocalIds;
                                  return program;
                              }
                              , memory, globalWorkSize, localWorkSize))
        return false;
    
    unsigned int n=0;
    auto errNum = clSetKernelArg(memory.kernel, n++, sizeof(cl_mem), &m_numIds_cl);
    errNum |= clSetKernelArg(memory.kernel, n++, sizeof(int), &m_numIds[0]);
    errNum |= clSetKernelArg(memory.kernel, n++, sizeof(cl_mem), &m_ids[0]);
    errNum |= clSetKernelArg(memory.kernel, n++, sizeof(cl_mem), &m_ids[2]);
    errNum |= clSetKernelArg(memory.kernel, n++, sizeof(cl_mem), &m_ids[1]);
    errNum |= clSetKernelArg(memory.kernel, n++, sizeof(cl_mem), &m_status);
    if (createCounter)
        errNum |= clSetKernelArg(memory.kernel, n++, sizeof(cl_mem), &data.m_statusCounter);
    if (errNum != CL_SUCCESS) {
        std::cerr << "VoroAlgoComputeData::recomputeIds: Error setting kernel arguments." << std::endl;
        return false;
    }
    cl_event programEvent;
    errNum = clEnqueueNDRangeKernel(queue, memory.kernel, 1, nullptr,
                                    globalWorkSize, localWorkSize,
                                    static_cast<unsigned int>(events.size()), events.data(), &programEvent);
    if (errNum != CL_SUCCESS) {
        std::cerr << "VoroAlgoComputeData::recomputeIds: Error queuing kernel for execution[" << errNum << "]." << std::endl;
        return false;
    }
    int oldNum=m_numIds[0];
    std::swap(m_ids[0],m_ids[2]);
    clEnqueueReadBuffer(queue, m_numIds_cl, CL_TRUE,
                        0, sizeof(int) * 2, m_numIds, 1, &programEvent, nullptr);
    if (stat && oldNum!=m_numIds[0])
        stat->m_numDone=oldNum-m_numIds[0];
    if (createCounter)
        data.sCountersSave(*stat);
    return true;
}

bool VoroAlgoComputeData::launch(VoroAlgoPrivateData const &data, cl_mem ids, int numIds,
                                 std::vector<cl_event> const &prevEvents, VoroAlgoStat &stat) const
{
    stat.m_num = numIds;
    if (numIds<=0)
        return true;
    if (m_P>=256) {
        std::cerr << "VoroAlgoComputeData::launch: P value is too big:" << m_P << "\n";
        return false;
    }
    if (m_debug)
        std::cerr << "-------- " << stat.m_name << "[voro_cell]: " << numIds << " --------\n";
    Stopwatch W("voro_cell[main]", m_debug);
    OpenCLMemory memory;
    
    size_t globalWorkSize[1];
    size_t localWorkSize[1];
    bool const createNeighbours=m_H>0 || (m_usedCells && m_keepDistance>0);
    int P1=int(m_P*m_PGlobalPercent);
    if (m_P-P1<4) P1=std::max(0,m_P-4);
    size_t cSize=16+size_t(m_V)*(sizeof(cl_uchar4))+size_t(m_P-P1)*sizeof(real4)+4*((size_t(m_P)+3)/4);
    if (createNeighbours) cSize += size_t(m_P)*sizeof(int);

    auto const &grid=data.m_grid;
    bool computeBordering=m_usedCells && m_keepDistance>1;
    std::stringstream s;
    s << "VoroCell";
#ifndef USE_DOUBLE
    s << "_A" << m_vhsAlgo;
#endif
    s << "_K" << m_K << "_P" << m_P;
    if (P1) s << "_P1" << P1;
    s << "_V" << m_V;
    if (m_H>0) s << "_H" << m_H << "x" << m_HC;
    if (m_usedCells) s << "_CU" << m_keepDistance << "x" << m_keepVolume;
    if (!computeBordering && grid) {
        s << "_G" << grid->m_size << "x" << grid->m_voxelSize << "_T" << m_T;
        if (m_outDomain) s << "o";
    }
    if (!computeBordering) s << data.dCountersProgramExtension();
    std::string programName(s.str());
    if (!m_context->createKernel(0, programName.c_str(), computeBordering ? "compute_bordering" : "compute_voro_cell",
                                 [this, grid, computeBordering, P1, &data]() -> std::string {
                                   static std::string s_convexCellAlgo;
                                   if (s_convexCellAlgo.empty()) {
                                       std::ifstream kernelFile(CVORO_CONVEX_CELL_FILE, std::ios::in); // changeme
                                       if (!kernelFile.is_open()) {
                                           std::cerr << "VoroAlgoComputeData::launch: Failed to open file for reading: " << CVORO_CONVEX_CELL_FILE  << std::endl;
                                           return "";
                                       }
                                       std::stringstream s1;
                                       s1 << kernelFile.rdbuf();
                                       s_convexCellAlgo = s1.str();
                                   }
                                   std::stringstream s1;
                                   s1 << s_defineReal;
#ifndef USE_DOUBLE
                                   s1 << "#define VHSAlgo " << m_vhsAlgo << "\n";
                                   s1 << "#define VOLUME_EPSILON2 1.e-6\n"; // 1.e-7 give few bad results
#else
                                   s1 << "#define VHSAlgo 2\n";
                                   s1 << "#define VOLUME_EPSILON2 1.e-12\n";
#endif
                                   s1 << "#define K " << m_K << "\n";
                                   s1 << "#define P " << m_P << "\n";
                                   s1 << "#define P1 " << P1 << "\n";
                                   s1 << "#define V " << m_V << "\n";
                                   s1 << "#define NO_CELL_LIMITS " << NO_CELL_LIMITS << "\n";
                                   s1 << "#define CUBE_EPSILON " << (grid ? 0.1 : 0) << "\n";
                                   if (!computeBordering && m_H>0) {
                                       s1 << "#define CHESSIAN\n";
                                       s1 << "#define H " << m_H << "\n";
                                       s1 << "#define HC " << m_HC << "\n";
                                   }
                                   if (m_usedCells) {
                                       s1 << "#define CUSED_CELL\n";
                                       s1 << "#define USED_CELL_EPSILON " << m_keepVolume << "\n";
                                       s1 << "#define CUSED_CELL_DIST " << m_keepDistance << "\n"; 
                                   }
                                   if (!computeBordering && grid) {
                                       s1 << "#define USE_GRID\n";
                                       s1 << "#define GRID_SIZE " << grid->m_size << "\n";
                                       s1 << "#define GRID_VOXEL_SIZE " << grid->m_voxelSize << "\n";
                                       if (m_outDomain) s1 << "#define GRID_OUT_DOMAIN\n";
                                       s1 << "#define T " << m_T << "\n";
                                   }
                                   if (!computeBordering) data.dCountersAddDefine(s1);
                                   std::string program(s1.str());
                                   program += getStatusCode();
                                   program += s_convexCellAlgo;
                                   return program;
                               }
                               , memory, globalWorkSize, localWorkSize, 128, cSize))
        return false;
    auto queue = m_context->getQueue();
    bool isProfilingEnabled=m_context->isProfilingEnabled();
    std::vector<cl_event> events(prevEvents);
    if (!computeBordering) data.dCountersFillZero(events);
    size_t const &nLocal=localWorkSize[0];
    int numLIds=numIds<=m_maxKComputed ? numIds : m_maxKComputed;
    size_t const maxPlanesMemory=300000000; // 300 mega must be enough
    if (numLIds*sizeof(real)*4*P1>maxPlanesMemory)
      numLIds=maxPlanesMemory/P1/sizeof(real)/4;
    int numSteps=(numIds+numLIds-1)/numLIds;
    Stopwatch cutW("cutW",false);
    double sum_knn = 0;
    double sum_convexcell = 0;
    if (P1)
        const_cast<VoroAlgoComputeData *>(this)->updatePlanes(numLIds*P1);
    for (int st=0; st<numSteps; ++st) {
        int kOffset=st*numLIds;
        int num=st+1<numSteps ? numLIds : numIds-kOffset;
        if (!m_kn->buildKnearests(m_K, ids, num, kOffset, sum_knn)) return false;
        Stopwatch WVoro("voro_cell", m_debug);
        unsigned int n=0;
        auto errNum = clSetKernelArg(memory.kernel, n++, nLocal*size_t(m_V)*sizeof(cl_uchar4), nullptr);
        errNum |= clSetKernelArg(memory.kernel, n++, nLocal*size_t(m_P-P1)*sizeof(real4), nullptr);
        if (P1)
            errNum |= clSetKernelArg(memory.kernel, n++, sizeof(cl_mem), &m_planes);
        errNum |= clSetKernelArg(memory.kernel, n++, nLocal*4*((size_t(m_P)+3)/4), nullptr);
        if (createNeighbours)
            errNum |= clSetKernelArg(memory.kernel, n++, nLocal*size_t(m_P)*sizeof(int), nullptr);    
        errNum |= clSetKernelArg(memory.kernel, n++, sizeof(int), &num);
        errNum |= clSetKernelArg(memory.kernel, n++, sizeof(cl_mem), &ids);
        errNum |= clSetKernelArg(memory.kernel, n++, sizeof(int), &kOffset);
        errNum |= clSetKernelArg(memory.kernel, n++, sizeof(cl_mem), &m_kn->getPoints());
        errNum |= clSetKernelArg(memory.kernel, n++, sizeof(cl_mem), &m_kn->getNearests());
        if (!computeBordering && grid) { 
            errNum |= clSetKernelArg(memory.kernel, n++, sizeof(cl_mem), &grid->m_inDomain);
            errNum |= clSetKernelArg(memory.kernel, n++, sizeof(cl_mem), &grid->m_points);
            errNum |= clSetKernelArg(memory.kernel, n++, sizeof(cl_mem), &grid->m_triangles);
            errNum |= clSetKernelArg(memory.kernel, n++, sizeof(cl_mem), &grid->m_trianglesList);
            errNum |= clSetKernelArg(memory.kernel, n++, sizeof(cl_mem), &grid->m_offsets);
        }
        errNum |= clSetKernelArg(memory.kernel, n++, sizeof(cl_mem), &m_status);
        if (!computeBordering)
            errNum |= clSetKernelArg(memory.kernel, n++, sizeof(cl_mem), &m_bary);
        if (!computeBordering && m_H>0) {
            errNum |= clSetKernelArg(memory.kernel, n++, sizeof(cl_mem), &m_hessian);
            errNum |= clSetKernelArg(memory.kernel, n++, sizeof(cl_mem), &m_hessianId);
        }
        if (m_usedCells)
            errNum |= clSetKernelArg(memory.kernel, n++, sizeof(cl_mem), &m_usedCells);
        if (!computeBordering) data.dCountersAddTo(memory.kernel, n, errNum);
        if (errNum != CL_SUCCESS) {
            std::cerr << "VoroAlgoComputeData::launch: Error setting kernel arguments." << std::endl;
            return false;
        }
        // Queue the kernel up for execution across the array
        globalWorkSize[0] = nLocal*((size_t(num)+nLocal-1)/nLocal);
        cl_event progEvent;
        errNum = clEnqueueNDRangeKernel(queue, memory.kernel, 1, nullptr,
                                        globalWorkSize, localWorkSize,
                                        static_cast<unsigned int>(events.size()), !events.empty() ? events.data() : nullptr, &progEvent);
        events.clear();
        if (errNum != CL_SUCCESS) {
            std::cerr << "VoroAlgoComputeData::launch: Error queuing kernel for execution[" << errNum << "]." << std::endl;
            return false;
        }
        clFinish(queue);
        if (isProfilingEnabled) {
            cl_ulong start, end;
            clGetEventProfilingInfo(progEvent, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, nullptr);
            clGetEventProfilingInfo(progEvent, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, nullptr);
            sum_convexcell += double(end-start)/1e9;
        }
        else
            sum_convexcell += WVoro.timeFromStart();
    }

    stat.m_times[0]=sum_knn;
    stat.m_times[1]=sum_convexcell;
    stat.m_times[2]=W.timeFromStart();
    if (!computeBordering) data.dCountersSave(stat);
    return true;
}

//------------------------------------------------------------
//    VoroAlgo
//------------------------------------------------------------
VoroAlgo::VoroAlgo(std::shared_ptr<OpenCLContext> context, bool debug)
    : m_K(25)
    , m_P(35)
    , m_T(100)
    , m_V(40)
    , m_PGlobalPercent(0)
      
    , m_H(0)
    , m_HC(1)
      
    , m_maxKComputed(100000)
    , m_keepDistance(1)
    , m_keepVolume(1)
      
    , m_enableCLProfiling(false)

    , m_enableDCountersHistoK(false)
    , m_enableDCountersHistoP(false)
    , m_enableDCountersHistoT(false)
    , m_enableDCountersHistoV(false)
    , m_enableDCountersMax(false)
    , m_enableDCountersStatus(false)
      
    , m_debug(debug)
      
    , m_data()
    , m_computeData()
{
    m_data = std::make_shared<VoroAlgoPrivateData>(context);
#ifndef USE_DOUBLE
    m_vhsAlgo[0]=1; m_vhsAlgo[1]=2;
#endif
}

VoroAlgo::~VoroAlgo()
{
}

void VoroAlgo::setGrid(Grid const &grid) {
    m_data->m_grid=std::make_shared<GridAccess>();
    m_data->m_grid->init(getContext(), grid);
}

//
// INPUT
//

void VoroAlgo::setInputPoints(cl_mem seeds, int numSeeds)
{
    if (getCLQueue()==nullptr && !getContext()->createCommandQueue(m_enableCLProfiling))
        return;
    m_computeData=std::make_shared<VoroAlgoComputeData>(getContext(), m_debug);
    m_computeData->m_kn=std::make_shared<KNearests>(getContext(), seeds, numSeeds, m_debug);
    m_computeData->m_numSeeds = getNumInputPoints();
    if (m_debug) m_computeData->m_kn->printStats(std::cerr);
}

void VoroAlgo::setInputPoints(std::vector<real> const &seeds)
{
    int numpoints=int(seeds.size()/4);
    if (getCLContext()==nullptr || numpoints==0)
        return;
    cl_mem points = clCreateBuffer(getCLContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   sizeof(real4) * size_t(numpoints), const_cast<real *>(seeds.data()), nullptr);
    setInputPoints(points, numpoints);
    if (points) clReleaseMemObject(points);
}

int VoroAlgo::getNumInputPoints() const {
    if (!m_computeData || !m_computeData->m_kn) return 0;
    return m_computeData->m_kn->getNumPoints();
}

cl_mem VoroAlgo::getInputPointsCL() const {
    if (!m_computeData || !m_computeData->m_kn) return nullptr;
    return m_computeData->m_kn->getPoints();
}

std::vector<real> VoroAlgo::getInputPoints() const
{
    std::vector<real> res;
    if (!m_computeData || !m_computeData->m_kn) return res;
    int numSeeds=m_computeData->m_kn->getNumPoints();
    if (numSeeds<=0) return res;
    res.resize(4*size_t(numSeeds), 0);
    clEnqueueReadBuffer(getCLQueue(), m_computeData->m_kn->getPoints(), CL_TRUE, 0, sizeof(real) * res.size(), res.data(), 0, nullptr, nullptr);
    return res;
}

//
// OUTPUT
//

int VoroAlgo::getNumOutputPoints() const {
    if (!m_computeData) return 0;
    return m_computeData->m_numSeeds;
}

cl_mem VoroAlgo::getBarycentersCL() const {
    if (!m_computeData) return nullptr;
    return m_computeData->m_bary;
}

cl_mem VoroAlgo::getHessiansCL() const {
    if (!m_computeData) return nullptr;
    return m_computeData->m_hessian;
}

cl_mem VoroAlgo::getHessianIdsCL() const {
    if (!m_computeData) return nullptr;
    return m_computeData->m_hessianId;
}

cl_mem VoroAlgo::getPermutationToInputPointsCL() const {
    if (!m_computeData) return nullptr;
    return m_computeData->m_permutationToPoints;
}

cl_mem VoroAlgo::getPermutationToUserCL() const
{
    if (!m_computeData || !m_computeData->m_kn) return nullptr;
    return m_computeData->m_permutationToUser ? m_computeData->m_permutationToUser : m_computeData->m_kn->getPermutation();
}

std::vector<real> VoroAlgo::getBarycenters() const {
    std::vector<real> res;
    if (!m_computeData) return res;
    
    int const &numSeeds=m_computeData->m_numSeeds;
    if (!numSeeds) return res;
    res.resize(4*size_t(numSeeds), 0);
    clEnqueueReadBuffer(getCLQueue(), m_computeData->m_bary, CL_TRUE, 0, sizeof(real) * res.size(), res.data(), 0, nullptr, nullptr);
    return res;
}

std::vector<real> VoroAlgo::getHessians() const
{
    std::vector<real> res;
    if (m_H<=0 || !m_computeData) return res;
    
    int const &numSeeds=m_computeData->m_numSeeds;
    if (!numSeeds) return res;
    res.resize(size_t(m_HC*m_H*numSeeds), 0);
    clEnqueueReadBuffer(getCLQueue(), m_computeData->m_hessian, CL_TRUE, 0, sizeof(real) * res.size(), res.data(), 0, nullptr, nullptr);
    return res;
}

std::vector<int> VoroAlgo::getPermutationToInputPoints() const
{
    std::vector<int> res;
    if (!m_computeData || m_computeData->m_permutationToPoints==nullptr) return res;
    
    int const &numSeeds=m_computeData->m_numSeeds;
    if (!numSeeds) return res;
    res.resize(size_t(numSeeds), 0);
    clEnqueueReadBuffer(getCLQueue(), m_computeData->m_permutationToPoints, CL_TRUE, 0, sizeof(int) * res.size(), res.data(), 0, nullptr, nullptr);
    return res;
}

std::vector<int> VoroAlgo::getPermutationToUser() const
{
    std::vector<int> res;
    cl_mem permToOrig=getPermutationToUserCL();
    int const &numSeeds=m_computeData->m_numSeeds;
    if (!numSeeds || !permToOrig) return res;
    res.resize(numSeeds);
    clEnqueueReadBuffer(getCLQueue(), permToOrig, CL_TRUE, 0, sizeof(int) * res.size(), res.data(), 0, nullptr, nullptr);
    return res;
}

std::vector<int> VoroAlgo::getHessianIds() const
{
    std::vector<int> res;
    if (m_H<=0 || !m_computeData) return res;
    
    int const &numSeeds=m_computeData->m_numSeeds;
    if (!numSeeds) return res;
    res.resize(size_t(m_H*numSeeds), 0);
    clEnqueueReadBuffer(getCLQueue(), m_computeData->m_hessianId, CL_TRUE, 0, sizeof(int) * res.size(), res.data(), 0, nullptr, nullptr);
    return res;
}

std::shared_ptr<OpenCLContext> VoroAlgo::getContext() const {
    return m_data->m_context;
}
cl_context VoroAlgo::getCLContext() const {
    return getContext()->getContext();
}
cl_command_queue VoroAlgo::getCLQueue() const {
    return getContext()->getQueue();
}

void VoroAlgo::initCounters()
{
    m_data->dCounterCreate(0, m_enableDCountersMax ? m_H>0 ? 5 : m_data->m_grid ? 4 : 3 : 0);
    m_data->dCounterCreate(1, m_enableDCountersHistoK ? 1+m_K : 0);
    m_data->dCounterCreate(2, m_enableDCountersHistoP ? m_P : 0);
    m_data->dCounterCreate(3, m_enableDCountersHistoV ? m_V : 0);
    m_data->dCounterCreate(4, ( m_enableDCountersHistoT && m_data->m_grid ) ? m_T : 0);

    m_data->sCounterCreate(m_enableDCountersStatus);
}

bool VoroAlgo::launch(std::function<bool(VoroAlgo const &)> saveData, std::vector<VoroAlgoStat> &stats)
{
    if (!m_computeData || !m_computeData->m_kn) {
        std::cerr << "VoroAlgo::launch: input point are not initialised\n";
        return false;
    }
    auto clContext=getCLContext();
    if (!clContext) {
        std::cerr << "VoroAlgo::launch: context is not initialised\n";
        return false;
    }
#ifdef USE_DOUBLE
    if (!getContext()->supportDouble()) {
        if (m_debug) std::cerr << "VoroAlgo::launch[extra]: stopping because the device does not support double\n";
        return false;
    }
#endif
    
    Stopwatch W("voro_cell[ALL]", m_debug);
    initCounters();
    
    m_computeData->m_maxKComputed=m_maxKComputed;
    m_computeData->m_K=m_K;

#ifndef USE_DOUBLE
    m_computeData->m_vhsAlgo=m_vhsAlgo[0];
#endif
    m_computeData->m_P=m_P;
    m_computeData->m_V=m_V;
    m_computeData->m_PGlobalPercent=m_PGlobalPercent;
    
    m_computeData->m_T=0;
    m_computeData->m_outDomain=false;

    if (m_H>0 && m_HC!=1 && m_HC!=4) {
        std::cerr << "VoroAlgo::launch: unexpected number of hessian component:" << m_HC << "\n";
        m_H=0;
    }
    m_computeData->m_H=m_H;
    m_computeData->m_HC=m_HC;

    // used cell (main phase, we can only compute 0 or 1 bordering)
    m_computeData->m_keepDistance= m_keepDistance>1 ? 1 : m_keepDistance;
    m_computeData->m_keepVolume = m_keepVolume;

    std::vector<cl_event> events;
    // init main memory
    m_computeData->initMemory(m_data->m_grid && m_keepDistance>=0, events);
    auto &numErrorIds=m_computeData->m_numIds;
    auto &errorIds=m_computeData->m_ids;
    if (numErrorIds[0] > 0) {
        stats.push_back(VoroAlgoStat("first"));
        if (!m_computeData->launch(*m_data, errorIds[0], numErrorIds[0], events, stats.back())) return false;
    }

    events.clear();
    int numSeeds=m_computeData->m_numSeeds;
    if (m_debug) showStatusStats(*getContext(), m_computeData->m_status, numSeeds, m_debug);

    // --- triangle overflow ---
    m_computeData->recomputeIds(*m_data, "stat==triangle_overflow", "tr_overflow",
                                numErrorIds[0] > 0 ? &stats.back() : nullptr);
    if (numErrorIds[1] > 0) {
        m_computeData->m_T = m_T;
        stats.push_back(VoroAlgoStat("tr_overflow"));
        if (!m_computeData->launch(*m_data, errorIds[1], numErrorIds[1], events, stats.back())) return false;
    }
    if (numErrorIds[0] > 0) {
        // be sure to use the maximum parameters
        m_computeData->m_T = m_T;
        m_computeData->m_K = m_K;
        m_computeData->m_P = m_P;
        m_computeData->m_V = m_V;
#ifndef USE_DOUBLE
        m_computeData->m_vhsAlgo = m_vhsAlgo[1];
#endif
        m_computeData->m_outDomain= m_data->m_grid && m_data->m_grid->m_hasVerticesOnBorders;
        for (int step=0; step<5; ++step) {
            if (numErrorIds[0]<0) break;
            // no need to continue if we do computation in real or if
            // the device does not support double, ie. this make no
            // sense
#ifndef USE_DOUBLE
            if (step>0 && (m_computeData->m_vhsAlgo<=1 || !getContext()->supportDouble())) {
                if (m_debug) std::cerr << "VoroAlgo::launch[extra]: stopping because the device does not support double\n";
                break;
            }
#endif
            // try to see what has happened
            std::vector<int> status;
            auto it=stats.back().m_idToHistoMap.find(10);
            if (it!=stats.back().m_idToHistoMap.end())
                status=it->second;
            else {
                // the status counters are not created, we need to created them, ...
                m_data->sCounterCreate(true);
                m_computeData->recomputeIds(*m_data, "true", "retrieveStatus", &stats.back());
                it=stats.back().m_idToHistoMap.find(10);
                if (it!=stats.back().m_idToHistoMap.end())
                    status=it->second;
            }
            if (status.size()<STATUS_NUM) {
                if (m_debug) std::cerr << "VoroAlgo::launch[extra]: oops can not retrieve the status\n";
                break;
            }
#if 1
            // possibility better with bad tuning
            bool relaunch=false;
            if (status[vertex_overflow]) { relaunch=true; m_computeData->m_V *= 1.5; }
            if (status[plane_overflow]) { relaunch=true; m_computeData->m_P *= 1.5; }
            if (status[security_radius_not_reached]) { relaunch=true; m_computeData->m_K *= 1.5; }
            if (status[triangle_overflow]) { relaunch=true; m_computeData->m_T *= 1.5; }
            if (!relaunch) break;
#else
            if (status[vertex_overflow] || status[plane_overflow] || status[security_radius_not_reached] || status[triangle_overflow]) {
                m_computeData->m_V *= 1.5;
                m_computeData->m_P *= 1.5;
                m_computeData->m_K *= 1.5;
                m_computeData->m_T *= 1.5;
            }
            else if (step!=0) // if step==0 we also want to relaunch it if we need exact predicate
                break;
#endif
            if (m_debug)
                std::cerr << "VoroAlgo::launch[extra]: relaunch with K=" << m_K << ", P=" << m_P << ", V=" << m_V << ", T=" << m_T << "\n";
            stats.push_back(VoroAlgoStat("extra"));
            if (!m_computeData->launch(*m_data, errorIds[1], numErrorIds[1], events, stats.back())) {
                if (m_debug) std::cerr << "VoroAlgo::launch[extra]: oops the parameters are too big.\n";
                break;
            }
            m_computeData->recomputeIds(*m_data, "stat!=empty_cell", "extra", &stats.back());
        }
    }
    // show stats
    if (m_debug) showStatusStats(*getContext(), m_computeData->m_status, numSeeds, m_debug);

    // retrieve used data
    if (m_computeData->m_usedCells) {
        if (m_keepDistance>1) {
            for (int k=1; k<m_keepDistance; ++k) {
                int numIds;
                m_computeData->m_keepDistance=k+1;
                if (!m_computeData->findBordering(errorIds[2], numIds) || numIds<=0) break;
                std::stringstream s;
                s << "bordering" << k+1;
                stats.push_back(VoroAlgoStat(s.str().c_str()));
                if (!m_computeData->launch(*m_data, errorIds[2], numIds, events, stats.back())) break;
            }
            m_computeData->m_keepDistance=m_keepDistance;
        }
        m_computeData->compressResult();
    }

    // let user save its data
    if (!saveData(*this)) return false;
    clFinish(getCLQueue());
    
    m_computeData.reset();
    return true;
}

