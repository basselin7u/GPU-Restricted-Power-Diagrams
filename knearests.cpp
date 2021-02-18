#include <stdio.h>
#include <assert.h>
#include <float.h>
#include <limits.h>


#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>

#include "cvoro_config.h"
#include "knearests.h"
#include "openCL.h"
#include "StopWatch.h"

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

static char const *s_cellPoint= {
    "int cellFromPoint(int dim, real4 p);"
    "int cellFromPoint(int dim, real4 p) {\n"
    "  int i =(int)floor(p.x * (real)dim / 1000.f);\n"
    "  int j =(int)floor(p.y * (real)dim / 1000.f);\n"
    "  int k =(int)floor(p.z * (real)dim / 1000.f);\n"
    "  i = max((int)0, min(i, dim - 1));\n"
    "  j = max((int)0, min(j, dim - 1));\n"
    "  k = max((int)0, min(k, dim - 1));\n"
    "  return i + j*dim + k*dim*dim;\n"
    "}\n"
};

KNearests::KNearests(std::shared_ptr<OpenCLContext> context, cl_mem points, int numpoints, bool debug)
    : m_context(context)
    , allocated_points(0)
    , dim(0)
    , gpu_stored_points(nullptr)
    , gpu_permutation(nullptr)
    , num_cell_offsets(0)
    , gpu_cell_offsets(nullptr)
    , gpu_cell_offset_dists(nullptr)
    , gpu_counters(nullptr)
    , nearest_knearests()
    , m_debug(debug)
{
    allocated_points = numpoints;

    dim = std::max<int>(1,int(round(pow(numpoints / 3.1f, 1.0f / 3))));

    Stopwatch W("KNearests", m_debug);

    //
    // create cell offsets, very naive approach, should be fine, pre-computed once
    //
    int Nmax = 16;
    while (Nmax>=8 && dim<Nmax) Nmax/=2;
    if (dim < Nmax) {
        std::cerr << "KNearests::KNearests: current implementation does not support low number of input points[" << numpoints << "]" << std::endl;
        exit(EXIT_FAILURE);
    }

    int alloc = Nmax*Nmax*Nmax*Nmax;
    std::vector<int> cell_offsets;
    cell_offsets.reserve(size_t(alloc));
    cell_offsets.push_back(0);
    std::vector<real> cell_offset_dists;
    cell_offset_dists.reserve(size_t(alloc));
    cell_offset_dists.push_back(0);
    for (int ring = 1; ring < Nmax; ring++) {
        real d = 1000.f*real(ring - 1) / dim;
        for (int k = -Nmax; k <= Nmax; k++) {
            for (int j = -Nmax; j <= Nmax; j++) {
                for (int i = -Nmax; i <= Nmax; i++) {
                    if (std::max<int>(abs(i), std::max<int>(abs(j), abs(k))) != ring) continue;

                    int id_offset = i + j*dim + k*dim*dim;
                    if (id_offset == 0) { 
                        std::cerr << "KNearests::KNearests: error generating offsets" << std::endl;
                        exit(EXIT_FAILURE); 
                    }
                    cell_offsets.push_back(id_offset);
                    cell_offset_dists.push_back(d*d); // squared
                }
            }
        }
    }
    num_cell_offsets = int(cell_offsets.size());
    auto clContext = m_context->getContext();
    auto clQueue = m_context->getQueue();
    gpu_cell_offsets = clCreateBuffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      sizeof(int) * size_t(num_cell_offsets), cell_offsets.data(),
                                      nullptr);
    gpu_cell_offset_dists = clCreateBuffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      sizeof(int) * cell_offset_dists.size(), cell_offset_dists.data(),
                                      nullptr);
    if (m_debug)
        std::cerr << "KNearests::KNearests: num_cell_offsets = " << num_cell_offsets << std::endl;

    //
    // prepare the counter
    //
    OpenCLMemory memory;
    int const numCounters=dim*dim*dim;
    cl_mem counters = clCreateBuffer(clContext, CL_MEM_READ_WRITE, sizeof(int) * size_t(numCounters+1), nullptr, nullptr);
    memory.retain(counters);
    cl_event fillEvent;
    int zero=0;
    clEnqueueFillBuffer(clQueue, counters, &zero, sizeof(int),
                        0, sizeof(int) * size_t(numCounters+1), 0, nullptr, &fillEvent);
    size_t globalWorkSize[1];
    size_t localWorkSize[1];
    if (!m_context->createKernel(numpoints, "KNCount", "count",
                                 []() {
                                     char const *count= {
                                                         "__kernel void count(__global real4 const *points, int num, int dim, volatile __global int *counter)\n"
                                                         "{\n"
                                                         "  int gid = get_global_id(0);\n"
                                                         "  if (gid>=num) return;\n"
                                                         "  atomic_add(&counter[cellFromPoint(dim, points[gid])],1);\n"
                                                         "}\n"
                                     };
                                     std::string source(s_defineReal);
                                     source += s_cellPoint;
                                     source += count;
                                     return source;
                                 }
                                 , memory, globalWorkSize, localWorkSize))
        return;
    auto errNum = clSetKernelArg(memory.kernel, 0, sizeof(cl_mem), &points);
    errNum |= clSetKernelArg(memory.kernel, 1, sizeof(int), &numpoints);
    errNum |= clSetKernelArg(memory.kernel, 2, sizeof(int), &dim);
    errNum |= clSetKernelArg(memory.kernel, 3, sizeof(cl_mem), &counters);
    if (errNum != CL_SUCCESS) {
        std::cerr << "KNearests::KNearests: Error setting kernel[count] arguments." << std::endl;
        return;
    }
    // Queue the kernel up for execution across the array
    cl_event programEvent;
    errNum = clEnqueueNDRangeKernel(clQueue, memory.kernel, 1, nullptr,
                                    globalWorkSize, localWorkSize,
                                    1, &fillEvent, &programEvent);
    if (errNum != CL_SUCCESS) {
        std::cerr << "KNearests::KNearests: Error queuing kernel[count] for execution." << std::endl;
        return;
    }

    // 
    std::vector<int> oCounters;
    oCounters.resize(size_t(dim*dim*dim)+1, 0);
    clEnqueueReadBuffer(clQueue, counters, CL_TRUE,
                        0, sizeof(int) * size_t(numCounters+1), oCounters.data(), 1, &programEvent, nullptr);
    int actIndex=0;
    for (auto &c : oCounters ) {
        int n = c;
        c = actIndex;
        actIndex += n;
    }
    cl_event wEvent;
    clEnqueueWriteBuffer(clQueue, counters, CL_FALSE,
      0, sizeof(int) * size_t(numCounters+1), oCounters.data(), 0, nullptr, &wEvent);
    gpu_counters=clCreateBuffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                sizeof(int) * oCounters.size(), oCounters.data(),
                                nullptr);
    //
    if (!m_context->createKernel(numpoints, "KNStore", "store",
                                 []() {
                                     char const *store= {
                                                         "__kernel void store(__global real4 const *points, int num, int dim, volatile __global int *counter,\n"
                                                         "                    __global real4 *stored_points, __global int *perm)\n"
                                                         "{\n"
                                                         "  int gid = get_global_id(0);\n"
                                                         "  if (gid>=num) return;\n"
                                                         "  int pos = atomic_add(&counter[cellFromPoint(dim, points[gid])],1);\n"
                                                         "  stored_points[pos] = points[gid];\n"
                                                         "  perm[pos] = gid;\n"
                                                         "}\n"
                                     };
                                     std::string source(s_defineReal);
                                     source += s_cellPoint;
                                     source += store;
                                     return source;
                                 }
                                 , memory, globalWorkSize, localWorkSize))
        return;
 
    gpu_stored_points = clCreateBuffer(clContext, CL_MEM_READ_WRITE, sizeof(real4) * size_t(numpoints), nullptr, nullptr);
    gpu_permutation = clCreateBuffer(clContext, CL_MEM_READ_WRITE, sizeof(cl_int) * size_t(numpoints), nullptr, nullptr);
    errNum = clSetKernelArg(memory.kernel, 0, sizeof(cl_mem), &points);
    errNum |= clSetKernelArg(memory.kernel, 1, sizeof(int), &numpoints);
    errNum |= clSetKernelArg(memory.kernel, 2, sizeof(int), &dim);
    errNum |= clSetKernelArg(memory.kernel, 3, sizeof(cl_mem), &counters);
    errNum |= clSetKernelArg(memory.kernel, 4, sizeof(cl_mem), &gpu_stored_points);
    errNum |= clSetKernelArg(memory.kernel, 5, sizeof(cl_mem), &gpu_permutation);
    if (errNum != CL_SUCCESS) {
        std::cerr << "KNearests::KNearests: Error setting kernel[store] arguments." << std::endl;
        return;
    }
    // Queue the kernel up for execution across the array
    errNum = clEnqueueNDRangeKernel(clQueue, memory.kernel, 1, nullptr,
                                    globalWorkSize, localWorkSize,
                                    1, &wEvent, nullptr);
    if (errNum != CL_SUCCESS) {
        std::cerr << "KNearests::KNearests: Error queuing kernel[store] for execution." << std::endl;
        return;
    }
    clFinish(clQueue); 
}

KNearests::~KNearests() {
    if (gpu_stored_points) clReleaseMemObject(gpu_stored_points);
    if (gpu_permutation) clReleaseMemObject(gpu_permutation);
    if (gpu_cell_offsets) clReleaseMemObject(gpu_cell_offsets);
    if (gpu_cell_offset_dists) clReleaseMemObject(gpu_cell_offset_dists);
    if (gpu_counters) clReleaseMemObject(gpu_counters);
    freeNearests();
}

void KNearests::freeNearests() {
    if (nearest_knearests) clReleaseMemObject(nearest_knearests);
    nearest_knearests=nullptr;
}

bool KNearests::buildKnearests(int K, cl_mem ids, int numIds, int id_offset, double &totalTime)
{
    freeNearests();
    Stopwatch W("buildKnearests", m_debug);
    OpenCLMemory memory;
    nearest_knearests=clCreateBuffer(m_context->getContext(), CL_MEM_READ_WRITE, size_t(K)*sizeof(int)*size_t(numIds), nullptr, nullptr);
    int mOne=-1;
    cl_event fillEvent;
    auto queue=m_context->getQueue();
    clEnqueueFillBuffer(queue, nearest_knearests, &mOne, sizeof(int),
                        0, size_t(K)*sizeof(int)*size_t(numIds), 0, nullptr, &fillEvent);
    
    std::stringstream s;
    s << "KNFind_K" << K;
    std::string progName=s.str();
    size_t const cSize=size_t(K)*(sizeof(real)+sizeof(unsigned int));
    size_t globalWorkSize[1];
    size_t localWorkSize[1];
    if (!m_context->createKernel(numIds, progName.c_str(), "build_knearests",
                                 [&K]()->std::string {
                                     static std::string s_knearestAlgo;
                                     if (s_knearestAlgo.empty()) {
                                         std::ifstream kernelFile(CVORO_KNEAREST_FILE, std::ios::in);
                                         if (!kernelFile.is_open()) {
                                             std::cerr << "KNearests::buildKnearests: Failed to open file for reading: " << CVORO_KNEAREST_FILE  << std::endl;
                                             return "";
                                         }
                                         std::stringstream s1;
                                         s1 << kernelFile.rdbuf();
                                         s_knearestAlgo = s1.str();
                                     }
                                     std::stringstream s1;
                                     s1 << "#define K " << K << "\n";
                                     std::string source(s1.str());
                                     source += s_defineReal;
                                     source += s_cellPoint;
                                     source += s_knearestAlgo;
                                     return source;
                                 }
                                 , memory, globalWorkSize, localWorkSize, 32, cSize))
        return false;
    
    auto errNum = clSetKernelArg(memory.kernel, 0, localWorkSize[0]*cSize, nullptr);
    errNum |= clSetKernelArg(memory.kernel, 1, sizeof(int), &dim);
    errNum |= clSetKernelArg(memory.kernel, 2, sizeof(cl_mem), &gpu_counters);
    errNum |= clSetKernelArg(memory.kernel, 3, sizeof(cl_mem), &gpu_stored_points);
    errNum |= clSetKernelArg(memory.kernel, 4, sizeof(int), &num_cell_offsets);
    errNum |= clSetKernelArg(memory.kernel, 5, sizeof(cl_mem), &gpu_cell_offsets);
    errNum |= clSetKernelArg(memory.kernel, 6, sizeof(cl_mem), &gpu_cell_offset_dists);
    errNum |= clSetKernelArg(memory.kernel, 7, sizeof(cl_mem), &ids);
    errNum |= clSetKernelArg(memory.kernel, 8, sizeof(int), &numIds);
    errNum |= clSetKernelArg(memory.kernel, 9, sizeof(int), &id_offset);
    errNum |= clSetKernelArg(memory.kernel, 10, sizeof(cl_mem), &nearest_knearests);
    if (errNum != CL_SUCCESS) {
        std::cerr << "KNearests::buildKnearests: Error setting kernel arguments." << std::endl;
        return false;
    }
    // Queue the kernel up for execution across the array
    cl_event progEvent;
    errNum = clEnqueueNDRangeKernel(queue, memory.kernel, 1, nullptr,
                                    globalWorkSize, localWorkSize,
                                    1, &fillEvent, &progEvent);
    
    if (errNum != CL_SUCCESS) {
        std::cerr << "KNearests::buildKnearests: Error queuing kernel for execution." << std::endl;
        return false;
    }
    clFinish(queue);
    if (m_context->isProfilingEnabled()) {
        cl_ulong start, end;
        clGetEventProfilingInfo(progEvent, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, nullptr);
        clGetEventProfilingInfo(progEvent, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, nullptr);
        totalTime += double(end-start)/1e9;
    }
    else
        totalTime += W.timeFromStart();
    return true;
}

void KNearests::printStats(std::ostream &output) const
{
    Stopwatch W("kn_printStats", m_debug);
    size_t const numCounters=size_t(dim*dim*dim);
    std::vector<int> counters;
    counters.resize(numCounters+1, 0);
    clEnqueueReadBuffer(m_context->getQueue(), gpu_counters, CL_TRUE,
                        0, sizeof(int) * (numCounters+1), counters.data(), 0, nullptr, nullptr);

    // stats on counters
    int tot = 0;
    int cmin = INT_MAX, cmax = 0;
    std::map<int, int> histo;
    for (size_t c = 0; c < numCounters; c++) {
        int num=counters[c+1]-counters[c];
        histo[num]++;
        cmin = std::min<int>(cmin, num);
        cmax = std::max<int>(cmax, num);
        tot += num;
    }
    output << "Grid:  points per cell: " << cmin << " (min), " << cmax << " (max), " << real(allocated_points)/real(numCounters) << " avg, total " << tot << std::endl;
    for (auto const &it : histo)
        output << "[" << it.first << "] => " << it.second << std::endl;
}
