#pragma once

#include <functional>
#include <iostream>
#include <string>
#include <set>

#ifdef __APPLE__
#  include <OpenCL/opencl.h>
#else
#  include <CL/opencl.h>
#endif

/// helper class to manage openCL memory automatically
struct OpenCLMemory {
public:
    /// constructor
    OpenCLMemory()
        : program(nullptr)
        , kernel(nullptr)
        , memory()
    {
    }
    /// destructor
    ~OpenCLMemory() {
        if (program) clReleaseProgram(program);
        if (kernel) clReleaseKernel(kernel);
        for (auto mem : memory) {
            if (mem) clReleaseMemObject(mem);
        }
    }
    /// retain a cl_mem object, it will be automatically deleted when this instance is release
    void retain(cl_mem mem) {
        memory.insert(mem);
    }
    /// release claim on a cl_mem object, the caller is responsible to delete it
    void release(cl_mem mem) {
        memory.erase(mem);
    }
    /// the program
    cl_program program;
    /// the kernel
    cl_kernel kernel;
protected:
    std::set<cl_mem> memory;
};

struct OpenCLContextCache;

/** an openCL container containing a cl_context, a cl_device_id and a cl_command_queue

   It contains a helper function to create kernel and does some caching to avoid
   recomputing the same program. You can also give a directory to store/retrieve
   compiled algorithm.

   \note on OSX, retrieving a compiled algorithm does not work when it is compiled
      for the CPU
 */
class OpenCLContext {
public:
    /// constructor
    explicit OpenCLContext(bool debug);
    /// destructor
    ~OpenCLContext();
    
    OpenCLContext(OpenCLContext const &)=delete;
    OpenCLContext &operator=(OpenCLContext const &)=delete;

    //
    // cl_context and cl_device_id
    //
    
    /// returns the cl_context
    cl_context getContext() const {
        return m_context;
    }
    /// returns true if the cl_context is defined
    bool hasContext() const {
        return m_context!=nullptr;
    }
    /** sets a cl_context and a device 

        \note the caller is responsible to release the cl_context when this
              instance is not longer used
     */
    bool setContextAndDevice(cl_context context, cl_device_id device);
    /** try to create a CPU cl_context

        \note it uses the first CPU device available
     */
    bool createContextOnCPU() { return createContext(false); }
    /** try to create a GPU cl_context

        \note it uses the first GPU device available
     */
    bool createContextOnGPU() { return createContext(true); }
    /** try to create a CPU/GPU cl_context

        \note it uses the first CPU/GPU device available
     */
    bool createContext(bool onGPU);
    /** returns true if the device support double.

        \note: the context and the device must be initialised. */
    bool supportDouble() const {
        return m_supportDouble;
    }

    //
    // queue
    //

    /// returns the current queue
    cl_command_queue getQueue() const {
        return m_queue;
    }
    /// check if a queue has be created
    bool hasQueue() const {
        return m_queue!=nullptr;
    }
    /// creates a queue
    bool createCommandQueue(bool enableCLProfiling);
    /// returns true if the queue has been build with enable profiling
    bool isProfilingEnabled() const {
        return m_isEnabledCLProfiling;
    }

    //
    // kernel & program
    //

    /** creates a kernel using getSource(or a cached version of the source sharing the same progName)
        to define the program, and funcName to define the kernal function ;
        then it fills globalSize and localSize using numIds, maxNLocel and localDataSize 

        \note if progName is sets, use the cache to check if a program with the same name was already
           created or exists in the cache directory, ...
     */
    bool createKernel(int numIds, char const *progName, char const *funcName, std::function<std::string()> getSource,
                      OpenCLMemory &memory, size_t (&globalSize)[1], size_t (&localSize)[1], int maxNLocal=64, size_t localDataSize=0) const;
    /** defines a cache directory to store/retrieve compiled program.

        \note: on OSX, caching CPU's executables does not seem to work...
    */
    void setCacheDirectory(std::string const &dir) {
        m_cacheDirectory=dir;
    }

    //
    // utilities
    //

    /// print all avalaible CL device
    static void printAllDevices(std::ostream &output);
	
protected:
    cl_program createProgram(std::function<std::string()> getSource, OpenCLMemory &memory, char const *progName) const;

    cl_context m_context;
    bool m_contextRetained;
    cl_device_id m_device;
    bool m_supportDouble;
    cl_command_queue m_queue;
    std::string m_cacheDirectory;
    bool m_isEnabledCLProfiling;
    bool m_debug;
private:
    OpenCLContextCache *m_data;
};

void CleanProgramCache();
