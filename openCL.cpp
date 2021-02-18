#include <fstream>
#include <iostream>
#include <map>
#include <memory>

#include "openCL.h"

//Define the max shared memory size used to 32 kB.
#define LOCAL_SIZE 32768

namespace {
struct ProgramStorage {
    explicit ProgramStorage(cl_program program=nullptr)
        : m_program(program)
    {
    }
    ProgramStorage(ProgramStorage const &)=delete;
    ProgramStorage &operator=(ProgramStorage const &)=delete;
    ~ProgramStorage()
    {
        if (m_program) clReleaseProgram(m_program);
    }
    bool ok() const
    {
        return m_program!=nullptr;
    }
    bool read(std::string const &filename, cl_context context, cl_device_id device, bool debug) {
        if (m_program) {
            clReleaseProgram(m_program);
            m_program=nullptr;
        }
        if (filename.empty()) return false;
        std::ifstream file(filename.c_str(), std::ifstream::binary);
        if (!file.is_open())
            return false;
        size_t len;
        file.read(reinterpret_cast<char *>(&len), sizeof(len));
        if (len==0) return false;
        std::string data;
        data.resize(len);
        file.read(const_cast<char *>(data.c_str()), long(len));
        if (!file.good()) {
            std::cerr << "ProgramStorage::read: problem when reading " << filename << "\n";
            return false;
        }
        if (debug)
            std::cerr << "ProgramStorage::read: read " << filename << "\n";
        
        unsigned char const *ptr=reinterpret_cast<unsigned char const *>(data.c_str());
        cl_program program=clCreateProgramWithBinary(context, 1, &device, &len, &ptr, nullptr, nullptr);
        if (program == nullptr)
        {
            std::cerr << "ProgramStorage::read: Failed to create CL program from binary " << filename << std::endl;
            return false;
        }

        auto errNum = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
        if (errNum != CL_SUCCESS)
        {
            // Determine the reason for the error
            char buildLog[16384];
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                                  sizeof(buildLog), buildLog, nullptr);
            
            std::cerr << "ProgramStorage::read: Error in build program: " << int(errNum) << " from " << filename << std::endl;
            std::cerr << buildLog;
            clReleaseProgram(program);
            data.clear();
            return false;
        }
        m_program=program;
        return true;
    }
    static bool save(cl_program program, std::string const &filename, bool debug) {
        if (filename.empty()) return true;
        if (!program) return false;
        size_t size;
        auto errNum=clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &size, nullptr);
        if (errNum != CL_SUCCESS)
        {
            std::cerr << "ProgramStorage::save: Can not get program binary sizes: " << int(errNum) << " for " << filename <<  std::endl;
            return false;
        }
        std::string data;
        data.resize(size);
        char *ptr=const_cast<char *>(data.c_str());
        clGetProgramInfo(program, CL_PROGRAM_BINARIES, size, &ptr, nullptr);
        if (errNum != CL_SUCCESS)
        {
            std::cerr << "ProgramStorage::save: Can not get program binaries: " << int(errNum) << " for " << filename << std::endl;
            return false;
        }
        std::ofstream file(filename.c_str(), std::ifstream::trunc | std::ifstream::binary);
        if (!file.is_open()) {
            std::cerr << "ProgramStorage:save: can not open " << filename << "\n";
            return false;
        }
        file.write(reinterpret_cast<char *>(&size), sizeof(size_t));
        file.write(data.c_str(), long(size));
        if (!file.good()) {
            std::cerr << "ProgramStorage::save: problem when saving " << filename << "\n";
            return false;
        }
        if (debug)
            std::cerr << "ProgramStorage::save: save " << filename << "\n";
        return true;
    }
    cl_program m_program;
};

}

struct OpenCLContextCache {
    std::map<std::string, std::shared_ptr<ProgramStorage> > m_nameToProgram;
};

OpenCLContext::OpenCLContext(bool debug)
    : m_context(nullptr)
    , m_contextRetained(false)
    , m_device(nullptr)
    , m_supportDouble(false)
    , m_queue(nullptr)
    , m_cacheDirectory()
    , m_isEnabledCLProfiling(false)
    , m_debug(debug)
    , m_data(new OpenCLContextCache)
{
}

OpenCLContext::~OpenCLContext() {
    if (m_queue != nullptr)
        clReleaseCommandQueue(m_queue);
    
    if (m_context != nullptr &&  m_contextRetained)
        clReleaseContext(m_context);
    delete m_data;
}

bool OpenCLContext::setContextAndDevice(cl_context context, cl_device_id device)
{
    if (m_context) {
        std::cerr << "OpenCLContext::setContextAndDevice: a context is already created\n";
        return false;
    }
    m_context=context;
    m_contextRetained=false;
    m_device=device;
    return true;
}

bool OpenCLContext::createContext(bool useGPU)
{
    if (m_context) {
        std::cerr << "OpenCLContext::createContext: a context is already created\n";
        return false;
    }
    cl_int errNum;
    cl_uint numPlatforms;
    cl_platform_id platformsId;
    // First, select an OpenCL platform to run on.
    // For this example, we simply choose the first available
    // platform. Normally, you would query for all available
    // platforms and select the most appropriate one.
    errNum = clGetPlatformIDs(1, &platformsId, &numPlatforms);
    if (errNum != CL_SUCCESS || numPlatforms <= 0) {
        if (m_debug)
            std::cerr << "OpenCLContext::createContext: Failed to find any OpenCL platforms." << std::endl;
        return false;
    }
    // Next, create an OpenCL context on the platform.
    cl_context_properties contextProperties[] = {
        CL_CONTEXT_PLATFORM, cl_context_properties(platformsId),
        0
    };
    m_context = clCreateContextFromType(contextProperties,
                                      useGPU ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU,
                                      nullptr, nullptr, &errNum);
    m_contextRetained = true;
    if (errNum != CL_SUCCESS || !m_context) {
        if (m_debug)
            std::cerr << "OpenCLContext::createContext: Could not create CPU/GPU context..." << std::endl;
        return false;
    }

    // ------------------------------------------------------------
    
    // In this example, we just choose the first available device.
    // In a real program, you would likely use all available
    // devices or choose the highest performance device based on
    // OpenCL device queries.
    errNum = clGetContextInfo(m_context, CL_CONTEXT_DEVICES, sizeof(cl_device_id), &m_device, nullptr);
    if (errNum != CL_SUCCESS) {
        if (m_debug)
            std::cerr << "OpenCLContext::createContext: Failed to get device IDs";
        return false;
    }
    cl_device_fp_config cfg;
    clGetDeviceInfo(m_device, CL_DEVICE_DOUBLE_FP_CONFIG, sizeof(cfg), &cfg, nullptr);
    m_supportDouble=cfg!=0;
    
    if (m_debug) {
        char name[256];
        clGetDeviceInfo(m_device, CL_DEVICE_NAME, 256, name, nullptr);
        std::cerr<<"Using device: " << name << std::endl;
    }
    
    return true;
}

bool OpenCLContext::createCommandQueue(bool enableCLProfiling)
{
    if (m_queue) {
        std::cerr << "OpenCLContext::createCommandQueue: oops, a queue already exists\n";
        return false;
    }
    m_queue = clCreateCommandQueue(m_context, m_device, enableCLProfiling ? CL_QUEUE_PROFILING_ENABLE : 0, nullptr);
    if (m_queue == nullptr) {
        std::cerr << "OpenCLContext::createCommandQueue: Failed to create commandQueue for device 0";
        return false;
    }
    m_isEnabledCLProfiling=enableCLProfiling;
    return true;
}


void OpenCLContext::printAllDevices(std::ostream &output)
{
    cl_int errNum;
    cl_uint numPlatforms;
    cl_platform_id platformsId[3];

    errNum = clGetPlatformIDs(3, platformsId, &numPlatforms);
    if (errNum != CL_SUCCESS || numPlatforms <= 0) {
        output << "OpenCLContext::printAllDevices: Failed to find any OpenCL platforms." << std::endl;
        return;
    }
    for (int i=0; i<int(numPlatforms); ++i) {
        cl_uint num;
        cl_device_id devices[4];
        clGetDeviceIDs(platformsId[i], CL_DEVICE_TYPE_ALL, 4, devices, &num);
        output << "--------platform " << i << " --------\n";

        size_t infoSize;
        clGetPlatformInfo(platformsId[i],CL_PLATFORM_EXTENSIONS,0,nullptr,&infoSize);
        char info[infoSize];
        clGetPlatformInfo(platformsId[i],CL_PLATFORM_EXTENSIONS,infoSize,info,nullptr);
        output << "\textensions: " << info << "\n";
        
        for (int j=0; j<int(num); ++j) {
            cl_device_id device = devices[j];
            char name[256];
            clGetDeviceInfo(device, CL_DEVICE_NAME, 256, name, nullptr);
            output << "--- device: " << name << "\n";
            
            size_t vSize;
            clGetDeviceInfo(device, CL_DEVICE_VERSION, 0, nullptr, &vSize);
            char version[vSize];
            clGetDeviceInfo(device, CL_DEVICE_VERSION, vSize, version, &vSize);
            output << "\t version: " << version << "\n";
            
            cl_device_fp_config cfg;
            clGetDeviceInfo(device, CL_DEVICE_DOUBLE_FP_CONFIG, sizeof(cfg), &cfg, nullptr);
            if (!cfg)
                output << "\tno double unit[DO NOT USE THIS DEVICE]\n";
            else {
                std::map<int, char const *> supported=
                    {
                        { CL_FP_DENORM, "denorms are not supported." },
                        { CL_FP_INF_NAN, "INF and NaNs are not supported." },
                        { CL_FP_ROUND_TO_NEAREST, "round to nearest even rounding mode not supported." },
                        { CL_FP_ROUND_TO_ZERO, "round to zero rounding mode not supported." },
                        { CL_FP_ROUND_TO_INF, "round to +ve and -ve infinity rounding modes are not supported." },
                        { CL_FP_FMA, "IEEE754-2008 fused multiply-add is not supported" }
                    };
                for (auto &it : supported)
                    if ((int(cfg) & it.first)==0) output << "\t" << it.second << "\n";
            }

            size_t dInfoSize;
            clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, 0, nullptr, &dInfoSize);
            char dInfo[dInfoSize];
            clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, dInfoSize, dInfo, &dInfoSize);
            output << "\t" << dInfo << "\n";
        }
    }
}

bool OpenCLContext::createKernel(int n, char const *progName, char const *funcName, std::function<std::string()> getSource,
                                 OpenCLMemory &memory, size_t (&globalSize)[1], size_t (&localSize)[1], int maxNLocal, size_t localDataSize) const
{
    if (memory.program) {
        clReleaseProgram(memory.program);
        memory.program=nullptr;
    }
    if (memory.kernel) {
        clReleaseKernel(memory.kernel);
        memory.kernel=nullptr;
    }
    cl_program program=createProgram(getSource, memory, progName);
    if (program == nullptr)
        return false;
    memory.kernel = clCreateKernel(program, funcName, nullptr);
    if (memory.kernel == nullptr) {
        std::cerr << "OpenCLContext::createKernel: Failed to create kernel[" << funcName << "]" << std::endl;
        return false;
    }
    // Queue the kernel up for execution across the array
    size_t maxLocal=1;
    clGetKernelWorkGroupInfo(memory.kernel, m_device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(maxLocal), &maxLocal, nullptr);
    size_t nLocal=1;
    while (2*nLocal<=maxLocal && 2*nLocal<=size_t(maxNLocal) && (localDataSize==0 || 2*nLocal*localDataSize<LOCAL_SIZE)) nLocal*=2;
    globalSize[0] = n>=0 ? nLocal*((size_t(n)+nLocal-1)/nLocal) : 0;
    localSize[0] = nLocal;
    return true;
}

cl_program OpenCLContext::createProgram(std::function<std::string()> getSource, OpenCLMemory &memory, char const *progName) const
{
    cl_program program=nullptr;
    if (progName) {
        auto it=m_data->m_nameToProgram.find(progName);
        if (it==m_data->m_nameToProgram.end()) {
            if (!m_cacheDirectory.empty()) {
                std::string filename=m_cacheDirectory+"/"+progName;
                auto storage=std::make_shared<ProgramStorage>();
                if (storage->read(filename, m_context, m_device, m_debug) && storage->ok()) {
                    m_data->m_nameToProgram[progName]=storage;
                    it=m_data->m_nameToProgram.find(progName);
                }
            }
        }
        if (it!=m_data->m_nameToProgram.end() && it->second.get()) {
            program=it->second->m_program;
            if (program)
                return program;
        }
    }
    auto source=getSource();
    if (source.empty()) {
        std::cerr << "OpenCLContext::createProgram: Failed to find CL program's source " << (progName ? progName : "unknown") << std::endl;
        return nullptr;
    }
    const char *srcStr = source.c_str();
    program = clCreateProgramWithSource(m_context, 1, const_cast<const char**>(&srcStr), nullptr, nullptr);
    if (program == nullptr)
    {
        std::cerr << "OpenCLContext::createProgram: Failed to create CL program from source"  << (progName ? progName : "unknown") << std::endl;
        return nullptr;
    }
    auto errNum = clBuildProgram(program, 1, &m_device, nullptr, nullptr, nullptr);

    if (errNum != CL_SUCCESS)
    {
        // Determine the reason for the error
        char buildLog[16384];
        clGetProgramBuildInfo(program, m_device, CL_PROGRAM_BUILD_LOG, sizeof(buildLog), buildLog, nullptr);
        
        std::cerr << "OpenCLContext::createProgram: Error in build program: " << int(errNum)  << " for "
                  << (progName ? progName : "unknown") << std::endl;
        std::cerr << buildLog;
        clReleaseProgram(program);
        return nullptr;
    }

    if (!progName) {
        memory.program=program;
        return program;
    }
    m_data->m_nameToProgram[progName]=std::make_shared<ProgramStorage>(program);
    if (m_cacheDirectory.empty()) return program;
    std::string filename=m_cacheDirectory+"/"+progName;
    ProgramStorage::save(program, filename, m_debug);
    return program;
}

