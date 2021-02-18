/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <string.h>

#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <string>

#include "cvoro_config.h"
#include "Options.h"
#include "voronoi.h"

Options::Options()
    : m_showHelp(false)
    , m_H(50)
    , m_HC(1)
    , m_K(20)
    , m_P(25)
    , m_T(100)
    , m_V(30)
    , m_maxKComputed(100000)
    , m_PSlowPercent(0)
      
    , m_CPU(false)
    , m_CPUDirectory()
    , m_GPU(false)
    , m_GPUDirectory()
      
    , m_keepDistance(1)
#ifndef USE_DOUBLE
    , m_keepVolume(1)
#else
    , m_keepVolume(0.001)
#endif
    , m_keepDistanceLast(0)
      
    , m_computeHessian(false)
    , m_debug(false)
    , m_gridFile()
    , m_checkGrid(false)

    , m_points4D(false)
    , m_loadBinaryPoints(false)
    , m_saveBinaryPoints("")

    , m_enableProfiling(false)
#ifndef USE_DOUBLE
    , m_cVoroIsInHalfSpaceKernel(2)
#endif
    , m_cVoroDebugCounterHisto(false)
    , m_cVoroDebugCounterMax(false)
    , m_cVoroDebugCounterStatus(false)
    , m_checkInvariants(false)
      
    , m_outputList()
    , m_showCLDevices(false)
      
    , m_boolArgsMap()
    , m_intArgsMap()
    , m_stringArgsMap()
{
    // init the map option name to member positions
    m_boolArgsMap = {
     { "-h", &m_showHelp}, { "--help", &m_showHelp}, 
     { "--CPU", &m_CPU }, { "--GPU", &m_GPU},
     { "--points4D", &m_points4D}, { "--loadBinaryPoints", &m_loadBinaryPoints},
     { "--debug", &m_debug }, { "--checkGrid", &m_checkGrid }, { "--enableProfiling", &m_enableProfiling },
     { "--cVoroDebugCounterMax", &m_cVoroDebugCounterMax }, { "--cVoroDebugCounterHisto", &m_cVoroDebugCounterHisto },
     { "--cVoroDebugCounterStatus", &m_cVoroDebugCounterStatus }, { "--printInvariants", &m_checkInvariants },
     { "--hessian", &m_computeHessian },
     { "--showAllDevices", &m_showCLDevices },
    };
    m_intArgsMap= {
     { "--H", &m_H }, { "--HC", &m_HC }, { "--K", &m_K }, { "--P", &m_P }, { "--T", &m_T }, { "--V", &m_V },
     { "--keepDist", &m_keepDistance}, { "--keepDist2", &m_keepDistanceLast}, { "--maxK", &m_maxKComputed },

#ifndef USE_DOUBLE
     { "--cVoroIsInHalfSpaceAlgo", &m_cVoroIsInHalfSpaceKernel },
#endif
    };
    m_stringArgsMap = {
      { "--grid", &m_gridFile },
      { "--CPU-directory", &m_CPUDirectory}, { "--GPU-directory", &m_GPUDirectory},
      { "--saveBinaryPoints", &m_saveBinaryPoints}
    };
}

Options::~Options()
{
}

int Options::parseOptions(int argc, char const *const *argv)
{
    parseConfigurationFile(CVORO_OPTIONS_FILE);
    for (int i = 1; i < argc; i++)
    {
        if (!parseAnOption(i, argc, argv))
            return i;
    }
    return argc;
}

bool Options::parseConfigurationFile(char const *configName)
{
    std::ifstream file(configName);
    if (!file.is_open())
        return false;
    std::vector<std::string> lines;
    for (std::string line; std::getline( file, line ); /**/ ) {
        // remove comment
        if (!line.empty() && line[0]=='#') continue;
        lines.push_back(line);
    }
    std::vector<char const *>args;
    size_t numLines=lines.size();
    for (size_t l=0; l<numLines; ++l) args.push_back(lines[l].c_str());
    args.push_back(nullptr);
    for (int i = 0; i < int(numLines); i++)
    {
        if (!parseAnOption(i, int(numLines), args.data())) {
            std::cerr << "Options::parseConfigurationFile: can not parse line " << i << ":\n";
            std::cerr << "\t" << lines[size_t(i)] << "\n";
            return false;
        }
    }
    return true;
}

bool Options::parseAnOption(int &i, int argc, char const *const *argv)
{
    if (strlen(argv[i])==0) return true; // no line is ok in a configuration file
    std::string option(argv[i]), arg;
    auto boolIt=m_boolArgsMap.find(option);
    if (boolIt!=m_boolArgsMap.end()) {
        *boolIt->second=true;
        return true;
    }
    
    std::size_t found = option.find('=');
    if (found!=std::string::npos) {
        arg=option.substr(found+1);
        option=option.substr(0,found);
    }
    auto intIt=m_intArgsMap.find(option);
    if (intIt!=m_intArgsMap.end()) {
        if (arg.empty() && i+1>=argc) return false;
        *intIt->second=atoi(arg.empty() ? argv[++i] : arg.c_str());
        return true;
    }
    auto stringIt=m_stringArgsMap.find(option);
    if (stringIt!=m_stringArgsMap.end()) {
        if (arg.empty() && i+1>=argc) return false;
        *stringIt->second= arg.empty() ? argv[++i] : arg.c_str();
        return true;
    }
    // special case list of strings or real
    if (option=="--output" && (!arg.empty() || i+1 < argc)) {
        m_outputList.push_back(arg.empty() ? argv[++i] : arg.c_str());
        return true;
    }
    if (option=="--keepVolume" && (!arg.empty() || i+1 < argc)) {
        m_keepVolume=real(atof(arg.empty() ? argv[++i] : arg.c_str()));
#ifndef USE_DOUBLE
        if (m_keepVolume<=0) m_keepVolume=1; // reset to default
#else
        if (m_keepVolume<=0) m_keepVolume=0.001; // reset to default
#endif
        return true;
    }
    if (option=="--PSlow" && (!arg.empty() || i+1 < argc)) {
        m_PSlowPercent=real(atof(arg.empty() ? argv[++i] : arg.c_str()));
        if (m_PSlowPercent<0 || m_PSlowPercent>1) m_PSlowPercent=0; // reset to default
        return true;
    }

    return false;
}

void Options::showOptions(std::ostream &o) const
{
    o << "\tOPTIONS:\n";
    o << "\t--help|-h: show this help\n";
    o << "\t--H=h: set the max number of Hessian values in a row[default 50]\n";
    o << "\t--HC=C: set the number of Hessian components(1 or 4)[default 1]\n";
    o << "\t--K=k: set the first K value[default 20]\n";
    o << "\t--P=p: set the first P value[default 25]\n";
    o << "\t--T=t: set the max number of grid's triangles[default 100]\n";
    o << "\t--V=v: set the first V value[default 30]\n";
    o << "\t--maxK=N: set the maximum computed K neighbours value[default 100000]\n";
    o << "\t--PSlow=percent: stores some planes' equations in global memory[default 0]\n";
    o << "\t\t 0: means stores no equation in global memory\n";
    o << "\t\t 0.5: means stores half equations in global memory\n";
    o << "\t\t 1: means stores all equations in global memory\n";
    o << "\n";
    o << "\t--points4D: assume that the seeds' file is a 4D ascii file\n";
    o << "\t--loadBinaryPoints: assume that the seeds' file is a 4D binary file\n";
    o << "\t--saveBinaryPoints=file: save the seeds' in a 4D binary file[default \"\"]\n";
    o << "\n";
    o << "\t--CPU: do the computation in CPU\n";
    o << "\t--GPU: do the computation in GPU\n";
    o << "\t--CPU-directory=DIR: the repository to read/store CPU compiled code[default \"\"]\n";
    o << "\t--GPU-directory=DIR: the repository to read/store GPU compiled code[default \"\"]\n";
    o << "\n";
    o << "\t--keepDist=d: set the keep distance, all phase except the last one[default 1]\n";
#ifndef USE_DOUBLE
    o << "\t--keepVolume=v: set the keep volume to use to define valid cells[default 1]\n";
#else
    o << "\t--keepVolume=v: set the keep volume to use to define valid cells[default 0.001]\n";
#endif
    o << "\t--keepDist2=d: set the last keep distance[default 0]\n";
    o << "\t\t 0: keep used cells\n";
    o << "\t\t 1: keep used cells and 1 bordering cells\n";
    o << "\t\t 2: keep used cells and 2 bordering cells\n";
    o << "\t\t ...\n";
    o << "\t\t -1: keep all cells\n";
    o << "\n";
    o << "\t--grid=FILE: set the grid input file to FILE\n";
    o << "\t\t if set, this defines the volum's domain.\n";
    o << "\t--checkGrid: checks if the grid if valid\n";
    o << "\t--hessian: compute the hessian value\n";
    o << "\n";
    o << "\n-------- output -----------\n";
    o << "\t--output=XXX: add an output to be generated with XXX in\n";
    o << "\t\t xyw: drop the result in grid[CG]PU.xyz file\n";
    o << "\t\t geogram: store the result in grid[CG]PU.geogram_ascii file\n";
    o << "\t\t volume: compute the result volumes and ouput it on stdout\n";
    o << "\t\t mg: drop for each initial seeds the mg and m the result in grid[CG]PU_mg.xyzw file\n";
    o << "\t--showAllDevices: show all available openCL devices\n";

    o << "\n-------- debugging/tuning -----------\n";
    o << "\t--debug: print a lot of informations(times, ...)\n";
    o << "\t--enableProfiling: enable openCL profiling\n";
#ifndef USE_DOUBLE    
    o << "\t--cVoroIsInHalfSpaceAlgo=algo: use specific algo for convex cell half space function[default 2]\n";
    o << "\t\t 0: real without error verifications\n"; 
    o << "\t\t 1: real with verifications\n"; 
    o << "\t\t 2: double without error verifications[default]\n";
#endif
    o << "\t--cVoroDebugCounterMax: allocate debug counters to compute the maximum number of K,P,V,T in cVoro algorithm\n";
    o << "\t--cVoroDebugCounterHisto: allocate debug counters to compute the histograms of K,P,V in cVoro algorithm\n";
    o << "\t--cVoroDebugCounterStatus: allocate status counters\n";
    o << "\t--printInvariants: displays the final volume and the barycenters at the begin and end of the simulation\n";
    o << "\n";
}

void Options::initVoroAlgoParameters(VoroAlgo &algo) const
{
    // kn
    algo.m_K = m_K;
    algo.m_maxKComputed = m_maxKComputed;
    // voro
#ifndef USE_DOUBLE
    algo.m_vhsAlgo[0] = m_cVoroIsInHalfSpaceKernel;
    algo.m_vhsAlgo[1] = 2;
#endif
    algo.m_P = m_P;
    algo.m_V = m_V;
    algo.m_T = m_T;
    algo.m_PGlobalPercent = m_PSlowPercent;
    
    // hessian
    algo.m_H = m_computeHessian ? m_H : 0;
    algo.m_HC = m_HC;
    // result
    algo.m_keepDistance = m_keepDistance;
    algo.m_keepVolume = m_keepVolume;
    // others
    algo.m_enableDCountersHistoK = algo.m_enableDCountersHistoP = algo.m_enableDCountersHistoT = algo.m_enableDCountersHistoV
        = m_cVoroDebugCounterHisto;
    algo.m_enableDCountersMax = m_cVoroDebugCounterMax;
    algo.m_enableDCountersStatus = m_cVoroDebugCounterStatus;
    algo.m_enableCLProfiling = m_enableProfiling;
}

Options &getOptions()
{
    static Options s_options;
    return s_options;
}

/* vim:set shiftwidth=4 softtabstop=4 noexpandtab: */
