/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */
#ifndef OPTIONS_H
#define OPTIONS_H

#include <iostream>
#include <map>
#include <string>
#include <vector>

class UsageHelper;

class VoroAlgo;

class Options
{
public:
	Options();
	virtual ~Options();
    Options(Options const &)=delete;
    Options &operator=(Options const &)=delete;

	int parseOptions(int argc, char const *const *argv);
    virtual void showOptions(std::ostream &o) const;

    void initVoroAlgoParameters(VoroAlgo &algo) const;
    
	bool m_showHelp;

    int m_H;
    int m_HC;
    int m_K;
    int m_P;
    int m_T;
    int m_V;
    int m_maxKComputed;
    real m_PSlowPercent;

    bool m_CPU;
    std::string m_CPUDirectory;
    bool m_GPU;
    std::string m_GPUDirectory;
    
    int m_keepDistance;
    real m_keepVolume;
    int m_keepDistanceLast;

    bool m_computeHessian;
    bool m_debug;
    
    std::string m_gridFile;
    bool m_checkGrid;

    bool m_points4D;
    bool m_loadBinaryPoints;
    std::string m_saveBinaryPoints;

    bool m_enableProfiling;
#ifndef USE_DOUBLE
    int m_cVoroIsInHalfSpaceKernel;
#endif
    bool m_cVoroDebugCounterHisto;
    bool m_cVoroDebugCounterMax;
    bool m_cVoroDebugCounterStatus;
    bool m_checkInvariants;

    std::vector<std::string> m_outputList;
    bool m_showCLDevices;
protected:
    bool parseConfigurationFile(char const *configName);
	virtual bool parseAnOption(int &i, int argc, char const *const *argv);

    std::map<std::string, bool *> m_boolArgsMap;
    std::map<std::string, int *> m_intArgsMap;
    std::map<std::string, std::string *> m_stringArgsMap;
};

Options &getOptions();
#endif

/* vim:set shiftwidth=4 softtabstop=4 noexpandtab: */
