#pragma once

#include <iostream>
#if defined(__linux__) || defined(__APPLE__)
#   include <sys/times.h>
#elif defined(WIN32) || defined(_WIN64)
#   include <windows.h>
#endif

class Stopwatch {
public:
    Stopwatch(const char* taskname, bool on=true) : taskname_(taskname), m_on(on), start_(now()) {
        if (!m_on) return;
        std::cerr << taskname_ << ":" << std::endl;
    }

    ~Stopwatch() {
        if (!m_on) return;
        double elapsed = now() - start_;
        std::cerr << taskname_ << ": " << elapsed << "s" << std::endl;	
    }

    double timeFromStart() {
        return now() - start_;
    }
    static double now() {
#if defined(__linux__) || defined(__APPLE__)
        tms now_tms;
        return double(times(&now_tms)) / 100.0;
#elif defined(WIN32) || defined(_WIN64)
        return double(GetTickCount()) / 1000.0;
#else
        return 0.0;
#endif
    }

//private:
    const char* taskname_;
    bool m_on;
	double start_;
};

