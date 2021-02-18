#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

#include "basic.h"
#include "cvoro_config.h"
#include "Grid.h"
#include "knearests.h"
#include "openCL.h"
#include "Options.h"
#include "Utils.h"
#include "voronoi.h"

/*
  \param[in] seeds (4*numSeeds), the positions of each seed
  \param[in] context, device, queue: the openCL context, device, queue
  \param[out] bary (4*numseeds), the position and volume of each cells,
  \param[out] hessian (0 or option.m_H*numseeds), the area of each bordering surface
  \param[out] hessianId (0 or option.m_H*numseeds), the id of the corresponding neighbour
  \param[in] grid: if nullptr, do the computation on [-0.1, 1000.1]^3,
                   if not, do the computation on the intersection of the grid volume and this box
  \note if compressResult is set (and a grid is defined), the unused seeds are removed
 */
static bool computeVoroDiagram(std::shared_ptr<OpenCLContext> context, std::vector<real> const &seeds,
                                      std::vector<real> &bary, std::vector<real> &hessian, std::vector<int> &hessianId, std::vector<int> &permutation, Grid const *grid)
{
    if (!context) return false;
    
    Options const &options=getOptions();
    VoroAlgo algo(context, options.m_debug);
    options.initVoroAlgoParameters(algo);
    algo.m_keepDistance=options.m_keepDistanceLast; // we want the final keep distance
    
    algo.setInputPoints(seeds);
    if (grid) algo.setGrid(*grid);
    std::vector<VoroAlgoStat> stats;

    bool res=algo.launch([&bary, &hessian, &hessianId, &permutation](VoroAlgo const &alg) {
            bary=alg.getBarycenters();
            permutation  = alg.getPermutationToUser();
            if (alg.m_H>0) {
                hessian=alg.getHessians();
                hessianId=alg.getHessianIds();
            }
            return true;
        }, stats);

    // print the statistic
    std::cout << "GTime[\"stat\"] = [";
    for (size_t i=0; i<stats.size(); i++) {
         stats[i].pythonPrint(std::cout);
         if (i<stats.size()-1) std::cout << ",\n";
    }
    std::cout << "]\n";

    return res;
}

int main(int argc, char** argv) {
    Options &options=getOptions();
    
    //initialize_geogram();
    int argn=options.parseOptions(argc, argv);
    if (options.m_showCLDevices) {
        OpenCLContext::printAllDevices(std::cerr);
        if (argn==argc && !options.m_showHelp)
            return 0;
    }
    if (options.m_showHelp || argn+1!=argc) {
        std::cerr << "Usage: " << argv[0] << " [options] points.xyz" << std::endl;
        std::cerr << std::endl;
        options.showOptions(std::cerr);
        return 1;
    }

    std::vector<real> pts;
    if (options.m_loadBinaryPoints) {
        if (!Utils::readBinarySeedsFile(argv[argn], pts))
            return 1;
    }
    else {
        if (!Utils::loadSeedsFile(argv[argn], pts, getOptions().m_points4D, false)) {
            std::cerr << argv[0] << ": could not load file " << argv[argn] << std::endl;
            return 1;
        }
    }
    if (!options.m_saveBinaryPoints.empty())
        Utils::saveBinarySeedsFile(options.m_saveBinaryPoints.c_str(), pts);

    real epsilon = options.m_keepVolume;
    if (!options.m_CPU && !options.m_GPU) {
        std::cerr << argv[0] << ": no device is defined\n";
        std::cerr << "\t forces computation on the GPU\n";
        options.m_GPU=true;
    }
    if (options.m_CPU || options.m_GPU) {
        Grid grid(options.m_debug);
        bool hasGRID=!options.m_gridFile.empty() && grid.load(options.m_gridFile.c_str());
        if (hasGRID && options.m_checkGrid && !grid.valid())
            std::cerr << argv[0] << ": the grid is invalid." << std::endl;
        if (options.m_checkInvariants) {
            if (hasGRID)
                grid.printInvariants();
            else {
                std::cerr << "Grid::load: volume=" << 1000*1000*1000 << std::endl;
                std::cerr << "Grid::load: bary=" << 500 << "x" << 500 << "x" << 500 << std::endl;
            }
        }
        for (int step=0; step<2; ++step) {
            if ((step==0 ? options.m_CPU : options.m_GPU)==false) continue;
            auto context=std::make_shared<OpenCLContext>(options.m_debug);
            if (!context->createContext(step==1) || !context->hasContext()) return 1;
            context->setCacheDirectory(step==0 ? options.m_CPUDirectory : options.m_GPUDirectory);
            std::vector<real> bary; // numSeed: volume of each voro cell
            std::vector<real> hessian; // if option.m_computeHessian, numSeed*option.m_H area of each bordering surface
            std::vector<int> hessidx; // if option.m_computeHessian, numSeed*option.m_H id of each bordering surface
            std::string name(step==0 ? "gridCPU" : "gridGPU");



            std::vector<int> permutation;
            computeVoroDiagram(context, pts, bary, hessian, hessidx, permutation, hasGRID ? &grid : nullptr);

            if (options.m_checkInvariants) {
                size_t nb_pts = bary.size()/4;
                double volume = 0;
                double minv = 1e20f;
                double maxv = -1e20f;
#pragma omp parallel for reduction(min:minv) reduction(max:maxv) reduction(+:volume)
                for (size_t i=0; i<nb_pts; i++) {
                    if (bary[i*4+3]<=epsilon) continue;
                    volume += bary[i*4 + 3];
                    if (bary[i*4 + 3] > maxv) maxv = bary[i*4 + 3];
                    if (bary[i*4 + 3] < minv) minv = bary[i*4 + 3];
                }

                std::cerr << "Sum of volumes: " << std::setprecision(20) << volume << " minvol: " << minv << " maxvol: " << maxv << std::endl;
                
                if (volume>0) {
                    double barycenter[3] = {0,0,0};
                    for (size_t i=0; i<nb_pts; i++) {
                        if (bary[i*4+3]<=NO_CELL_LIMITS) continue;
                        for (size_t j=0; j<3; ++j)
                            barycenter[j]+=bary[i*4+j];
                    }
                    std::cerr << "Cells barycenter=" << std::setprecision(20) << barycenter[0]/volume << "x"
                              << std::setprecision(20) << barycenter[1]/volume << "x"
                              << std::setprecision(20) << barycenter[2]/volume << std::endl;
                }
            }
            
            for (auto const &s : options.m_outputList ) {
                if (s=="xyz")
                    Utils::dropXYZFile(bary, true, (name+".xyz").c_str(),
                                       [&bary, epsilon](size_t i) { return bary[4*i+3] > NO_CELL_LIMITS && std::abs(bary[4 * i + 3]) >= epsilon; });
                else if (s=="geogram")
                    Utils::dropXYZGeogram(bary, true, (name+".geogram_ascii").c_str(),
                                          [](size_t /*i*/) { return true; },
                                          /* or [&bary, epsilon](size_t i) { returns bary[4*i+3] > NO_CELL_LIMITS && std::abs(bary[4 * i + 3]) >= epsilon; }
                                             if you only want usefull cell */
                                          { {"w", [&bary](size_t i) { return bary[4*i+3]; } }});

                else if (s=="volume") {
                    if (!options.m_checkInvariants) {
                        size_t nb_pts = bary.size()/4;
                        double volume = 0;
                        double minv = 1e20f;
                        double maxv = -1e20f;
#pragma omp parallel for reduction(min:minv) reduction(max:maxv) reduction(+:volume)
                        for (size_t i=0; i<nb_pts; i++) {
                            if (bary[i*4+3]<=epsilon) continue;
                            volume += bary[i*4 + 3];
                            if (bary[i*4 + 3] > maxv) maxv = bary[i*4 + 3];
                            if (bary[i*4 + 3] < minv) minv = bary[i*4 + 3];
                        }

                        std::cerr << "Sum of volumes: " << std::setprecision(20) << volume << " minvol: " << minv << " maxvol: " << maxv << std::endl;
                    }
                }
                else if (s=="mg") {
                    std::ofstream ofs(name+"_mg.xyzw", std::ofstream::out);
                    if (!ofs.is_open()) {
                        std::cerr << argv[0] << ": could not create file " << name << "_mg.xyzw" << std::endl;
                    }
                    else {
                        ofs << pts.size()/4 << "\n";
                        std::map<size_t,size_t> origToPos;
                        for (size_t i=0; i<permutation.size(); ++i) origToPos[permutation[i]]=i;
                        for (size_t i=0; i<pts.size()/4; ++i) {
                            auto it=origToPos.find(i);
                            if (it==origToPos.end()) {
                                ofs << "0 0 0 0\n";
                                continue;
                            }
                            size_t pt=it->second;
                            if (bary[4*pt+3]<=epsilon)
                                ofs << "0 0 0 0\n";
                            else
                                ofs << std::setprecision(20) << bary[4*pt+0] << " "
                                    << std::setprecision(20) << bary[4*pt+1] << " "
                                    << std::setprecision(20) << bary[4*pt+2] << " "
                                    << std::setprecision(20) << bary[4*pt+3] << "\n";
                        }
                    }
                }
                else {
                    std::cerr << argv[0] << ": ignored output : " << s << "\n";
                }
            }
        }
    }
    else if (!options.m_saveBinaryPoints.empty()) {
        std::cerr << argv[0] << ": no options --CPU or --GPU" << std::endl;
        return 1;
    }
    return 0;
}
