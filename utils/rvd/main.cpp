/* -*- Mode: C++; c-default-style: "k&r"; indent-tabs-mode: nil; tab-width: 2; c-basic-offset: 2 -*- */

/*
 * GEOGRAM example program:
 * compute Restricted Voronoi diagrams, i.e.
 * intersection between a Voronoi diagram and a
 * tetrahedral mesh (or a triangulated mesh).
 */ 

/*
 *  Copyright (c) 2012-2014, Bruno Levy
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *  this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *  this list of conditions and the following disclaimer in the documentation
 *  and/or other materials provided with the distribution.
 *  * Neither the name of the ALICE Project-Team nor the names of its
 *  contributors may be used to endorse or promote products derived from this
 *  software without specific prior written permission.
 * 
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 *  If you modify this software, you should include a notice giving the
 *  name of the person performing the modification, the date of modification,
 *  and the reason for such modification.
 *
 *  Contact: Bruno Levy
 *
 *     Bruno.Levy@inria.fr
 *     http://www.loria.fr/~levy
 *
 *     ALICE Project
 *     LORIA, INRIA Lorraine, 
 *     Campus Scientifique, BP 239
 *     54506 VANDOEUVRE LES NANCY CEDEX 
 *     FRANCE
 *
 */

#include <iomanip>
#include <fstream>
#include <vector>

#include <geogram/basic/common.h>
#include <geogram/basic/logger.h>
#include <geogram/basic/command_line.h>
#include <geogram/basic/command_line_args.h>
#include <geogram/basic/stopwatch.h>
#include <geogram/basic/file_system.h>
#include <geogram/basic/process.h>
#include <geogram/mesh/mesh.h>
#include <geogram/mesh/mesh_geometry.h>
#include <geogram/mesh/mesh_topology.h>
#include <geogram/mesh/mesh_io.h>
#include <geogram/mesh/mesh_repair.h>
#include <geogram/mesh/mesh_fill_holes.h>
#include <geogram/mesh/mesh_preprocessing.h>
#include <geogram/mesh/mesh_degree3_vertices.h>
#include <geogram/mesh/mesh_tetrahedralize.h>
#include <geogram/delaunay/delaunay.h>
#include <geogram/voronoi/RVD.h>
#include <geogram/voronoi/RVD_callback.h>
#include <geogram/voronoi/RVD_mesh_builder.h>
#include <geogram/numerics/predicates.h>

#include "Utils.h"

int main(int argc, char** argv) {
  using namespace GEO;
  
  GEO::initialize();
  
  try {
    
    Stopwatch Wtot("Total time");

    std::vector<std::string> filenames;
    
    CmdLine::import_arg_group("standard");
    CmdLine::import_arg_group("algo");
    CmdLine::declare_arg("input", "", "volume mesh");
    CmdLine::declare_arg("points", "", "points (.xyz)");
    CmdLine::declare_arg("output", "", "output file (.xyzw): mg, m");
    if(!CmdLine::parse(argc, argv, filenames, "") || !filenames.empty()) {
	    std::cout << "Syntax error: try " << argv[0] << " -h\n";
      return 1;
    }

    std::string mesh_filename=CmdLine::get_arg("input");
    if (mesh_filename.empty()) {
	    std::cout << "Error:" << argv[0] << " can not find the input file\n";
	    return 1;
    }
    std::string points_filename=CmdLine::get_arg("points");
    if (points_filename.empty()) {
	    std::cout << "Error:" << argv[0] << " can not find the vertices\n";
	    return 1;
    }
    std::string output_filename=CmdLine::get_arg("output");

    if (!output_filename.empty()) 
      Logger::out("I/O") << "Output = " << output_filename << std::endl;
    
    Logger::div("Loading data");

    Mesh M_in, points_in;
    Mesh M_out;

    // load mesh
    {
      MeshIOFlags flags;
	    flags.set_element(MESH_CELLS);
      if(!mesh_load(mesh_filename, M_in, flags))
        return 1;
    }
    if(M_in.cells.nb() == 0) {
	    Logger::out("RVD") << "Mesh does not have tetrahedra, tetrahedralizing"
                         << std::endl;
	    mesh_tetrahedralize(M_in);
    }

    // compute invariant
    std::cout << "Volume=" << std::setprecision(20) << mesh_cells_volume(M_in) << "\n";
    if (1) {
      double bary[4]={0,0,0,0};
      
      for(index_t c: M_in.cells) {
        const double* p0 = M_in.vertices.point_ptr(M_in.cells.vertex(c,0));
        const double* p1 = M_in.vertices.point_ptr(M_in.cells.vertex(c,1));
        const double* p2 = M_in.vertices.point_ptr(M_in.cells.vertex(c,2));
        const double* p3 = M_in.vertices.point_ptr(M_in.cells.vertex(c,3));
        double cVol_4 = mesh_cell_volume(M_in,c)/4;
        for (size_t j=0; j<3; ++j)
          bary[j]+=cVol_4*(p0[j]+p1[j]+p2[j]+p3[j]);
        bary[3]+=4*cVol_4;
      }
      std::cerr << "Barycenter=" << std::setprecision(20) << bary[0]/bary[3] << "x"
                << std::setprecision(20) << bary[1]/bary[3] << "x"
                << std::setprecision(20) << bary[2]/bary[3] << std::endl;
    }

    // load point
    std::vector<float> pts;
    if (!Utils::loadSeedsFile(points_filename.c_str(), pts, false, false) || pts.size()<6) {
      std::cerr << argv[0] << ": could not load file " << points_filename << std::endl;
      return 1;
    }

    size_t numPts=pts.size()/4;
    Delaunay_var delaunay = Delaunay::create(3);
    std::vector<double> dPts(3*numPts);
    for (size_t i=0; i<numPts; ++i) { // float xyzw => double xyz
      for (size_t j=0; j<3; ++j) dPts[3*i+j]=double(pts[4*i+j]);
    }

    // create RVD
    RestrictedVoronoiDiagram_var RVD = RestrictedVoronoiDiagram::create(delaunay, &M_in);
    {
      Stopwatch W("Delaunay");
      delaunay->set_vertices(index_t(numPts), dPts.data());
    }
    RVD->set_volumetric(true);

    // compute mg and m
    Logger::div("Restricted Voronoi Diagram");
    std::vector<double> mg(3*numPts), m(numPts,0);
    RVD->compute_centroids_in_volume(mg.data(), m.data());

    // check invariant
    double vol=0;
    for (auto &v : m)
      vol+=v;
    std::cerr << "Volume des cellules:" << std::setprecision(20) << vol << "\n";
    if (vol>0) {
      double bary[3]={0,0,0};
      for (size_t i=0; i<numPts; ++i) {
        for (size_t j=0; j<3; ++j)
          bary[j]+=mg[3*i+j];
      }
      std::cerr << "Cells barycenter=" << std::setprecision(20) << bary[0]/vol << "x"
                << std::setprecision(20) << bary[1]/vol << "x"
                << std::setprecision(20) << bary[2]/vol << std::endl;

    }

    // save mgx, mgy, mgz, m
    if (!output_filename.empty()) {
      Logger::div("Save");
      std::ofstream ofs(output_filename, std::ofstream::out);
      if (!ofs.is_open()) {
        std::cerr << argv[0] << ": could not create file " << output_filename << std::endl;
        return 1;
      }
      ofs << numPts << "\n";
      for (size_t i=0; i<numPts; ++i) {
        ofs << std::setprecision(20) << mg[3*i+0] << " "
            << std::setprecision(20) << mg[3*i+1] << " "
            << std::setprecision(20) << mg[3*i+2] << " "
            << std::setprecision(20) << m[i] << "\n";
      }
    }
  }
  catch(const std::exception& e) {
    std::cerr << "Received an exception: " << e.what() << std::endl;
    return 1;
  }
  
  Logger::out("") << "Everything OK, Returning status 0" << std::endl;
  return 0;
}

