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

#include <cmath>
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

#include "geometry.h"

struct VoxelGrid {
  VoxelGrid(GEO::index_t nb_voxels_per_dimension) {
    size = nb_voxels_per_dimension;
    voxel_size = float(1000. / double(size));
    in_domain.resize(nb_voxels());
  }

  GEO::vec3 voxel_corner_pos(GEO::index_t id) { return voxel_size*GEO::vec3(id%size, (id / size) % size, id / (size*size)); }
  GEO::vec3 voxel_center_pos(GEO::index_t id) { return voxel_size*GEO::vec3(0.5 + double(id%size), 0.5 + double((id / size) % size), 0.5 + double(id / (size*size))); }
  GEO::index_t id(GEO::index_t i, GEO::index_t  j, GEO::index_t  k) { return k*size*size + j*size + i;}
  GEO::index_t nb_voxels() { return size*size*size; }
  
  void save(std::string const &filename) {
    std::ofstream f;
    f.open(filename);
    f.precision(20);
    f << size << "\n";
    for (GEO::index_t i=0; i<nb_voxels(); ++i) f << in_domain[i] << " "; f << "\n";
    f << T_pts.size() << "\n";
    for (GEO::index_t i=0; i<T_pts.size(); ++i) f << T_pts[i] << " "; f << "\n";
    f << T_triplets.size() << "\n";
    for (GEO::index_t i=0; i<T_triplets.size(); ++i) f << T_triplets[i] << " "; f << "\n";
    f << T_indir.size() << "\n";
    for (GEO::index_t i=0; i<T_indir.size(); ++i) f << T_indir[i] << " "; f << "\n";
    f << T_offset.size() << "\n";
    for (GEO::index_t i=0; i<T_offset.size(); ++i) f << T_offset[i] << " "; f << "\n";
    f.close();
  }

  // data
  GEO::index_t size ;
  GEO::vector<char> in_domain;
  GEO::vector<GEO::vec3> T_pts;
  GEO::vector<GEO::index_t> T_triplets;
  GEO::vector<GEO::index_t> T_indir;
  GEO::vector<GEO::index_t> T_offset;

  // precomputed 
  float voxel_size;
};

int main(int argc, char** argv) {
  using namespace GEO;
  
  GEO::initialize();
  
  try {
    
    Stopwatch Wtot("Total time");

    std::vector<std::string> filenames;
    
    CmdLine::import_arg_group("standard");
    CmdLine::import_arg_group("algo");
    CmdLine::declare_arg("division", 75, "numbers of division by axis");
    CmdLine::declare_arg("grid", "", "volume grid(output)");
    CmdLine::declare_arg("output", "", "modified volume");
    if(!CmdLine::parse(argc, argv, filenames, "meshfile") ) {
      return 1;
    }
    if(filenames.size() != 1) {
	    std::cout << "Error:" << argv[0] << "meshfile\n";
	    return 1;
    }
    index_t num_divide=75;
    if(CmdLine::get_arg_int("division") > 0)
      num_divide=index_t(CmdLine::get_arg_int("division"));

    std::string mesh_filename = filenames[0];
    std::string grid_filename = CmdLine::get_arg("grid");
    std::string modif_filename = CmdLine::get_arg("output");
    if (!grid_filename.empty())
      Logger::out("I/O") << "Grid = " << grid_filename << std::endl;
    if (!modif_filename.empty())
      Logger::out("I/O") << "Modif = " << modif_filename << std::endl;
    
    Logger::div("Loading data");

    Mesh M_in, grid_in;
    Mesh M_out;

    // load mesh
    {
      MeshIOFlags flags;
	    flags.set_element(MESH_CELLS);
      if(!mesh_load(mesh_filename, M_in, flags))
        return 1;
    }
    // code to reproject point on a sub grid ( retrieved from OGF::MeshGrobGlobalParamCommands::export_notet_domain_definition )
    M_in.cells.clear();
    M_in.vertices.remove_isolated();
    M_in.facets.triangulate();
    
    if (1) {
      double min[3], max[3];
      get_bbox(M_in, min, max);
      double maxdim = 0;
      for (int i=0; i<3; ++i) maxdim = std::max(maxdim, max[i]-min[i]);
      for (index_t v=0; v<M_in.vertices.nb(); ++v)
        M_in.vertices.point(v)=(990.222/ maxdim) *(M_in.vertices.point(v) - vec3(min[0],min[1],min[2])) + vec3(4.998, 4.998, 4.998);
    }

    if (!modif_filename.empty()) {
      Logger::div("Saving modified data");
      mesh_save(M_in, modif_filename);
    }

    if (!grid_filename.empty()) {
      Logger::div("Creating grid");
      VoxelGrid grid(num_divide);
      for (index_t v=0; v<M_in.vertices.nb(); ++v)
        grid.T_pts.push_back(M_in.vertices.point(v));
      for (index_t f=0; f<M_in.facets.nb(); ++f) {
        for (index_t lv=0; lv<3; ++lv) 
          grid.T_triplets.push_back(M_in.facets.vertex(f,lv));
      }
      
      std::cerr << "generate list of triangles per voxel\n";
      vector<vector<index_t> > voxel_to_tri_id(grid.nb_voxels());
      for (index_t f=0; f<M_in.facets.nb(); ++f) {
        GEO::vec3 boxMin, boxMax;
        for(index_t c = 0; c < 3; c++) {
          boxMin[c] = Numeric::max_float64();
          boxMax[c] = Numeric::min_float64();
        }
        for(index_t lv=0; lv<3; ++lv) {
          const double* p = M_in.vertices.point_ptr(M_in.facets.vertex(f,lv));
          for(index_t c = 0; c < 3; c++) {
            boxMin[c] = std::min(boxMin[c], p[c]);
            boxMax[c] = std::max(boxMax[c], p[c]);
          }
        }

        boxMin = double(grid.size)* boxMin / 1000.;
        boxMax = double(grid.size)* boxMax / 1000.;
        for (index_t i = index_t(std::floor(boxMin[0])); i <= index_t(std::ceil(boxMax[0])); i++) {
          if (i>=grid.size) continue;
          for (index_t j = index_t(std::floor(boxMin[1])); j <= index_t(std::ceil(boxMax[1])); j++) {
            if (j>=grid.size) continue;
            for (index_t k = index_t(std::floor(boxMin[2])); k <= index_t(std::ceil(boxMax[2])); k++) {
              if (k>=grid.size) continue;
              index_t id = grid.id(i, j, k);
              float center[3];
              for (index_t d=0; d<3; ++d) center[d] = float(grid.voxel_center_pos(id)[d]);
              float halfbox_size[3] = { 0.5f*grid.voxel_size , 0.5f*grid.voxel_size , 0.5f*grid.voxel_size };
              vec3 trivert[3];
              for (index_t lv=0; lv<3; ++lv)
                trivert[lv] = M_in.vertices.point(M_in.facets.vertex(f, lv));
              if (triBoxOverlap(center, halfbox_size, trivert))
                voxel_to_tri_id[id].push_back(f);
            }
          }
        }
      }
      
      grid.T_offset.push_back(0);
      for (index_t i=0; i<grid.nb_voxels(); ++i) {
        for(index_t lf=0; lf<voxel_to_tri_id[i].size(); ++lf)
          grid.T_indir.push_back(voxel_to_tri_id[i][lf]);
        grid.T_offset.push_back(grid.T_indir.size());
      }
    
      std::cerr << "Compute points step\n";
      // compute points
      for (index_t i=0; i<grid.nb_voxels(); ++i)
        grid.in_domain[i] = 0;
      mesh_tetrahedralize(M_in, false, true, 1.);
      for(index_t c=0; c<M_in.cells.nb(); ++c) {
        vec3 boxMin, boxMax;
        for(index_t ce = 0; ce < 3; ce++) {
          boxMin[ce] = Numeric::max_float64();
          boxMax[ce] = Numeric::min_float64();
        }
        for(index_t lv=0; lv<4; ++lv) {
          const double* p = M_in.vertices.point_ptr(M_in.cells.vertex(c,lv));
          for(index_t ce = 0; ce < 3; ce++) {
            boxMin[ce] = std::min(boxMin[ce], p[ce]);
            boxMax[ce] = std::max(boxMax[ce], p[ce]);
          }
        }
        boxMin = double(grid.size)* boxMin / 1000.;
        boxMax = double(grid.size)* boxMax / 1000.;

        for (index_t i = index_t(floor(boxMin[0])); i <= index_t(ceil(boxMax[0])); i++) {
          if (i>=grid.size) continue;
          for (index_t j = index_t(floor(boxMin[1])); j <= index_t(ceil(boxMax[1])); j++) {
            if (j>=grid.size) continue;
            for (index_t k = index_t(floor(boxMin[2])); k <= index_t(ceil(boxMax[2])); k++) {
              if (k>=grid.size) continue;
              index_t id = grid.id(i, j, k);
              bool all_inside = true;
              bool one_outside = false;

              for(index_t f=0; f<4; ++f) {
                vec3 P[3];
                for(index_t lv=0; lv<3; ++lv) P[lv] = M_in.vertices.point(M_in.cells.facet_vertex(c, f, lv));
                vec3 n = cross(P[2] - P[0], P[1] - P[0]);
                n = normalize(n);
                double eps = .001;
                double signed_dist = dot(grid.voxel_corner_pos(id) - P[0], n);
                all_inside = all_inside && (signed_dist < -eps);
                one_outside = one_outside || (signed_dist > eps);
              }
              if (all_inside) grid.in_domain[id] = 2;
              else if (!one_outside && grid.in_domain[id] != 2) grid.in_domain[id] = 1;
            }
          }
        }
      }

      grid.save(grid_filename);
    }
  }
  catch(const std::exception& e) {
    std::cerr << "Received an exception: " << e.what() << std::endl;
    return 1;
  }
  
  Logger::out("") << "Everything OK, Returning status 0" << std::endl;
  return 0;
}

