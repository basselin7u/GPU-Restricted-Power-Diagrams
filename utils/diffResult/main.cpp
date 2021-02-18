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
#include <iostream>
#include <vector>

#include "Utils.h"

int main(int argc, char** argv) {
  std::vector<double> res[2];
  if (argc!=3 || !Utils::loadResFile(argv[1],res[0]) || !Utils::loadResFile(argv[2],res[1])) {
    std::cerr << "Error:" << argv[0] << " res1.xyzw res2.xyzw\n";
  }
  if (res[0].size() != res[1].size() || (res[0].size()%4)!=0) {
    std::cerr << "Error:" << argv[0] << " files " << argv[1] << " and " << argv[2] << " do not contain the same numbers of data\n";
    return 1;
  }
  size_t numPts=res[0].size()/4;
#if 0
  // if a list of result is in form x,y,z,m, transforms it to  mx, my, mz, m
  for (size_t i=0; i<numPts; ++i) {
    for (size_t j=0; j<3; ++j)
      res[1][4*i+j]*=res[1][4*i+3];
  }
#endif
  double (bary[2])[4]={{0,0,0,0}, {0,0,0,0}};
  for (int st=0; st<2; ++st) {
    for (size_t i=0; i<numPts; ++i) {
      for (size_t j=0; j<4; ++j)
        bary[st][j]+=res[st][4*i+j];
    }
  }
  std::cout.precision(20);
  std::cout << "------------------- intrinsic (sum) --------------------------\n";
  std::cout << "Volume : " << bary[0][3] << "\t" << bary[1][3] << "\t" << (bary[1][3]-bary[0][3]) << "\n";
  std::cout << "Barycenter : " << bary[0][0]/bary[0][3] << "x" << bary[0][1]/bary[0][3] << "x" << bary[0][2]/bary[0][3] << "\t"
            << bary[1][0]/bary[1][3] << "x" << bary[1][1]/bary[1][3] << "x" << bary[1][2]/bary[1][3] << "\t" 
            << (bary[1][0]/bary[1][3]-bary[0][0]/bary[0][3]) << "x" << (bary[1][1]/bary[1][3]-bary[0][1]/bary[0][3]) << "x" << (bary[1][2]/bary[1][3]-bary[0][2]/bary[0][3]) << "\n";

  double maxDiff[4]={0,0,0,0}, maxMDiff[3]={0,0,0};
  double aveDiff[4]={0,0,0,0}; // sum of the differences used with numDiff to compute the average
  double maveDiff[3]={0,0,0};  // sum of the mx,my,mz used with numDiff to compute the average of mx,my,mz
  size_t numDiff=0;
  for (size_t i=0; i<numPts; ++i) {
    double diff=std::abs(res[1][4*i+3]-res[0][4*i+3]);
    if (diff>maxDiff[3]) maxDiff[3]=diff;
    if (res[0][4*i+3]<=0 || res[1][4*i+3]<=0) continue;
    ++numDiff;
    aveDiff[3]+=diff;
    for (size_t j=0; j<3; ++j) {
      diff=std::abs(res[1][4*i+j]/res[1][4*i+3]-res[0][4*i+j]/res[0][4*i+3]);
      aveDiff[j]+=diff;
      if (diff>maxDiff[j]) maxDiff[j]=diff;
      diff=std::abs(res[1][4*i+j]-res[0][4*i+j]);
      maveDiff[j]+=diff;
      if (diff>maxMDiff[j]) maxMDiff[j]=diff;
    }
  }
  std::cout << "------------------- maximal difference by cells--------------------------\n";
  std::cout << "Volume : " << maxDiff[3] << "\n";
  std::cout << "Barycenter : " << maxDiff[0] << "x" << maxDiff[1] << "x" << maxDiff[2] << "\n";
  std::cout << "Barycenter x volume : " << maxMDiff[0] << "x" << maxMDiff[1] << "x" << maxMDiff[2] << "\n";
  std::cout << "------------------- average difference by cells--------------------------\n";
  std::cout << "Volume : " << aveDiff[3]/numDiff << "\n";
  std::cout << "Barycenter : " << aveDiff[0]/numDiff << "x" << aveDiff[1]/numDiff << "x" << aveDiff[2]/numDiff << "\n";
  std::cout << "Barycenter x volume : " << maveDiff[0]/numDiff << "x" << maveDiff[1]/numDiff << "x" << maveDiff[2]/numDiff << "\n";
  return 0;
}

