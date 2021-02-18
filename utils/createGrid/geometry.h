/*
 *  OGF/Graphite: Geometry and Graphics Programming Library + Utilities
 *  Copyright (C) 2000-2015 INRIA - Project ALICE
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 *
 *  If you modify this software, you should include a notice giving the
 *  name of the person performing the modification, the date of modification,
 *  and the reason for such modification.
 *
 *  Contact for Graphite: Bruno Levy - Bruno.Levy@inria.fr
 *  Contact for this Plugin: Nicolas Ray - nicolas.ray@inria.fr
 *
 *     Project ALICE
 *     LORIA, INRIA Lorraine,
 *     Campus Scientifique, BP 239
 *     54506 VANDOEUVRE LES NANCY CEDEX
 *     FRANCE
 *
 *  Note that the GNU General Public License does not permit incorporating
 *  the Software into proprietary programs.
 *
 * As an exception to the GPL, Graphite can be linked with the following
 * (non-GPL) libraries:
 *     Qt, tetgen, SuperLU, WildMagic and CGAL
 */

#pragma once

#include <geogram/basic/geometry.h>

namespace GEO {
    /*******************************************************************************/    

    
    /* Triangle/triangle intersection test routine,
     * by Tomas Moller, 1997.
     * See article "A Fast Triangle-Triangle Intersection Test",
     * Journal of Graphics Tools, 2(2), 1997
     *
     * Updated June 1999: removed the divisions -- a little faster now!
     * Updated October 1999: added {} to CROSS and SUB macros
     *
     * int NoDivTriTriIsect(double V0[3],double V1[3],double V2[3],
     *                      double U0[3],double U1[3],double U2[3])
     *
     * parameters: vertices of triangle 1: V0,V1,V2
     *             vertices of triangle 2: U0,U1,U2
     * result    : returns 1 if the triangles intersect, otherwise 0
     *
     */
    int NoDivTriTriIsect(
                         double V0[3], double V1[3], double V2[3],
                         double U0[3], double U1[3], double U2[3]
                         );

    /********************************************************/
    /* AABB-triangle overlap test code                      */
    /* by Tomas Akenine-Möller                              */
    /* Function: int triBoxOverlap(float boxcenter[3],      */
    /*          float boxhalfsize[3],vec3 triverts[3]); */
    /* History:                                             */
    /*   2001-03-05: released the code in its first version */
    /*   2001-06-18: changed the order of the tests, faster */
    /*                                                      */
    /* Acknowledgement: Many thanks to Pierre Terdiman for  */
    /* suggestions and discussions on how to optimize code. */
    /* Thanks to David Hunt for finding a ">="-bug!         */
    /********************************************************/
    int  triBoxOverlap(
                       float boxcenter[3], float boxhalfsize[3], GEO::vec3 triverts[3]
    );

    /*******************************************************************************/        
    
}
