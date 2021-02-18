#pragma once

// do not modify cvoro_config.h, it is build from cvoro_config.h.in
/* #undef USE_DOUBLE */

#ifndef USE_DOUBLE
#  define real float
#  define real4 cl_float4
#else
#  define real double
#  define real4 cl_double4
#endif

#  define CVORO_STATUS_FILE "/Users/alonso/sources/cVoro/openCLDistrib/Status.h"
#  define CVORO_CONVEX_CELL_FILE "/Users/alonso/sources/cVoro/openCLDistrib/ConvexCell.cl"
#  define CVORO_KNEAREST_FILE "/Users/alonso/sources/cVoro/openCLDistrib/knearests.cl"
#  define CVORO_OPTIONS_FILE "/Users/alonso/sources/cVoro/openCLDistrib/cvoro_options.txt"

