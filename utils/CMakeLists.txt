##############################################################################
# Graphite/// root CMakeList
##############################################################################

# CMake 2.8.11 is required for 2 reasons:
# - it is the first version that fully supports the specification of Visual
# Studio toolsets (v110_xp).
# - it is the version that supports the command string(TIMESTAMP ...)
cmake_minimum_required(VERSION 2.15.0)

project(RVD)
include(${CMAKE_SOURCE_DIR}/cmake/rvd.cmake)

##############################################################################

# If there is a bundled Geogram or Vorpaline distribution, compile it as well.
if(IS_DIRECTORY ${CMAKE_SOURCE_DIR}/geogram)
   add_subdirectory(geogram)
endif()

add_subdirectory(rvd/)
add_subdirectory(diffResult/)
add_subdirectory(createGrid/)

