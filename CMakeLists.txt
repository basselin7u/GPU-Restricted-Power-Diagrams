cmake_minimum_required(VERSION 2.8)

option(WITH_ASAN "toggle to asan support" OFF)

find_package(OpenCL)

if (OpenCL_FOUND)
  include_directories( ${OpenCL_INCLUDE_DIR} )
endif()

find_package(OpenMP)
if (OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
  
option(USE_DOUBLE "do the computation in double" OFF)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -DCL_TARGET_OPENCL_VERSION=120")


configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cvoro_config.h.in ${CMAKE_CURRENT_SOURCE_DIR}/cvoro_config.h)

set(SOURCES_LIB
    Grid.cpp
    knearests.cpp
    openCL.cpp
    voronoi.cpp
)
set(SOURCES_UTILS
    Options.cpp
    Utils.cpp
)
if (WITH_ASAN)
   set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer -fsanitize=address")
endif()
   
if (APPLE)
    # openCL functions are now deprecated
    add_definitions(-Wall -Wextra -Wno-deprecated-declarations)
    # to check all warnings add  -Weverything -Wno-c++98-compat -Wno-deprecated-declarations -Wno-padded -Wno-source-uses-openm
endif()

#
# build test voronoi
#
set(SOURCES_TEST
    test_voronoi.cpp
)
add_executable(test_voronoi ${SOURCES_LIB} ${SOURCES_UTILS} ${SOURCES_TEST})
if (OpenCL_FOUND)
  target_link_libraries(test_voronoi ${OpenCL_LIBRARY} )
endif()



