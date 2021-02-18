##############################################################################
# CMake file for rvd
##############################################################################

# CMake 2.8.11 is required for 2 reasons:
# - it is the first version that fully supports the specification of Visual
# Studio toolsets (v110_xp).
# - it is the version that supports the command string(TIMESTAMP ...)
cmake_minimum_required(VERSION 2.8.11)

if("$ENV{GEOGRAM_SOURCE_DIR}" STREQUAL "")
  set(
     GEOGRAM_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../geogram/"
     CACHE PATH "full path to the Geogram (or Vorpaline) installation"
  )
else()
  set(
    GEOGRAM_SOURCE_DIR "$ENV{GEOGRAM_SOURCE_DIR}"
    CACHE PATH "full path to the Geogram (or Vorpaline) installation"
  )
endif()

##############################################################################

# DLLs/shared objects are generated in Rvd binary directory

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${RVD_SOURCE_DIR}/${RELATIVE_BIN_OUTPUT_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${RVD_SOURCE_DIR}/${RELATIVE_LIB_OUTPUT_DIR})

##############################################################################

include(CheckIncludeFileCXX)
include(CheckCXXCompilerFlag)

if(MSVC)
# Check for standard to use
  check_cxx_compiler_flag(/std=c++17 HAVE_FLAG_STD_CXX17)
  if (HAVE_FLAG_STD_CXX17)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std=c++17")
  else()
    set(CMAKE_CXX_STANDARD 14)
  endif()
else()
# Check for standard to use
  check_cxx_compiler_flag(-std=c++17 HAVE_FLAG_STD_CXX17)
  if (HAVE_FLAG_STD_CXX17)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
  else()
    set(CMAKE_CXX_STANDARD 14)
  endif()
endif(MSVC)

set(CMAKE_REQUIRED_FLAGS ${CMAKE_CXX_FLAGS})
CHECK_INCLUDE_FILE_CXX("optional" OPTIONAL_FOUND)
if (NOT OPTIONAL_FOUND OR MSVC)
   add_definitions(-DWITHOUT_OPTIONAL)
   message("optional not found")
endif()

# add the main sources
aux_source_directories(SOURCES "Source Files\\" .)

#==============================================================================
# THIRD PARTY 
#==============================================================================

#==============================================================================

if(VORPALINE_WITH_ASAN AND (("${CMAKE_BUILD_TYPE}" STREQUAL "Release") OR ("${CMAKE_BUILD_TYPE}" STREQUAL "RelWithDebInfo")) AND NOT WIN32)
    add_flags(CMAKE_CXX_FLAGS -fsanitize=address -fno-omit-frame-pointer)
    add_flags(CMAKE_C_FLAGS -fsanitize=address -fno-omit-frame-pointer)
endif()

# generate the configuration files
#   on windows in the sources directories
#   on linux/mac in the build directories to avoid potential conflicts


include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_executable(createGrid ${SOURCES})

#if(USE_BUILTIN_GEOGRAM)
	target_link_libraries(createGrid geogram exploragram ${OPENGL_LIBRARIES})
	if(${GEOGRAM_WITH_VORPALINE})
	  target_link_libraries(createGrid vorpalib)
	endif()
#endif()
