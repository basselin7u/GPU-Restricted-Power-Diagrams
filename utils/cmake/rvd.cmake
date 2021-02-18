
##############################################################################

set(RVD_SOURCE_DIR ${CMAKE_SOURCE_DIR})
include(${RVD_SOURCE_DIR}/cmake/rvd_config.cmake)

# CMakeOptions is included after (so that it is 
# possible to override user-editable variables in
# it instead of using CMakeGUI)

if(EXISTS ${RVD_SOURCE_DIR}/CMakeOptions.txt)
   message(STATUS "Using options file: ${RVD_SOURCE_DIR}/CMakeOptions.txt")
   include(${RVD_SOURCE_DIR}/CMakeOptions.txt)
endif()

include(${GEOGRAM_SOURCE_DIR}/cmake/geogram.cmake)

##############################################################################

# Usage: copy_geogram_DLLs_for(target)
#   where target denotes a build target (usually the rvd executable)
# Under Windows: copies the geogram DLLs used by Rvd in the binaries directory
#  (else it cannot find them, why ? I don't know...)
# Under Linux: does nothing

macro(copy_geogram_DLLs_for __target)

  if(WIN32 AND NOT USE_BUILTIN_GEOGRAM)

    add_custom_command(
      TARGET ${__target} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
      ${GEOGRAM_SOURCE_DIR}/${RELATIVE_BIN_DIR}/geogram.dll 
          ${CMAKE_SOURCE_DIR}/${RELATIVE_BIN_DIR}/geogram.dll
    )

    add_custom_command(
      TARGET ${__target} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
      ${GEOGRAM_SOURCE_DIR}/${RELATIVE_BIN_DIR}/geogram_gfx.dll 
         ${CMAKE_SOURCE_DIR}/${RELATIVE_BIN_DIR}/geogram_gfx.dll
    )

    add_custom_command(
      TARGET ${__target} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
      ${GEOGRAM_SOURCE_DIR}/${RELATIVE_BIN_DIR}/geogram_glfw3.dll
         ${CMAKE_SOURCE_DIR}/${RELATIVE_BIN_DIR}/geogram_glfw3.dll
    )

    if(GEOGRAM_WITH_VORPALINE)
      add_custom_command(
        TARGET ${__target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${GEOGRAM_SOURCE_DIR}/${RELATIVE_BIN_DIR}/vorpalib.dll 
            ${CMAKE_SOURCE_DIR}/${RELATIVE_BIN_DIR}/vorpalib.dll
      )
    endif()

  endif()

endmacro()

##############################################################################

include_directories(${RVD_SOURCE_DIR}/src/lib)
link_directories(${RVD_SOURCE_DIR}/${RELATIVE_LIB_DIR})

if (NOT USE_BUILTIN_GEOGRAM AND NOT WIN32)
   # add classic geogram library directory
   link_directories(${GEOGRAM_SOURCE_DIR}/build/${VORPALINE_PLATFORM}-${CMAKE_BUILD_TYPE}/lib)
endif()

##############################################################################
