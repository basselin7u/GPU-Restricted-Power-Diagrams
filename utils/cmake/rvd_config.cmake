# User-configurable variables:
# - Path to Geogram

# Path to geogram
#################

if(IS_DIRECTORY ${CMAKE_SOURCE_DIR}/geogram/)
   set(
      GEOGRAM_SOURCE_DIR "${CMAKE_SOURCE_DIR}/geogram/"
      CACHE PATH "full path to the Geogram (or Vorpaline) installation"
   )
   set(USE_BUILTIN_GEOGRAM TRUE)
else()
   set(
      GEOGRAM_SOURCE_DIR "${CMAKE_SOURCE_DIR}/../geogram/"
      CACHE PATH "full path to the Geogram (or Vorpaline) installation"
   )
   set(USE_BUILTIN_GEOGRAM FALSE)
endif()
