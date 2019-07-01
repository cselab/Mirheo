cmake_minimum_required(VERSION 3.8)

if(POLICY CMP0060)
  cmake_policy(SET CMP0060 NEW)
endif()

if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

# compile options for Mirheo library
option(USE_NVTX "enable NVTX profiling" OFF)
option(MEMBRANE_DOUBLE "compute membrane forces in double precision" OFF)
option(ROD_DOUBLE      "compute rod forces in double precision" OFF)

project(Mirheo LANGUAGES C CXX CUDA)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(MIR "libmirheo")
set(MIR_MAIN "mirheo_main")

# version
execute_process(COMMAND
  git describe --abbrev=0
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  OUTPUT_VARIABLE MIR_VERSION
  ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

# the commit's SHA1, and whether the building workspace was dirty or not
execute_process(COMMAND
  git describe --match=NeVeRmAtCh --always --abbrev=40 --dirty
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  OUTPUT_VARIABLE GIT_SHA1
  ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

# Add pybind11
add_subdirectory(src/extern/pybind11)

# add mpark/variant
add_subdirectory(src/extern/variant)

# CUDA
# Dynamic linking should allegedly improve UCX-based MPI,
# but it seriously affects other code aspects (maybe stream polling?)
#set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# MPI
find_package(MPI  REQUIRED)

# On CRAY systems things are complicated
# This workaround should work to supply
# nvcc with correct mpi paths
# Libraries should not be needed here as
# we link with MPI wrapper anyways
if (DEFINED ENV{CRAY_MPICH_DIR})
  set(MPI_C_INCLUDE_DIRS   "$ENV{CRAY_MPICH_DIR}/include")
  set(MPI_CXX_INCLUDE_DIRS "$ENV{CRAY_MPICH_DIR}/include")
endif()

include_directories(${MPI_CXX_INCLUDE_DIRS})
set(CMAKE_C_COMPILER   ${MPI_C_COMPILER})
set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})
set(CMAKE_CUDA_HOST_LINK_LAUNCHER ${MPI_CXX_COMPILER})

# Require c++11
#set(PYBIND11_CPP_STANDARD -std=c++14)
set(CMAKE_CXX_STANDARD  14)
set(CMAKE_CUDA_STANDARD 14)

# Alias some dirs
include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/src/ )
set(CORE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/core/")
set(PLUGINS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/plugins/")
set(BINDINGS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/bindings/")
set(PYTHON_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/mirheo")
include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/src/extern/cuda_variant/ )

# generate version.cpp
configure_file(
  "${CORE_DIR}/version.cpp.in"
  "${CORE_DIR}/version.cpp"
  )

#generate version.py
configure_file(
  "${PYTHON_LIB_DIR}/version.py.in"
  "${PYTHON_LIB_DIR}/version.py"
  )

# Find sources
file(GLOB_RECURSE SOURCES_BINDINGS  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "${BINDINGS_DIR}/*.cpp")
file(GLOB_RECURSE SOURCES_MAIN      RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
     "${CORE_DIR}/*.cu" "${PLUGINS_DIR}/*.cu" "${CORE_DIR}/*.cpp" "${PLUGINS_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/extern/pugixml/src/*.cpp")
   
# Define the main library
add_library(main_compiled OBJECT ${SOURCES_MAIN})
set_property(TARGET main_compiled PROPERTY POSITION_INDEPENDENT_CODE 1)

add_library(${MIR_MAIN} SHARED $<TARGET_OBJECTS:main_compiled>)

add_library(${MIR} MODULE ${SOURCES_BINDINGS} $<TARGET_OBJECTS:main_compiled>)
target_link_libraries(${MIR} PRIVATE pybind11::module)
set_target_properties(${MIR} PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                                        SUFFIX "${PYTHON_MODULE_EXTENSION}")

target_link_libraries(${MIR} PRIVATE ${CUDA_LIBRARIES})
target_link_libraries(${MIR_MAIN} PRIVATE ${CUDA_LIBRARIES})
target_link_libraries(${MIR_MAIN} PRIVATE mpark_variant)

#######################################################
# Optional packages
# For field dumps
set(HDF5_PREFER_PARALLEL ON)
find_package(HDF5 REQUIRED)

# On CRAY systems things are complicated
# This workaround should work to supply
# nvcc with correct hdf paths
if (DEFINED ENV{CRAY_HDF5_DIR})
  set(HDF5_INCLUDE_DIRS "$ENV{HDF5_DIR}/include")
  set(HDF5_LIBRARIES    "$ENV{HDF5_DIR}/lib/libhdf5.so")
endif()

if (${HDF5_FOUND})
  include_directories(${HDF5_INCLUDE_DIRS})
  target_link_libraries(${MIR} PRIVATE ${HDF5_LIBRARIES})
  target_link_libraries(${MIR_MAIN} PRIVATE ${HDF5_LIBRARIES})
endif()


# For pretty stacktrace in case of a crash
find_package(LIBBFD REQUIRED)
if (${LIBBFD_FOUND})
  include_directories(${LIBBFD_INCLUDE_DIRS})
  target_link_libraries(${MIR} PRIVATE ${LIBBFD_BFD_LIBRARY})
  target_link_libraries(${MIR_MAIN} PRIVATE ${LIBBFD_BFD_LIBRARY})
endif()
########################################################

# Setup compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG --use_fast_math -lineinfo -g")
set(CMAKE_CUDA_FLAGS_DEBUG "-O0 -G -g")

if (USE_NVTX)
  add_definitions(-DUSE_NVTX)
  target_link_libraries(${MIR} PRIVATE "-lnvToolsExt")
  target_link_libraries(${MIR_MAIN} PRIVATE "-lnvToolsExt")
  message("compiling with USE_NVTX ON")
endif()

if (MEMBRANE_DOUBLE)
  add_definitions(-DMEMBRANE_FORCES_DOUBLE)
  message("compiling with MEMBRANE_DOUBLE ON")
endif()

if (ROD_DOUBLE)
  add_definitions(-DROD_FORCES_DOUBLE)
  message("compiling with ROD_DOUBLE ON")
endif()

# Silence deprecation warnings for CUDA >= 9
if (CUDA_VERSION_MAJOR GREATER_EQUAL 9)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --Wno-deprecated-declarations")
endif()

# Auto-detect compute capability if not provided
if (NOT DEFINED CUDA_ARCH_NAME)
  set(CUDA_ARCH_NAME Auto)
endif()

# The options come out crooked, fix'em
cuda_select_nvcc_arch_flags(BUGGED_ARCH_FLAGS ${CUDA_ARCH_NAME})
unset(CUDA_ARCH_NAME CACHE)

string(REPLACE "gencode;" "gencode=" ARCH_FLAGS_LIST "${BUGGED_ARCH_FLAGS}")
string(REPLACE ";" " " CUDA_ARCH_FLAGS "${ARCH_FLAGS_LIST}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CUDA_ARCH_FLAGS}")

# Linker flags
set(CMAKE_LINK_FLAGS "${CMAKE_LINK_FLAGS} -rdynamic -flto -g")

# Choose Release mode as default.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
    "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()
