cmake_minimum_required(VERSION 3.8)

if(POLICY CMP0060)
  cmake_policy(SET CMP0060 NEW)
endif()

if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(options)
include(version)

getMirheoVersion(MIR_VERSION MIR_VERSION_CMAKE_FORMAT)
getMirheoSHA1(MIR_SHA1)

message("Compiling libmirheo version ${MIR_VERSION_CMAKE_FORMAT}")

project(Mirheo VERSION ${MIR_VERSION_CMAKE_FORMAT} LANGUAGES C CXX CUDA)

set(LIB_MIR_CORE "mirheoCore")
set(LIB_MIR_CORE_AND_PLUGINS "mirheoCoreAndPlugins")
set(LIB_MIR "libmirheo")

add_library(${LIB_MIR_CORE}             STATIC)
add_library(${LIB_MIR_CORE_AND_PLUGINS} STATIC)
add_library(${LIB_MIR}                  MODULE)

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
find_package(MPI REQUIRED)

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

# Require c++14
set(PYBIND11_CPP_STANDARD -std=c++14)
set(CMAKE_CXX_STANDARD  14)
set(CMAKE_CUDA_STANDARD 14)

# Alias some dirs
include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/src/ )
set(CORE_DIR       "${CMAKE_CURRENT_SOURCE_DIR}/src/mirheo/core/")
set(PLUGINS_DIR    "${CMAKE_CURRENT_SOURCE_DIR}/src/mirheo/plugins/")
set(BINDINGS_DIR   "${CMAKE_CURRENT_SOURCE_DIR}/src/mirheo/bindings/")
set(PYTHON_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mirheo")
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

# Find sources and add them to the libraries
file(GLOB_RECURSE SOURCES_CORE     RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "${CORE_DIR}/*.cu" "${CORE_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/extern/pugixml/src/*.cpp")
file(GLOB_RECURSE SOURCES_PLUGINS  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "${PLUGINS_DIR}/*.cu" "${PLUGINS_DIR}/*.cpp")
file(GLOB_RECURSE SOURCES_BINDINGS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "${BINDINGS_DIR}/*.cpp")

target_sources(${LIB_MIR_CORE}             PRIVATE ${SOURCES_CORE})
target_sources(${LIB_MIR_CORE_AND_PLUGINS} PRIVATE ${SOURCES_PLUGINS})
target_sources(${LIB_MIR}                  PRIVATE ${SOURCES_BINDINGS})

set_property(TARGET ${LIB_MIR_CORE_AND_PLUGINS} PROPERTY POSITION_INDEPENDENT_CODE 1)

target_link_libraries(${LIB_MIR} PRIVATE pybind11::module)
set_target_properties(${LIB_MIR} PROPERTIES
  PREFIX "${PYTHON_MODULE_PREFIX}"
  SUFFIX "${PYTHON_MODULE_EXTENSION}")

target_link_libraries(${LIB_MIR_CORE} PUBLIC ${CUDA_LIBRARIES})
target_link_libraries(${LIB_MIR_CORE} PUBLIC mpark_variant)

target_link_libraries(${LIB_MIR_CORE_AND_PLUGINS} PUBLIC ${LIB_MIR_CORE})
target_link_libraries(${LIB_MIR}                 PRIVATE ${LIB_MIR_CORE_AND_PLUGINS})

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
  target_include_directories(${LIB_MIR_CORE}             PRIVATE ${HDF5_INCLUDE_DIRS})
  target_include_directories(${LIB_MIR_CORE_AND_PLUGINS} PRIVATE ${HDF5_INCLUDE_DIRS})
  target_link_libraries(${LIB_MIR_CORE}             PRIVATE ${HDF5_LIBRARIES})
  target_link_libraries(${LIB_MIR_CORE_AND_PLUGINS} PRIVATE ${HDF5_LIBRARIES})
endif()


# For pretty stacktrace in case of a crash
find_package(LIBBFD REQUIRED)
if (${LIBBFD_FOUND})
  target_include_directories(${LIB_MIR_CORE} PUBLIC ${LIBBFD_INCLUDE_DIRS})
  target_link_libraries(${LIB_MIR_CORE} PUBLIC ${LIBBFD_BFD_LIBRARY})
endif()
########################################################

# Setup compiler flags
set(WARNING_FLAGS "-Wall -Wextra -Wno-unknown-pragmas -Werror")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g ${WARNING_FLAGS} -pedantic ")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda  --compiler-options \"${WARNING_FLAGS}\"")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG --use_fast_math -lineinfo -g")
set(CMAKE_CUDA_FLAGS_DEBUG "-O0 -G -g")


if (DOUBLE_PRECISION)
  add_definitions(-DMIRHEO_DOUBLE_PRECISION)
  message("compiling with DOUBLE_PRECISION ON")
  set(MEMBRANE_DOUBLE ON)
  set(ROD_DOUBLE ON)
endif()

if (MEMBRANE_DOUBLE)
  add_definitions(-DMEMBRANE_FORCES_DOUBLE)
  message("compiling with MEMBRANE_DOUBLE ON")
endif()

if (ROD_DOUBLE)
  add_definitions(-DROD_FORCES_DOUBLE)
  message("compiling with ROD_DOUBLE ON")
endif()

if (USE_NVTX)
  add_definitions(-DUSE_NVTX)
  target_link_libraries(${LIB_MIR_CORE} PUBLIC "-lnvToolsExt")
  message("compiling with USE_NVTX ON")
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

set_property(TARGET ${LIB_MIR_CORE_AND_PLUGINS} PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET ${LIB_MIR_CORE}             PROPERTY POSITION_INDEPENDENT_CODE ON)

# Linker flags
set(CMAKE_LINK_FLAGS "${CMAKE_LINK_FLAGS} -rdynamic -flto -g")

# Choose Release mode as default.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
    "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()
