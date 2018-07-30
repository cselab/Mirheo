cmake_minimum_required(VERSION 3.8)

if(POLICY CMP0060)
    cmake_policy(SET CMP0060 NEW)
endif()

project(uDeviceX LANGUAGES C CXX CUDA)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(UDX "libudevicex")
set(UDX_CUDA "udevicex_cuda")

# Add the pybind11
add_subdirectory(src/extern/pybind11)

# CUDA
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
#set(PYBIND11_CPP_STANDARD -std=c++11)
set(CMAKE_CXX_STANDARD  11)
set(CMAKE_CUDA_STANDARD 11)

# Alias some dirs
include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/src/ )
set(CORE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/core/")
set(PLUGINS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/plugins/")
set(BINDINGS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/bindings/")

# Find sources
file(GLOB_RECURSE CXX_SOURCES_CORE      RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "${CORE_DIR}/*.cpp")
file(GLOB_RECURSE CXX_SOURCES_PLUGINS   RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "${PLUGINS_DIR}/*.cpp")
file(GLOB_RECURSE CXX_SOURCES_BINDINGS  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "${BINDINGS_DIR}/*.cpp")

file(GLOB_RECURSE CUDA_SOURCES_CORE     RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "${CORE_DIR}/*.cu")
file(GLOB_RECURSE CUDA_SOURCES_PLUGINS  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "${PLUGINS_DIR}/*.cu")
file(GLOB_RECURSE CUDA_SOURCES_BINDINGS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "${BINDINGS_DIR}/*.cu")

set( CXX_SOURCES  ${CXX_SOURCES_CORE}  ${CXX_SOURCES_PLUGINS}  ${CXX_SOURCES_BINDINGS})
set(CUDA_SOURCES ${CUDA_SOURCES_CORE} ${CUDA_SOURCES_PLUGINS} ${CUDA_SOURCES_BINDINGS})

# Define the main executable
#add_executable(udevicex ../apps/udevicex/main.cu ${CXX_SOURCES} ${CUDA_SOURCES})
add_library(cuda_compiled OBJECT ${CUDA_SOURCES})
set_property(TARGET cuda_compiled PROPERTY POSITION_INDEPENDENT_CODE 1)

add_library(${UDX_CUDA} SHARED $<TARGET_OBJECTS:cuda_compiled>)
add_library(${UDX} MODULE ${CXX_SOURCES} $<TARGET_OBJECTS:cuda_compiled>)
target_link_libraries(${UDX} PRIVATE pybind11::module)
set_target_properties(${UDX} PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                                        SUFFIX "${PYTHON_MODULE_EXTENSION}")

target_link_libraries(${UDX} PRIVATE ${CUDA_LIBRARIES})
target_link_libraries(${UDX_CUDA} PRIVATE ${CUDA_LIBRARIES})


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
  target_link_libraries(${UDX} PRIVATE ${HDF5_LIBRARIES})
  target_link_libraries(${UDX_CUDA} PRIVATE ${HDF5_LIBRARIES})
endif()


# For pretty stacktrace in case of a crash
find_package(LIBBFD REQUIRED)
if (${LIBBFD_FOUND})
  include_directories(${LIBBFD_INCLUDE_DIRS})
  target_link_libraries(${UDX} PRIVATE ${LIBBFD_BFD_LIBRARY})
  target_link_libraries(${UDX_CUDA} PRIVATE ${LIBBFD_BFD_LIBRARY})
endif()
########################################################

# Setup nvcc flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG --use_fast_math -lineinfo -g")
set(CMAKE_CUDA_FLAGS_DEBUG "-O0 -G -g")

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

