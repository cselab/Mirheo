cmake_minimum_required(VERSION 3.8)

project(uDeviceX LANGUAGES C CXX CUDA)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/../cmake")

# Add the pybind11
add_subdirectory(extern/pybind11)

# CUDA
find_package(CUDA REQUIRED)

# MPI
find_package(MPI  REQUIRED)
include_directories(${MPI_CXX_INCLUDE_PATH})
set(CMAKE_C_COMPILER   ${MPI_C_COMPILER})
set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})
set(CMAKE_CUDA_HOST_LINK_LAUNCHER ${MPI_CXX_COMPILER})

# Require c++11
#set(PYBIND11_CPP_STANDARD -std=c++11)
set(CMAKE_CXX_STANDARD  11)
set(CMAKE_CUDA_STANDARD 11)

# Alias some dirs
include_directories( ./ )
set(CORE_DIR "${CMAKE_SOURCE_DIR}/core/")
set(PLUGINS_DIR "${CMAKE_SOURCE_DIR}/plugins/")

# Find sources
file(GLOB_RECURSE CXX_SOURCES_CORE     RELATIVE ${CMAKE_SOURCE_DIR} "${CORE_DIR}/*.cpp")
file(GLOB_RECURSE CXX_SOURCES_PLUGINS  RELATIVE ${CMAKE_SOURCE_DIR} "${PLUGINS_DIR}/*.cpp")

file(GLOB_RECURSE CUDA_SOURCES_CORE    RELATIVE ${CMAKE_SOURCE_DIR} "${CORE_DIR}/*.cu")
file(GLOB_RECURSE CUDA_SOURCES_PLUGINS RELATIVE ${CMAKE_SOURCE_DIR} "${PLUGINS_DIR}/*.cu")

set( CXX_SOURCES  ${CXX_SOURCES_CORE}  ${CXX_SOURCES_PLUGINS})
set(CUDA_SOURCES ${CUDA_SOURCES_CORE} ${CUDA_SOURCES_PLUGINS})

# Define the main executable
add_executable(udevicex ../apps/udevicex/main.cu ${CXX_SOURCES} ${CUDA_SOURCES})

#######################################################
# Optional packages
# For field dumps
set(HDF5_PREFER_PARALLEL ON)
find_package(HDF5 REQUIRED)
if (${HDF5_FOUND})
    include_directories(${HDF5_INCLUDE_DIRS})
    target_link_libraries(udevicex ${HDF5_LIBRARIES})
endif(${HDF5_FOUND})

# For pretty stacktrace in case of a crash
find_package(LIBBFD REQUIRED)
if (${LIBBFD_FOUND})
    include_directories(${LIBBFD_INCLUDE_DIRS})
    target_link_libraries(udevicex ${LIBBFD_BFD_LIBRARY})
endif(${LIBBFD_FOUND})
########################################################

# Setup nvcc flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG --use_fast_math -lineinfo -g")
set(CMAKE_CUDA_FLAGS_DEBUG "-O0 -G -g")

# Auto-detect compute capability if not provided
if (NOT DEFINED CUDA_ARCH_FLAGS)
    cuda_select_nvcc_arch_flags(BUGGED_ARCH_FLAGS Auto)
    string(REPLACE ";" "=" CUDA_ARCH_FLAGS "${BUGGED_ARCH_FLAGS}")
endif(NOT DEFINED CUDA_ARCH_FLAGS)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CUDA_ARCH_FLAGS}")

# Linker flags
set(CMAKE_LINK_FLAGS "${CMAKE_LINK_FLAGS} -rdynamic -flto -g")

# Choose Release mode as default.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
        "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif(NOT CMAKE_BUILD_TYPE)

#cuda_compile(CUDA_COMPILED_FILES ${CUDA_SOURCES})# SHARED)
#pybind11_add_module(test1 main.cpp ${cuda_files} )
