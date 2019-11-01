cmake_minimum_required(VERSION 3.8)

if (POLICY CMP0060)
    cmake_policy(SET CMP0060 NEW)
endif()

if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

project(MIR_UnitTests LANGUAGES C CXX CUDA)
enable_testing()

# googletest
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/extern/googletest")

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

# Require c++14
set(CMAKE_CXX_STANDARD  14)
set(CMAKE_CUDA_STANDARD 14)

include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/../src/ )
include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/../src/extern/cuda_variant/ )

function (add_test_executable dirName nodes)
  # Find sources
  file(GLOB SOURCES RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/${dirName}/*.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/${dirName}/*.cpp")

  string(REGEX REPLACE "/" "_" execNameId ${dirName})
  
  set(EXEC_NAME "test_${execNameId}")
  
  add_executable(${EXEC_NAME} ${SOURCES})
  target_link_libraries(${EXEC_NAME} PRIVATE ${CUDA_LIBRARIES})
  target_link_libraries(${EXEC_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../build/libmirheo_main.so")
  target_link_libraries(${EXEC_NAME} PRIVATE gtest)
  add_test(NAME ${EXEC_NAME}
           COMMAND mir.run --runargs "-n ${nodes}" ./${EXEC_NAME})
endfunction()

add_test_executable(celllists 1)
add_test_executable(id64 1)
add_test_executable(integration/particles 1)
add_test_executable(integration/rigid 1)
add_test_executable(interaction 1)
add_test_executable(map 1)
add_test_executable(inertia_tensor 1)
add_test_executable(marching_cubes 1)
add_test_executable(object_deleter 1)
add_test_executable(onerank 1)
add_test_executable(packers/exchange 1)
add_test_executable(packers/redistribute 1)
add_test_executable(packers/simple 1)
add_test_executable(pid 1)
add_test_executable(reduce 1)
add_test_executable(restart 4)
add_test_executable(rng 1)
add_test_executable(rod/discretization 1)
add_test_executable(rod/energy 1)
add_test_executable(rod/forces 1)
add_test_executable(roots 1)
add_test_executable(scheduler 1)
add_test_executable(serializer 1)
add_test_executable(triangle_invariants 1)
add_test_executable(variant 1)
add_test_executable(warpScan 1)


# Setup nvcc flags
set(WARNING_FLAGS "-Wall -Wextra -Wno-unknown-pragmas -Werror")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g ${WARNING_FLAGS} -pedantic")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda  --compiler-options \"${WARNING_FLAGS}\"")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG --use_fast_math -lineinfo -g")
set(CMAKE_CUDA_FLAGS_DEBUG "-O0 -G -g")


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

