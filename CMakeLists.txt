cmake_minimum_required(VERSION 3.10)

if ("3.12.0" VERSION_LESS_EQUAL ${CMAKE_VERSION} AND ${CMAKE_VERSION} VERSION_LESS "3.13.0")
  message(FATAL_ERROR "CUDA is broken in CMake 3.12 due to incorrect linker flags (current version is ${CMAKE_VERSION}). \
Please use another version of CMake.\n\
For more info see the following: \
https://gitlab.kitware.com/cmake/cmake/commit/e768d96c74579c79e184027775e51b08cd77fe45")
endif()

if(POLICY CMP0060)
  cmake_policy(SET CMP0060 NEW)
endif()

if(POLICY CMP0069) # for INTERPROCEDURAL_OPTIMIZATION
  cmake_policy(SET CMP0069 NEW)
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

# *************************
# include external packages
# *************************

# Add pybind11
if (NOT TARGET pybind11)
  add_subdirectory(src/extern/pybind11)
endif()
set(PYBIND11_CPP_STANDARD -std=c++14)

# Add mpark/variant
if (NOT TARGET mpark_variant)
    add_subdirectory(src/extern/variant)
endif()

# Add pugixml
if (NOT TARGET pugixml)
    add_subdirectory(src/extern/pugixml)
endif()
set_property(TARGET pugixml-static PROPERTY POSITION_INDEPENDENT_CODE ON)

# CUDA
# Dynamic linking should allegedly improve UCX-based MPI,
# but it seriously affects other code aspects (maybe stream polling?)
#set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
find_package(CUDA 9.2 REQUIRED)

# MPI
include(mpi)
set(CMAKE_CUDA_HOST_LINK_LAUNCHER ${MPI_CXX_COMPILER})

# **********************
# Optional packages
include(hdf5 REQUIRED)
find_package(LIBBFD REQUIRED)
# **********************


# Setup compiler flags

set(cxx_flags -g)
set(cuda_flags --expt-extended-lambda)

set(cxx_private_flags -Wall -Wextra -Wno-unknown-pragmas -pedantic -Wconversion -Werror)
set(cuda_private_flags --compiler-options "-Wall -Wextra -Wno-unknown-pragmas -Werror")


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

# uncomment to use sanitizer
# set(san_opt "-fsanitize=undefined")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${san_opt} -g")
# set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${san_opt}")

# Choose Release mode as default.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
    "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()

# *************************
# add the mirheo libraries
# *************************

# Alias some dirs
set(MIR_BASE_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(CORE_DIR       "${MIR_BASE_INCLUDE_DIR}/mirheo/core/")
set(PYTHON_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mirheo")

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


set(LIB_MIR_CORE "mirheoCore")
set(LIB_MIR_CORE_AND_PLUGINS "mirheoCoreAndPlugins")
set(LIB_MIR "libmirheo")


add_subdirectory(src)

# link time optimization
# set_target_properties(${LIB_MIR_CORE}             PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
# set_target_properties(${LIB_MIR_CORE_AND_PLUGINS} PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
# set_target_properties(${LIB_MIR}                  PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)


# uncomment to profile time spent to compile each object
#set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CMAKE_COMMAND} -E time")
