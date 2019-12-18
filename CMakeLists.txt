cmake_minimum_required(VERSION 3.10)

if ("3.12.0" VERSION_LESS_EQUAL ${CMAKE_VERSION} AND ${CMAKE_VERSION} VERSION_LESS "3.13.0")
  message(FATAL_ERROR "CUDA is broken in CMake 3.12 due to incorrect linker flags (current version is ${CMAKE_VERSION}). \
Please use an older or a newer version of CMake.\n\
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

set(LIB_MIR_CORE "mirheoCore")
set(LIB_MIR_CORE_AND_PLUGINS "mirheoCoreAndPlugins")
set(LIB_MIR "libmirheo")


# Alias some dirs
set(SRC_DIR        "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(CORE_DIR       "${SRC_DIR}/mirheo/core/")
set(PLUGINS_DIR    "${SRC_DIR}/mirheo/plugins/")
set(BINDINGS_DIR   "${SRC_DIR}/mirheo/bindings/")
set(PYTHON_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mirheo")

# Find sources and add them to the libraries
file(GLOB_RECURSE SOURCES_CORE     RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "${CORE_DIR}/*.cu" "${CORE_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/extern/pugixml/src/*.cpp")
file(GLOB_RECURSE SOURCES_PLUGINS  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "${PLUGINS_DIR}/*.cu" "${PLUGINS_DIR}/*.cpp")
file(GLOB_RECURSE SOURCES_BINDINGS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "${BINDINGS_DIR}/*.cpp")

add_library(${LIB_MIR_CORE}             STATIC ${SOURCES_CORE})
add_library(${LIB_MIR_CORE_AND_PLUGINS} STATIC ${SOURCES_PLUGINS})
add_library(${LIB_MIR}                  MODULE ${SOURCES_BINDINGS})
add_library(mirheo::core                ALIAS  ${LIB_MIR_CORE})  # :: to prevent from typos when using mirheo as a dependency.

# Add pybind11
if (NOT TARGET pybind11)
    add_subdirectory(src/extern/pybind11)
endif()

# Add mpark/variant
if (NOT TARGET mpark_variant)
    add_subdirectory(src/extern/variant)
endif()

# CUDA
# Dynamic linking should allegedly improve UCX-based MPI,
# but it seriously affects other code aspects (maybe stream polling?)
#set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
find_package(CUDA REQUIRED)

include(mpi)

target_link_libraries(${LIB_MIR_CORE} PUBLIC MPI::MPI_CXX)
target_link_libraries(${LIB_MIR_CORE} PUBLIC MPI::MPI_CXX)

set(CMAKE_CUDA_HOST_LINK_LAUNCHER ${MPI_CXX_COMPILER})

# Require c++14
set(PYBIND11_CPP_STANDARD -std=c++14)

target_compile_features(${LIB_MIR_CORE}             PUBLIC cxx_std_14)
target_compile_features(${LIB_MIR_CORE_AND_PLUGINS} PUBLIC cxx_std_14)
target_compile_features(${LIB_MIR}                  PUBLIC cxx_std_14)


target_include_directories(${LIB_MIR_CORE}             PUBLIC ${SRC_DIR} )
target_include_directories(${LIB_MIR_CORE_AND_PLUGINS} PUBLIC ${SRC_DIR} )
target_include_directories(${LIB_MIR_CORE}             PUBLIC ${SRC_DIR}/extern/cuda_variant/ )

target_include_directories(${LIB_MIR_CORE}             PUBLIC ${MPI_CXX_INCLUDE_DIRS})
target_include_directories(${LIB_MIR_CORE_AND_PLUGINS} PUBLIC ${MPI_CXX_INCLUDE_DIRS})

target_include_directories(${LIB_MIR_CORE}             PUBLIC ${CUDA_INCLUDE_DIRS})
target_include_directories(${LIB_MIR_CORE_AND_PLUGINS} PUBLIC ${CUDA_INCLUDE_DIRS})


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
include(hdf5)

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


# link time optimization
# set_target_properties(${LIB_MIR_CORE}             PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
# set_target_properties(${LIB_MIR_CORE_AND_PLUGINS} PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
# set_target_properties(${LIB_MIR}                  PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)

# Setup compiler flags

set(cxx_flags -g)
set(cuda_flags --expt-extended-lambda)

set(cxx_private_flags -Wall -Wextra -Wno-unknown-pragmas -pedantic -Wconversion -Werror)
set(cuda_private_flags --compiler-options "-Wall -Wextra -Wno-unknown-pragmas -Werror")

target_compile_options(${LIB_MIR_CORE} PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:${cxx_private_flags}>
  $<$<COMPILE_LANGUAGE:CUDA>:${cuda_private_flags}>
  )

target_compile_options(${LIB_MIR_CORE_AND_PLUGINS} PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:${cxx_private_flags}>
  $<$<COMPILE_LANGUAGE:CUDA>:${cuda_private_flags}>
  )

target_compile_options(${LIB_MIR_CORE} PUBLIC
  $<$<COMPILE_LANGUAGE:CXX>:${cxx_flags}>
  $<$<COMPILE_LANGUAGE:CUDA>:${cuda_flags}
                             $<$<CONFIG:DEBUG>:-O0;-g;-G>
			     $<$<CONFIG:RELEASE>:-g;-DNDEBUG;--use_fast_math;-lineinfo>
			     >
  )

# Set options

if (DOUBLE_PRECISION)
  target_compile_definitions(${LIB_MIR_CORE} PUBLIC MIRHEO_DOUBLE_PRECISION)
  message("compiling with DOUBLE_PRECISION ON")
  set(MEMBRANE_DOUBLE ON)
  set(ROD_DOUBLE ON)
endif()

if (MEMBRANE_DOUBLE)
  target_compile_definitions(${LIB_MIR_CORE} PRIVATE MEMBRANE_FORCES_DOUBLE)
  message("compiling with MEMBRANE_DOUBLE ON")
endif()

if (ROD_DOUBLE)
  target_compile_definitions(${LIB_MIR_CORE} PRIVATE ROD_FORCES_DOUBLE)
  message("compiling with ROD_DOUBLE ON")
endif()

if (USE_NVTX)
  target_compile_definitions(${LIB_MIR_CORE} PRIVATE USE_NVTX)
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

# uncomment to use sanitizer
# set(san_opt "-fsanitize=undefined")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${san_opt} -g")
# set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${san_opt}")

# Choose Release mode as default.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
    "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()

# uncomment to profile time for each target
#set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CMAKE_COMMAND} -E time")
