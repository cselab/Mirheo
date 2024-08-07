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

if(POLICY CMP0148)
  cmake_policy(SET CMP0148 NEW)
  set(PYBIND11_FINDPYTHON ON)
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# ***********************
# Options
# ***********************

option(MIR_BUILD_PYTHON_MODULE "Build mirheo python module"        ON )
option(MIR_BUILD_TESTS         "Build mirheo unit tests"           OFF)
option(MIR_ENABLE_LTO          "enable link time optimization"     OFF)
option(MIR_ENABLE_SANITIZER    "enable ub sanitizer"               OFF)
option(MIR_PROFILE_COMPILATION "print compilation profiling info"  OFF)
option(MIR_ENABLE_STACKTRACE   "print a stacktrace when failing"   ON )

set(mir_definitions "")

include(options)
include(version)

getMirheoVersion(MIR_VERSION MIR_VERSION_CMAKE_FORMAT)
getMirheoSHA1(MIR_SHA1)

message("Compiling libmirheo version ${MIR_VERSION_CMAKE_FORMAT}")

project(Mirheo VERSION ${MIR_VERSION_CMAKE_FORMAT} LANGUAGES C CXX CUDA)

# ***********************
# Alias directories
# ***********************

set(MIR_BASE_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(MIR_CORE_DIR         "${MIR_BASE_INCLUDE_DIR}/mirheo/core/")
set(PYTHON_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mirheo")

# ***********************
# generate version files
# ***********************

configure_file(
  "${MIR_CORE_DIR}/version.cpp.in"
  "${MIR_CORE_DIR}/version.cpp"
  )
configure_file(
  "${PYTHON_LIB_DIR}/version.py.in"
  "${PYTHON_LIB_DIR}/version.py"
  )

# *************************
# add the mirheo libraries
# *************************

set(LIB_MIR_CORE "mirheoCore")
set(LIB_MIR_CORE_AND_PLUGINS "mirheoCoreAndPlugins")
set(LIB_MIR "libmirheo")

add_subdirectory(src)

if (MIR_ENABLE_LTO)
  set_target_properties(${LIB_MIR_CORE}             PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
  set_target_properties(${LIB_MIR_CORE_AND_PLUGINS} PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
  set_target_properties(${LIB_MIR}                  PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

# *************************
# unit tests
# *************************

if (MIR_BUILD_TESTS)
  enable_testing()
  add_subdirectory(units)
endif()

if (MIR_PROFILE_COMPILATION)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CMAKE_COMMAND} -E time")
endif()
