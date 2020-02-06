set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# *************************
# include external packages
# *************************

if (BUILD_PYTHON_MODULE)
  # Add pybind11
  if (NOT TARGET pybind11)
    add_subdirectory(extern/pybind11)
  endif()
  set(PYBIND11_CPP_STANDARD -std=c++14)
endif()

# Add mpark/variant
if (NOT TARGET mpark_variant)
    add_subdirectory(extern/variant)
endif()

# Add pugixml
if (NOT TARGET pugixml)
    add_subdirectory(extern/pugixml)
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

set(cxx_private_flags -Wall -Wextra -Wno-unknown-pragmas -pedantic -Wconversion -Werror -Wshadow)
set(cuda_private_flags --compiler-options "-Wall -Wextra -Wno-unknown-pragmas -Werror -Wshadow")


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

if (ENABLE_SANITIZER)
  set(san_opt "-fsanitize=undefined")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${san_opt} -g")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${san_opt}")
endif()

# Choose Release mode as default.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
    "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()

add_subdirectory(mirheo)
