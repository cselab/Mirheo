set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# *************************
# include external packages
# *************************

if (MIR_BUILD_PYTHON_MODULE)
  # Add pybind11
  if (NOT TARGET pybind11 AND NOT TARGET pybind11::headers)
    add_subdirectory(extern/pybind11)
  endif()
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

include(hdf5 REQUIRED)

if (MIR_ENABLE_STACKTRACE)
  find_package(LIBBFD REQUIRED)
endif()


# Setup compiler flags

set(cxx_flags -g)
set(cuda_flags --expt-extended-lambda --expt-relaxed-constexpr -Wno-deprecated-gpu-targets)

# set(Werror "-Werror")
set(Werror "")
set(cxx_private_flags -Wall -Wextra -Wno-unknown-pragmas -pedantic -Wconversion ${Werror} -Wshadow)
set(cuda_private_flags --compiler-options "-Wall -Wextra -Wno-unknown-pragmas ${Werror} -Wshadow")

# Option to specify architectures. Auto-detect if not provided.
set(MIR_CUDA_ARCH_NAME "Auto" CACHE STRING "Architectures to compile for. Auto by default.")

# The options come out crooked, fix'em
cuda_select_nvcc_arch_flags(BUGGED_ARCH_FLAGS ${MIR_CUDA_ARCH_NAME})

string(REPLACE "gencode;" "gencode=" ARCH_FLAGS_LIST "${BUGGED_ARCH_FLAGS}")
string(REPLACE ";" " " CUDA_ARCH_FLAGS "${ARCH_FLAGS_LIST}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CUDA_ARCH_FLAGS}")

# Export the determined flags to codes using Mirheo as a library.
set(MIR_CUDA_ARCH_FLAGS "${CUDA_ARCH_FLAGS}" CACHE INTERNAL "Architecture flags.")

# Linker flags
set(CMAKE_LINK_FLAGS "${CMAKE_LINK_FLAGS} -rdynamic -flto -g")

if (MIR_ENABLE_SANITIZER)
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
