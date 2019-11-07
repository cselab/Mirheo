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

# For supporting CMake < 3.9:

if(NOT TARGET MPI::MPI_CXX)
    add_library(MPI::MPI_CXX IMPORTED INTERFACE)

    set_property(TARGET MPI::MPI_CXX
                 PROPERTY INTERFACE_COMPILE_OPTIONS ${MPI_CXX_COMPILE_FLAGS})
    set_property(TARGET MPI::MPI_CXX
                 PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${MPI_CXX_INCLUDE_PATH}")
    set_property(TARGET MPI::MPI_CXX
                 PROPERTY INTERFACE_LINK_LIBRARIES ${MPI_CXX_LINK_FLAGS} ${MPI_CXX_LIBRARIES})
endif()
