set(HDF5_PREFER_PARALLEL ON)
find_package(HDF5 REQUIRED)

# HDF5_PREFER_PARALLEL is only a preference: when no parallel build is available,
# find_package() succeeds with a serial HDF5. The XDMF backend then calls MPI-only
# entry points unconditionally (H5Pset_fapl_mpio, H5Pset_dxpl_mpio in
# mirheo/core/xdmf/hdf5_helpers.cpp), so the mismatch only surfaces much later as
# "'H5Pset_dxpl_mpio' was not declared in this scope". Reject it here instead.
if (NOT HDF5_IS_PARALLEL)
  message(FATAL_ERROR
    "Mirheo requires a parallel (MPI-enabled) HDF5 build, but the HDF5 found at "
    "'${HDF5_INCLUDE_DIRS}' is serial. Install a parallel HDF5 (e.g. libhdf5-openmpi-dev "
    "or libhdf5-mpich-dev) and, if needed, point HDF5_ROOT at it.")
endif()

# On CRAY systems things are complicated
# This workaround should work to supply
# nvcc with correct hdf paths
if (DEFINED ENV{HDF5_DIR})
  set(HDF5_INCLUDE_DIRS "$ENV{HDF5_DIR}/include")
  set(HDF5_LIBRARIES    "$ENV{HDF5_DIR}/lib/libhdf5.so")
endif()
