set(HDF5_PREFER_PARALLEL ON)
find_package(HDF5 REQUIRED)

# On CRAY systems things are complicated
# This workaround should work to supply
# nvcc with correct hdf paths
if (DEFINED ENV{CRAY_HDF5_DIR})
  set(HDF5_INCLUDE_DIRS "$ENV{HDF5_DIR}/include")
  set(HDF5_LIBRARIES    "$ENV{HDF5_DIR}/lib/libhdf5.so")
endif()
