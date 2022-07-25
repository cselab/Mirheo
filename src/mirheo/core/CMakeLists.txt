set(sources
  celllist.cu
  domain.cpp
  logger.cpp
  marching_cubes.cpp
  mirheo.cpp
  mirheo_object.cpp
  mirheo_state.cpp
  plugins.cpp
  postproc.cpp
  simulation.cpp
  task_scheduler.cpp
  version.cpp
)

add_library(${LIB_MIR_CORE} STATIC ${sources})

add_subdirectory(analytical_shapes)
add_subdirectory(bouncers)
add_subdirectory(exchangers)
add_subdirectory(field)
add_subdirectory(initial_conditions)
add_subdirectory(integrators)
add_subdirectory(interactions)
add_subdirectory(managers)
add_subdirectory(mesh)
add_subdirectory(object_belonging)
add_subdirectory(pvs)
add_subdirectory(rigid)
add_subdirectory(types)
add_subdirectory(utils)
add_subdirectory(walls)
add_subdirectory(xdmf)


target_include_directories(${LIB_MIR_CORE} PUBLIC ${MIR_BASE_INCLUDE_DIR} )
target_include_directories(${LIB_MIR_CORE} PUBLIC ${MIR_BASE_INCLUDE_DIR}/extern/cuda_variant/ )
target_include_directories(${LIB_MIR_CORE} PUBLIC ${MPI_CXX_INCLUDE_DIRS})
target_include_directories(${LIB_MIR_CORE} PUBLIC ${CUDA_INCLUDE_DIRS})

# CUB. Already available in CUDA since version 11.0.
if ("${CUDA_VERSION}" VERSION_LESS "11.0")
    target_include_directories(${LIB_MIR_CORE} PUBLIC ${MIR_BASE_INCLUDE_DIR}/extern/cub/)
endif()

target_link_libraries(${LIB_MIR_CORE} PUBLIC MPI::MPI_CXX)
target_link_libraries(${LIB_MIR_CORE} PUBLIC ${CUDA_LIBRARIES})
target_link_libraries(${LIB_MIR_CORE} PRIVATE pugixml-static) # don t use the alias here because we need to set a property later

if (${HDF5_FOUND})
  target_include_directories(${LIB_MIR_CORE} PRIVATE ${HDF5_INCLUDE_DIRS})
  target_link_libraries(${LIB_MIR_CORE}      PRIVATE ${HDF5_LIBRARIES})
endif()

if (${LIBBFD_FOUND})
  target_include_directories(${LIB_MIR_CORE} PUBLIC ${LIBBFD_INCLUDE_DIRS})
  target_link_libraries(${LIB_MIR_CORE} PUBLIC ${LIBBFD_BFD_LIBRARY})
endif()


# set compilation flags

target_compile_features(${LIB_MIR_CORE} PUBLIC cxx_std_17)

target_compile_options(${LIB_MIR_CORE} PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:${cxx_private_flags}>
  $<$<COMPILE_LANGUAGE:CUDA>:${cuda_private_flags}>
  )

target_compile_options(${LIB_MIR_CORE} PUBLIC
  $<$<COMPILE_LANGUAGE:CXX>:${cxx_flags}>
  $<$<COMPILE_LANGUAGE:CUDA>:${cuda_flags}
                             $<$<CONFIG:DEBUG>:-O0;-g;-G;--std=c++17>
			     $<$<CONFIG:RELEASE>:-g;-DNDEBUG;--use_fast_math;-lineinfo;--std=c++17>
			     >
  )

# -fPIC
set_property(TARGET ${LIB_MIR_CORE} PROPERTY POSITION_INDEPENDENT_CODE ON)

# set options

if (MIR_DOUBLE_PRECISION)
  target_compile_definitions(${LIB_MIR_CORE} PUBLIC MIRHEO_DOUBLE_PRECISION)
  message("compiling with MIRHEO_DOUBLE_PRECISION ON")
  set(MIR_MEMBRANE_DOUBLE ON)
  set(MIR_ROD_DOUBLE ON)
endif()

if (MIR_MEMBRANE_DOUBLE)
  target_compile_definitions(${LIB_MIR_CORE} PUBLIC MIRHEO_MEMBRANE_FORCES_DOUBLE)
  message("compiling with MIRHEO_MEMBRANE_DOUBLE ON")
endif()

if (MIR_ROD_DOUBLE)
  target_compile_definitions(${LIB_MIR_CORE} PUBLIC MIRHEO_ROD_FORCES_DOUBLE)
  message("compiling with MIRHEO_ROD_DOUBLE ON")
endif()

if (MIR_USE_NVTX)
  target_compile_definitions(${LIB_MIR_CORE} PRIVATE MIRHEO_USE_NVTX)
  target_link_libraries(${LIB_MIR_CORE} PUBLIC "-lnvToolsExt")
  message("compiling with MIRHEO_USE_NVTX ON")
endif()

# Alias

add_library(mirheo::core ALIAS ${LIB_MIR_CORE})

# make sure these sources are not used elsewhere
unset(sources_cpp)
unset(sources_cu)
