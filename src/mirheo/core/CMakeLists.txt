set(sources_cpp
  analytical_shapes/api.cpp
  bouncers/interface.cpp
  domain.cpp
  exchangers/engines/mpi.cpp
  exchangers/engines/single_node.cpp
  exchangers/exchanger_interfaces.cpp
  field/from_function.cpp
  field/interface.cpp
  initial_conditions/from_array.cpp
  initial_conditions/helpers.cpp
  initial_conditions/membrane.cpp
  initial_conditions/membrane_with_type_ids.cpp
  initial_conditions/restart.cpp
  initial_conditions/rigid.cpp
  initial_conditions/rod.cpp
  initial_conditions/uniform.cpp
  initial_conditions/uniform_filtered.cpp
  initial_conditions/uniform_sphere.cpp
  integrators/factory.cpp
  integrators/interface.cpp
  integrators/sub_step.cpp
  interactions/factory.cpp
  interactions/interface.cpp
  interactions/pairwise/base_pairwise.cpp
  interactions/pairwise/factory_helper.cpp
  interactions/parameters_wrap.cpp
  interactions/rod/base_rod.cpp
  interactions/utils/step_random_gen.cpp
  logger.cpp
  managers/interactions.cpp
  marching_cubes.cpp
  mesh/factory.cpp
  mesh/membrane.cpp
  mesh/mesh.cpp
  mesh/off.cpp
  mirheo.cpp
  mirheo_object.cpp
  mirheo_state.cpp
  object_belonging/interface.cpp
  plugins.cpp
  postproc.cpp
  pvs/checkpoint/helpers.cpp
  pvs/data_manager.cpp
  pvs/factory.cpp
  pvs/membrane_vector.cpp
  pvs/object_vector.cpp
  pvs/packers/generic_packer.cpp
  pvs/packers/objects.cpp
  pvs/packers/particles.cpp
  pvs/packers/rods.cpp
  pvs/particle_vector.cpp
  pvs/restart/helpers.cpp
  pvs/rigid_ashape_object_vector.cpp
  pvs/rigid_object_vector.cpp
  pvs/rod_vector.cpp
  pvs/views/ov.cpp
  pvs/views/pv.cpp
  pvs/views/rov.cpp
  pvs/views/rv.cpp
  simulation.cpp
  snapshot.cpp
  task_scheduler.cpp
  types/str.cpp
  types/variant_type_wrapper.cpp
  utils/common.cpp
  utils/config.cpp
  utils/file_wrapper.cpp
  utils/folders.cpp
  utils/nvtx.cpp
  utils/reflection.cpp
  utils/stacktrace_explicit.cpp
  utils/strprintf.cpp
  version.cpp
  walls/interface.cpp
  walls/stationary_walls/sdf.cpp
  xdmf/channel.cpp
  xdmf/grids.cpp
  xdmf/hdf5_helpers.cpp
  xdmf/xdmf.cpp
  xdmf/xmf_helpers.cpp
  )

set(sources_cu
  bouncers/from_mesh.cu
  bouncers/from_rod.cu
  bouncers/from_shape.cu
  celllist.cu
  exchangers/exchange_helpers.cu
  exchangers/object_halo_exchanger.cu
  exchangers/object_halo_extra_exchanger.cu
  exchangers/object_redistributor.cu
  exchangers/object_reverse_exchanger.cu
  exchangers/particle_halo_exchanger.cu
  exchangers/particle_redistributor.cu
  field/from_file.cu
  integrators/const_omega.cu
  integrators/oscillate.cu
  integrators/rigid_vv.cu
  integrators/translate.cu
  integrators/vv.cu
  interactions/membrane/base_membrane.cu
  interactions/membrane/factory.cu
  interactions/membrane/prerequisites.cu
  interactions/obj_rod_binding.cu
  interactions/pairwise/factory.cu
  interactions/rod.cu
  object_belonging/mesh_belonging.cu
  object_belonging/object_belonging.cu
  object_belonging/rod_belonging.cu
  object_belonging/shape_belonging.cu
  pvs/utils/compute_com_extents.cu
  rigid/operations.cu
  walls/simple_stationary_wall.cu
  walls/wall_helpers.cu
  walls/wall_with_velocity.cu
  )

add_library(${LIB_MIR_CORE} STATIC ${sources_cpp} ${sources_cu})


target_include_directories(${LIB_MIR_CORE} PUBLIC ${MIR_BASE_INCLUDE_DIR} )
target_include_directories(${LIB_MIR_CORE} PUBLIC ${MIR_BASE_INCLUDE_DIR}/extern/cuda_variant/ )
target_include_directories(${LIB_MIR_CORE} PUBLIC ${MPI_CXX_INCLUDE_DIRS})
target_include_directories(${LIB_MIR_CORE} PUBLIC ${CUDA_INCLUDE_DIRS})


target_link_libraries(${LIB_MIR_CORE} PUBLIC MPI::MPI_CXX)
target_link_libraries(${LIB_MIR_CORE} PUBLIC ${CUDA_LIBRARIES})
target_link_libraries(${LIB_MIR_CORE} PUBLIC mpark_variant)
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

target_compile_features(${LIB_MIR_CORE} PUBLIC cxx_std_14)

target_compile_options(${LIB_MIR_CORE} PRIVATE
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

# -fPIC
set_property(TARGET ${LIB_MIR_CORE} PROPERTY POSITION_INDEPENDENT_CODE ON)

# set options

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

# Alias

add_library(mirheo::core ALIAS ${LIB_MIR_CORE})

# make sure these sources are not used elsewhere
unset(sources_cpp)
unset(sources_cu)
