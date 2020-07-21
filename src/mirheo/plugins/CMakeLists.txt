set(sources_cpp
  channel_dumper.cpp
  dump_mesh.cpp
  dump_particles_with_mesh.cpp
  dump_xyz.cpp
  factory.cpp
  particle_channel_saver.cpp
  utils/time_stamp.cpp
  utils/xyz.cpp
  )

set(sources_cu
  add_force.cu
  add_torque.cu
  anchor_particle.cu
  average_flow.cu
  average_relative_flow.cu
  berendsen_thermostat.cu
  density_control.cu
  displacement.cu
  dump_obj_stats.cu
  dump_particles.cu
  exchange_pvs_flux_plane.cu
  force_saver.cu
  impose_profile.cu
  impose_velocity.cu
  magnetic_orientation.cu
  membrane_extra_force.cu
  outlet.cu
  particle_checker.cu
  particle_drag.cu
  pin_object.cu
  pin_rod_extremity.cu
  rdf.cu
  stats.cu
  temperaturize.cu
  vacf.cu
  velocity_control.cu
  velocity_inlet.cu
  virial_pressure.cu
  wall_force_collector.cu
  wall_repulsion.cu
  )

add_library(${LIB_MIR_CORE_AND_PLUGINS} STATIC ${sources_cpp} ${sources_cu})

target_include_directories(${LIB_MIR_CORE_AND_PLUGINS} PUBLIC ${MIR_BASE_INCLUDE_DIR} )
target_include_directories(${LIB_MIR_CORE_AND_PLUGINS} PUBLIC ${MPI_CXX_INCLUDE_DIRS})
target_include_directories(${LIB_MIR_CORE_AND_PLUGINS} PUBLIC ${CUDA_INCLUDE_DIRS})

target_link_libraries(${LIB_MIR_CORE_AND_PLUGINS} PUBLIC ${LIB_MIR_CORE})

if (${HDF5_FOUND})
  target_include_directories(${LIB_MIR_CORE_AND_PLUGINS} PRIVATE ${HDF5_INCLUDE_DIRS})
  target_link_libraries(${LIB_MIR_CORE_AND_PLUGINS} PRIVATE ${HDF5_LIBRARIES})
endif()


# Compilation flags

target_compile_features(${LIB_MIR_CORE_AND_PLUGINS} PUBLIC cxx_std_14)

target_compile_options(${LIB_MIR_CORE_AND_PLUGINS} PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:${cxx_private_flags}>
  $<$<COMPILE_LANGUAGE:CUDA>:${cuda_private_flags}>
  )

# -fPIC
set_property(TARGET ${LIB_MIR_CORE_AND_PLUGINS} PROPERTY POSITION_INDEPENDENT_CODE ON)

# Alias
add_library(mirheo::coreAndPlugins ALIAS ${LIB_MIR_CORE_AND_PLUGINS})

# make sure these sources are not used elsewhere
unset(sources_cpp)
unset(sources_cu)
