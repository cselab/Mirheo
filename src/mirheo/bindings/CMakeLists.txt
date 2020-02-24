set(sources_cpp
  bindings.cpp
  bouncers.cpp
  initial_conditions.cpp
  integrators.cpp
  interactions.cpp
  mirheo.cpp
  object_belonging_checkers.cpp
  particle_vectors.cpp
  plugins.cpp
  vector_types.cpp
  utils.cpp
  walls.cpp
  )

add_library(${LIB_MIR} MODULE ${sources_cpp})

target_link_libraries(${LIB_MIR} PRIVATE ${LIB_MIR_CORE_AND_PLUGINS})
target_link_libraries(${LIB_MIR} PRIVATE pybind11::module)
set_target_properties(${LIB_MIR} PROPERTIES
  PREFIX "${PYTHON_MODULE_PREFIX}"
  SUFFIX "${PYTHON_MODULE_EXTENSION}")

# Compilation flags

target_compile_features(${LIB_MIR} PUBLIC cxx_std_14)

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
