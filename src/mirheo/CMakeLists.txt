add_subdirectory(core)
add_subdirectory(plugins)

if (MIR_BUILD_PYTHON_MODULE)
  add_subdirectory(bindings)
endif()
