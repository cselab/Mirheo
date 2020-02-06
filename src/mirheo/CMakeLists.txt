add_subdirectory(core)
add_subdirectory(plugins)

if (BUILD_PYTHON_MODULE)
  add_subdirectory(bindings)
endif()
