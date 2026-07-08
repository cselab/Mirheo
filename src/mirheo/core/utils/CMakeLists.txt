target_sources(${LIB_MIR_CORE} PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/common.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/compile_options.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/file_wrapper.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/nvtx.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/path.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/stacktrace_explicit.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/strprintf.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/unique_mpi_comm.cpp
  )
