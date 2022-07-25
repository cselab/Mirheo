target_sources(${LIB_MIR_CORE} PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/interface.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/mesh_belonging.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/object_belonging.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/rod_belonging.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/shape_belonging.cu
  )
