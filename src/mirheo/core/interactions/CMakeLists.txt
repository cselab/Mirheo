target_sources(${LIB_MIR_CORE} PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/factory.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/interface.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/obj_binding.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/obj_rod_binding.cu
  )

add_subdirectory(chain)
add_subdirectory(membrane)
add_subdirectory(pairwise)
add_subdirectory(rod)
add_subdirectory(utils)
