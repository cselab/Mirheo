target_sources(${LIB_MIR_CORE} PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/interface.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/simple_stationary_wall.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/wall_helpers.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/wall_with_velocity.cu
  )

add_subdirectory(stationary_walls)
