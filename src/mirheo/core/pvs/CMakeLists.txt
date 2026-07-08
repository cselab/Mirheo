target_sources(${LIB_MIR_CORE} PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/chain_vector.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/data_manager.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/membrane_vector.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/object_vector.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/particle_vector.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rigid_ashape_object_vector.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rigid_object_vector.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rod_vector.cpp
  )

add_subdirectory(checkpoint)
add_subdirectory(packers)
add_subdirectory(restart)
add_subdirectory(utils)
add_subdirectory(views)
