target_sources(${LIB_MIR_CORE} PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/exchange_entity.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/interface.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/object_halo_exchanger.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/object_halo_extra_exchanger.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/object_redistributor.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/object_reverse_exchanger.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/particle_halo_exchanger.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/particle_redistributor.cu
  )

add_subdirectory(engines)
