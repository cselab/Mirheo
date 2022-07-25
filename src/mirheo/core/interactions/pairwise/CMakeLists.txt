target_sources(${LIB_MIR_CORE} PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/base_pairwise.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/density.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/dpd.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/dpd_visco_elastic.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/factory.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/factory_helper.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/growing_repulsive_lj.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/lj.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/mdpd.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/morse.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/norandom_dpd.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/repulsive_lj.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/sdpd.cu
  )
