target_sources(${LIB_MIR_CORE} PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/const_omega.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/interface.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/minimize.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/oscillate.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/rbc_shardlow.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/rigid_vv.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/shear.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/shear_pol_chain.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/sub_step.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/translate.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/vv.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/vv_pol_chain.cu
  )
