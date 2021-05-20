include(GoogleTest)
option(INSTALL_GTEST OFF)
option(BUILD_GMOCK OFF)

add_subdirectory("extern/googletest")

function (add_test_executable dirName nodes)
  # Find sources
  file(GLOB SOURCES RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/${dirName}/*.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/${dirName}/*.cpp")

  string(REGEX REPLACE "/" "_" execNameId ${dirName})

  set(EXEC_NAME "test_${execNameId}")

  add_executable(${EXEC_NAME} ${SOURCES})
  target_link_libraries(${EXEC_NAME} PRIVATE ${CUDA_LIBRARIES} ${LIB_MIR_CORE} gtest)

  add_test(NAME ${EXEC_NAME}
           COMMAND mir.run --runargs "-n ${nodes}" ./${EXEC_NAME})
endfunction()

add_test_executable(celllists 1)
add_test_executable(file_wrapper 1)
add_test_executable(id64 1)
add_test_executable(integration/particles 1)
add_test_executable(integration/rigid 1)
add_test_executable(interaction/dpd 1)
add_test_executable(quaternion 1)
add_test_executable(map 1)
add_test_executable(mesh 1)
add_test_executable(inertia_tensor 1)
add_test_executable(marching_cubes 1)
add_test_executable(onerank 1)
add_test_executable(packers/exchange 1)
add_test_executable(packers/redistribute 1)
add_test_executable(packers/simple 1)
add_test_executable(pid 1)
add_test_executable(reduce 1)
add_test_executable(restart 4)
add_test_executable(rng 1)
add_test_executable(rod/discretization 1)
add_test_executable(rod/energy 1)
add_test_executable(rod/forces 1)
add_test_executable(roots 1)
add_test_executable(scheduler 1)
add_test_executable(serializer 1)
add_test_executable(snapshot 1)
add_test_executable(str_types 1)
add_test_executable(triangle_invariants 1)
add_test_executable(utils 1)
add_test_executable(variant 1)
add_test_executable(warpScan 1)

if (MIR_ENABLE_SANITIZER)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=undefined -g")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=undefined")
endif()
