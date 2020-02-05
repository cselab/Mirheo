cmake_minimum_required(VERSION 3.8)

if (POLICY CMP0060)
  cmake_policy(SET CMP0060 NEW)
endif()

if(POLICY CMP0069) # for INTERPROCEDURAL_OPTIMIZATION
  cmake_policy(SET CMP0069 NEW)
endif()

if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

project(MIR_UnitTests LANGUAGES C CXX CUDA)
enable_testing()

# libmirheo
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../" "${CMAKE_CURRENT_SOURCE_DIR}/../build")

# googletest
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/extern/googletest")

function (add_test_executable dirName nodes)
  # Find sources
  file(GLOB SOURCES RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/${dirName}/*.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/${dirName}/*.cpp")

  string(REGEX REPLACE "/" "_" execNameId ${dirName})
  
  set(EXEC_NAME "test_${execNameId}")
  
  add_executable(${EXEC_NAME} ${SOURCES})
  target_link_libraries(${EXEC_NAME} PRIVATE ${CUDA_LIBRARIES})
  target_link_libraries(${EXEC_NAME} PRIVATE mirheoCore)
  target_link_libraries(${EXEC_NAME} PRIVATE gtest)

  add_test(NAME ${EXEC_NAME}
           COMMAND mir.run --runargs "-n ${nodes}" ./${EXEC_NAME})
endfunction()

add_test_executable(celllists 1)
add_test_executable(file_wrapper 1)
add_test_executable(id64 1)
add_test_executable(integration/particles 1)
add_test_executable(integration/rigid 1)
add_test_executable(interaction 1)
add_test_executable(quaternion 1)
add_test_executable(map 1)
add_test_executable(inertia_tensor 1)
add_test_executable(marching_cubes 1)
add_test_executable(object_deleter 1)
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
add_test_executable(variant 1)
add_test_executable(warpScan 1)


# Linker flags
set(CMAKE_LINK_FLAGS "${CMAKE_LINK_FLAGS} -rdynamic -flto -g")

# Choose Release mode as default.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
    "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()

# uncomment to use sanitizer
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=undefined -g")
#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=undefined")
