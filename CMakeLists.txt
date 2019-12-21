cmake_minimum_required(VERSION 3.1.0)
#set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
#set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")

project(Voxelyze.Ti VERSION 0.1.0 LANGUAGES CXX CUDA)

message(${CMAKE_CURRENT_LIST_DIR})
set(INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/include/VX)
set(SRC_DIR ${CMAKE_CURRENT_LIST_DIR}/src/VX/*.c*)
file(GLOB SRC_FILES ${SRC_DIR})

set(TI_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/include/Ti)
set(TI_SRC_DIR ${CMAKE_CURRENT_LIST_DIR}/src/Ti/*.c*)
file(GLOB TI_SRC_FILES ${TI_SRC_DIR})
# include(CTest)
# enable_testing()

add_executable(Voxelyze.Ti main.cu ${SRC_FILES} ${TI_SRC_FILES})
target_include_directories(Voxelyze.Ti PUBLIC ${INCLUDE_DIR} ${TI_INCLUDE_DIR})

# set(CPACK_PROJECT_NAME ${PROJECT_NAME})
# set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
# include(CPack)
