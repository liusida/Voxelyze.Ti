cmake_minimum_required(VERSION 3.1.0)
#set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
#set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")

project(Voxelyze.Ti VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14) # set C++ standard to C++14

set(INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/include)

set(VX_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/include/VX)
set(VX_SRC_DIR ${CMAKE_CURRENT_LIST_DIR}/src/VX/*.c*)
file(GLOB VX_SRC_FILES ${VX_SRC_DIR})

set(TI_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/include/Ti)
set(TI_SRC_DIR ${CMAKE_CURRENT_LIST_DIR}/src/Ti/*.c*)
file(GLOB TI_SRC_FILES ${TI_SRC_DIR})
# include(CTest)
# enable_testing()

add_executable(Voxelyze.Ti main.cu ${VX_SRC_FILES} ${TI_SRC_FILES})
set_target_properties(Voxelyze.Ti PROPERTIES CUDA_SEPARABLE_COMPILATION ON) # needed for library compilation
target_include_directories(Voxelyze.Ti PUBLIC ${INCLUDE_DIR} ${VX_INCLUDE_DIR} ${TI_INCLUDE_DIR})

# set(CPACK_PROJECT_NAME ${PROJECT_NAME})
# set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
# include(CPack)
