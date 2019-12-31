project(Voxelyze VERSION 3.0 LANGUAGES CUDA)
cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 14) # set C++ standard to C++14

file(GLOB TI_SRC_FILE Ti/*.c*)
file(GLOB VX_SRC_FILE VX/*.c*)
add_library(Voxelyze_lib ${VX_SRC_FILE} ${TI_SRC_FILE})
set_target_properties(Voxelyze_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON) # needed for library compilation
include_directories(include include/Ti include/VX)

#enable cuda debug
target_compile_options(Voxelyze_lib PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-G -arch=compute_60 -code=compute_60>)
set(CMAKE_VERBOSE_MAKEFILE ON)
