cmake_minimum_required(VERSION 3.1.0)
set(ENV{CMAKE_BUILD_PARALLEL_LEVEL} 8)  # building with multiple CPU cores, saving life.

project(Voxelyze.Ti VERSION 0.1.0)

message(${CMAKE_CURRENT_LIST_DIR})
set(INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/include)
set(SRC_DIR ${CMAKE_CURRENT_LIST_DIR}/src/*.c*)
file(GLOB SRC_FILES ${SRC_DIR})

# include(CTest)
# enable_testing()

add_executable(Voxelyze.Ti main.cpp ${SRC_FILES})
target_include_directories(Voxelyze.Ti PUBLIC ${INCLUDE_DIR})

# set(CPACK_PROJECT_NAME ${PROJECT_NAME})
# set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
# include(CPack)
