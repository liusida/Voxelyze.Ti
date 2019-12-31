project(glGraphics CXX C CUDA)
cmake_minimum_required(VERSION 3.10)
file(GLOB SRC_FILE *.c*)
add_library(glGraphics_lib ${SRC_FILE})
include_directories(include)

set_property(TARGET glGraphics_lib PROPERTY CXX_STANDARD 17)

#for visualization
set(OpenGL_GL_PREFERENCE GLVND)
if (UNIX)
    find_package(OpenGL REQUIRED) # find and include OpenGL
    if (OpenGL_FOUND)
        message(STATUS "OPENGL FOUND")
        target_include_directories(glGraphics_lib PRIVATE ${OpenGL_INCLUDE_DIRS})
        target_link_libraries(glGraphics_lib PRIVATE ${OpenGL_LIBRARIES} GL)
    endif()
else()
    find_package(OPENGL REQUIRED) # find and include OpenGL
    if (OPENGL_FOUND)
        message(STATUS "OPENGL FOUND")
        target_include_directories(glGraphics_lib PRIVATE ${OPENGL_INCLUDE_DIRS})
        target_link_libraries(glGraphics_lib PRIVATE ${OPENGL_LIBRARIES})
    endif()
endif()

find_package(glfw3 CONFIG REQUIRED)
if (glfw3_FOUND)
    message(STATUS "GLFW FOUND")
    target_include_directories(glGraphics_lib PRIVATE ${glfw3_INCLUDE_DIRS})
    target_link_libraries(glGraphics_lib PRIVATE ${glfw3_LIBRARIES})
endif()

find_package(GLEW REQUIRED) # GLEW
if (GLEW_FOUND)
    message(STATUS "GLEW FOUND")
    target_include_directories(glGraphics_lib PRIVATE ${GLEW_INCLUDE_DIRS})
    target_link_libraries(glGraphics_lib PRIVATE ${GLEW_LIBRARIES} glfw)
endif()

find_package(glm CONFIG REQUIRED) # glm
if (glm_FOUND)
    message(STATUS "GLM FOUND")
    target_include_directories(glGraphics_lib PRIVATE ${glm_INCLUDE_DIRS})
    target_link_libraries(glGraphics_lib PRIVATE ${glm_LIBRARIES} glm)
endif()
