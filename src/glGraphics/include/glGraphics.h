#if !defined(GL_GRAPHICS_H)
#define GL_GRAPHICS_H
#include <vector>
#include <string>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <stb_image.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "shader_m.h"

class glGraphics
{
private:
    /* data */
public:
    glGraphics(/* args */);
    ~glGraphics();
    bool running();
    int init();
    void draw( const std::vector<float> &points );

    static void framebuffer_size_callback(GLFWwindow* window, int width, int height);
    void processInput();
    // settings
    const unsigned int SCR_WIDTH = 800;
    const unsigned int SCR_HEIGHT = 600;
     GLFWwindow* window;
     Shader ourShader;
     unsigned int VAO, VBO, EBO;

};

#endif // GL_GRAPHICS_H
