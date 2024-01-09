#include <fluid.cuh>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include "params.h"

// 着色器源码
const char *vertexShaderSource = "#version 330 core\n"
    "layout (location = 0) in vec3 aPos;\n"
    "layout (location = 1) in vec2 aTexCoord;\n"
    "out vec2 TexCoord;\n"
    "void main()\n"
    "{\n"
    "   gl_Position = vec4(aPos, 1.0);\n"
    "   TexCoord = aTexCoord;\n"
    "}\0";

const char *fragmentShaderSource = "#version 330 core\n"
    "out vec4 FragColor;\n"
    "in vec2 TexCoord;\n"
    "uniform sampler2D ourTexture;\n"
    "void main()\n"
    "{\n"
    "   FragColor = texture(ourTexture, TexCoord);\n"
    "}\0";

void add_force(FluidCUDA &fluid, int i) {
    // 0.1 < x < 0.3, 0.4 < y < 0.6, 0.4 < z < 0.6
    for (int x = 0.2F * CELLS_X; x <= 0.3F * CELLS_X; x++) {
        for (int y = 0.45F * CELLS_Y; y <= 0.55F * CELLS_Y; y++) {
            for (int z = 0.45F * CELLS_Z; z <= 0.55F * CELLS_Z; z++) {
                fluid.add_U_x_force_at(z - 5, y, x, FORCE_SCALE * (sinf(i / 80.0F) + 1));
                fluid.add_source_at(z - 5, y, x, 0, 0.1F);
            }
        }
    }

    // // 0.7 < x < 0.9, 0.4 < y < 0.6, 0.4 < z < 0.6
    for (int x = 0.7F * CELLS_X; x <= 0.8F * CELLS_X; x++) {
        for (int y = 0.45F * CELLS_Y; y <= 0.55F * CELLS_Y; y++) {
            for (int z = 0.45F * CELLS_Z; z <= 0.55F * CELLS_Z; z++) {
                fluid.add_U_x_force_at(z + 5, y, x, -FORCE_SCALE * (sinf(i / 80.0F) + 1));
                fluid.add_source_at(z + 5, y, x, 1, 0.1F);
            }
        }
    }
}

FluidCUDA fluid;
bool      simulateFlag = false;
void      processInput(GLFWwindow *window);

int main() {
    // 初始化和配置 GLFW
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // 创建窗口
    GLFWwindow *window = glfwCreateWindow(600, 600, "LearnOpenGL", NULL, NULL);
    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

    if (window == NULL) {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    // 加载所有OpenGL函数指针
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    // 编译着色器
    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);
    // 检查编译错误（省略）

    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);
    // 检查编译错误（省略）

    unsigned int shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    // 检查链接错误（省略）

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    // 设置顶点数据和缓冲区，并配置顶点属性
    float vertices[] = {
        // 位置          // 纹理坐标
        1.0f,
        1.0f,
        0.0f,
        1.0f,
        1.0f,
        // 右上角
        1.0f,
        -1.0f,
        0.0f,
        1.0f,
        0.0f,
        // 右下角
        -1.0f,
        -1.0f,
        0.0f,
        0.0f,
        0.0f,
        // 左下角
        -1.0f,
        1.0f,
        0.0f,
        0.0f,
        1.0f // 左上角
    };
    unsigned int indices[] = {
        0,
        1,
        3,
        // 第一个三角形
        1,
        2,
        3 // 第二个三角形
    };

    unsigned int VBO, VAO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // 位置属性
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(0);
    // 纹理坐标属性
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // 创建纹理
    unsigned int texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    // 设置纹理环绕和过滤方式
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // 假设图像大小和格式已知
    int width, height;
    width  = 600;
    height = 600;

    fluid.init();
    int i = 0;

    // 渲染循环
    while (!glfwWindowShouldClose(window)) {
        // 输入
        processInput(window);

        if (simulateFlag) {
            add_force(fluid, i++);
            fluid.step();
        }

        fluid.render();
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_FLOAT, fluid.get_render_buffer());

        // 渲染
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // 绑定纹理
        glBindTexture(GL_TEXTURE_2D, texture);

        // 绘制物体
        glUseProgram(shaderProgram);
        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

        // 交换缓冲区和轮询IO事件
        glfwSwapBuffers(window);
        glfwPollEvents();


    }

    // 释放资源
    glfwTerminate();
    return 0;
}

void processInput(GLFWwindow *window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
        fluid.rot_left(0.1);
    }

    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
        fluid.rot_left(-0.1);
    }

    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) {
        fluid.rot_up(0.1);
    }

    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) {
        fluid.rot_up(-0.1);
    }

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
        fluid.zoom_in(0.1);
    }

    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
        fluid.zoom_in(-0.1);
    }

    if (glfwGetKey(window, GLFW_KEY_ENTER) == GLFW_PRESS) {
        fluid.cleanup();
        fluid.init();
    }

    if (glfwGetKey(window, GLFW_KEY_P) == GLFW_PRESS) {
        simulateFlag = false;
    }

    if (glfwGetKey(window, GLFW_KEY_O) == GLFW_PRESS) {
        simulateFlag = true;
    }
}
