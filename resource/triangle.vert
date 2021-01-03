#version 460
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec3 inColor;

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
};

layout(location = 0) smooth out vec3 vColor;

void main(){
    gl_Position = proj * view * model * vec4(inPosition, 0.0, 1.0);
    vColor = inColor;
}