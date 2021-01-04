#version 460
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) smooth in vec3 vColor;
layout(location = 1) smooth in vec2 vTexCoord;
layout(location = 0) out vec4 fragColor;

layout(binding = 1) uniform sampler2D texSampler;

void main(){
    //fragColor = texture(texSampler, vTexCoord);
    fragColor = vec4(vColor, 1.0);
}