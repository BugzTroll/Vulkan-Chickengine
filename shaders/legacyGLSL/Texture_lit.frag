#version 450

#extension GL_KHR_vulkan_glsl : enable

//shader input
layout (location = 0) in vec3 inColor;
layout (location = 1) in vec2 inTexCoord;

//shader output
layout (location = 0) out vec4 outFragColor;

struct Camera
{
	mat4 view;
	mat4 proj;
	mat4 viewProj;
};

//uniforms
layout (set = 0, binding = 0) uniform SceneData
{
	vec4 fogColor; // w : exponent
	vec4 fogDistances; //x : min, y : max, zw:unused
	vec4 ambiantColor;
	vec4 sunlightDirection; //w: sun power
	vec4 sunlightColor;
	Camera cameraData;
}sceneData;

layout (set = 2, binding = 0 ) uniform sampler2D tex1;

void main()
{
	vec3 color = texture(tex1, inTexCoord).xyz; //sample texture
	outFragColor = vec4(color, 1.0f); //set color
}