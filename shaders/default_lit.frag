#version 450

//shader input
layout (location = 0) in vec3 inColor;

//shader output
layout (location = 0) out vec4 outFragColor;

//uniforms
layout (set = 0, binding = 1) uniform SceneData
{
	vec4 fogColor; // w : exponent
	vec4 fogDistances; //x : min, y : max, zw:unused
	vec4 ambiantColor;
	vec4 sunlightDirection; //w: sun power
	vec4 sunlightColor;
}sceneData;

void main()
{
	outFragColor = vec4(inColor + sceneData.ambiantColor.xyz, 1.0f); //adding ambiant color
	//outFragColor = vec4(1.f, 0.f, 0.f, 1.0f); //adding ambiant color
}