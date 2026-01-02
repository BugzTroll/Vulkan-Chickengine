#version 460

layout (location = 0) in vec3 vPosition;
layout (location = 1) in vec3 vNormal;
layout (location = 2) in vec3 vColor;

layout (location = 0) out vec3 outColor;

struct ObjectData
{
	mat4 model;
};

// UNIFORMS //

layout(set=0, binding=0) uniform CameraBuffer
{
	mat4 view;
	mat4 proj;
	mat4 viewProj;
} cameraData;

// PUSH CONSTANTS //

//must match the push_constant struct .cpp 1to1
layout (push_constant) uniform constants
{
	vec4 data;
	mat4 render_matrix;
} PushConstants;

// STORAGE BUFFERS //

layout (std140, set = 1, binding = 0) readonly buffer ObjectBuffer
{
	ObjectData objects[];
} objectBuffer;


void main()
{
	//using SB
	mat4 modelMatrix = objectBuffer.objects[gl_BaseInstance].model;
	mat4 transformMatrix = (cameraData.viewProj * modelMatrix);
	gl_Position = transformMatrix * vec4(vPosition, 1.f);
	outColor = vColor;

	//using Push constants
	//mat4 transformMatrix = (cameraData.viewProj * PushConstants.render_matrix);
	//gl_Position = transformMatrix * vec4(vPosition, 1.f);
	//outColor = vColor;
}