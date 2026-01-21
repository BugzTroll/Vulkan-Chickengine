#version 460

layout (location = 0) in vec3 vPosition;
layout (location = 1) in vec3 vNormal;
layout (location = 2) in vec3 vColor;
layout (location = 3) in vec2 vTexCoord;

layout (location = 0) out vec3 outColor;
layout (location = 1) out vec2 outTexCoord;

struct ObjectData
{
	mat4 model;
};

struct Camera
{
	mat4 view;
	mat4 proj;
	mat4 viewProj;
};

// UNIFORMS //
layout (set = 0, binding = 0) uniform SceneData
{
	vec4 fogColor; // w : exponent
	vec4 fogDistances; //x : min, y : max, zw:unused
	vec4 ambiantColor;
	vec4 sunlightDirection; //w: sun power
	vec4 sunlightColor;
	Camera cameraData;
}sceneData;

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
	mat4 transformMatrix = (sceneData.cameraData.viewProj * modelMatrix);
	gl_Position = transformMatrix * vec4(vPosition, 1.f);
	outColor = vColor;
	outTexCoord = vTexCoord;
	//using Push constants
	//mat4 transformMatrix = (cameraData.viewProj * PushConstants.render_matrix);
	//gl_Position = transformMatrix * vec4(vPosition, 1.f);
	//outColor = vColor;
}