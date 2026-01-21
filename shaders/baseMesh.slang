struct VSOutput
{
	float4 position : SV_POSITION;
	float3 color : COLOR;
	float2 uv: TEXCOORD;
}

struct VSInput
{
	float3 position : POSITION;
	float3 normal : NORMAL;
	float3 color: COLOR;
	float2 uv: TEXCOORD;
}

struct Camera
{
	float4x4 view;
	float4x4 proj;
	float4x4 viewProj;
}

struct SceneData
{
	float4 fogColor;
	float4 fogDistances;
	float4 ambiantColor;
    float4 sunlightDirection;
    float4 sunlightColor;
	Camera cameraData;
}

struct ObjectData
{
	float4x4 model;
}

[vk::binding(0, 0)]
ConstantBuffer<SceneData> sceneUbo; //Scene info like camera, fog values, dir lights ect

[vk::binding(0, 1)]
StructuredBuffer<ObjectData> objSb; //per object data aka : model transform

[shader("vertex")]
VSOutput vsmain(uint instanceID: SV_StartInstanceLocation, VSInput input)
{
    float4x4 modelMatrix = objSb[instanceID].model;
    float4x4 transformMatrix = mul(sceneUbo.cameraData.viewProj, modelMatrix);
    VSOutput output = VSOutput(0);
    float4 transformedPoint = mul(transformMatrix, float4(input.position, 1.f));
    output.position = transformedPoint;
	output.color = input.color;
	output.uv = input.uv;
	return output;
}

[shader("pixel")]
float4 psmain(VSOutput vertIn): SV_Target
{
    return float4(vertIn.color + sceneUbo.ambiantColor.xyz, 1.0f);
 	//return float4(vertIn.uv.x, vertIn.uv.y, 0.f, 1.f);
}