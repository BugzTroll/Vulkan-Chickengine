struct VSOutput
{
    float4 position : SV_POSITION;
    float3 normal : NORMAL;
	float3 color : COLOR;
	float2 uv: TEXCOORD;
}

//TODO maybe not with attributes like this
struct VSInput
{
	float3 position : POSITION;
	float3 normal : NORMAL;
	float3 color: COLOR;
	float2 uv: TEXCOORD;
}

struct CameraData
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

    CameraData camera;
}

struct GLTFMaterialData
{
    float4 colorFactors;
    float4 metalRoughFactors;
}

struct ObjectData
{
    float4x4 renderMatrix;
    VSInput* vertexBuffer;
}

[vk::binding(0, 0)]
ConstantBuffer<SceneData> sceneUbo;

[vk::binding(0, 1)]
ConstantBuffer<GLTFMaterialData> materialDataUbo;

[vk::binding(1, 1)]
Sampler2D colorTexSampler;

[vk::binding(2, 1)]
Sampler2D metalRoughTexSampler;

[vk::push_constant]
uniform ObjectData objectDataPc;

[shader("vertex")]
VSOutput vsmain(uint vertexID: SV_VertexID)
{
    const float3 positions[3] = float3[3]
	(
        float3(1.f, 1.f, 0.f),
        float3(-1.f, 1.f, 0.f),
        float3(0.f, -1.f, 0.f)
	);

    //float4 position = float4(positions[vertexID], 1.0f);

    VSInput v = objectDataPc.vertexBuffer[vertexID];
    float4 position = float4(v.position, 1.0f);

    VSOutput VSOut;
    VSOut.position = mul(sceneUbo.camera.viewProj, mul(objectDataPc.renderMatrix, position));
    VSOut.normal = mul(objectDataPc.renderMatrix, float4(v.normal, 0.f)).xyz;
    VSOut.uv = v.uv;
    VSOut.color = v.color;

    return VSOut;
}

[shader("pixel")]
float4 psmain(VSOutput vertIn): SV_Target
{
    float lightValue = 1.0f; //max(dot(vertIn.normal, sceneUbo.sunlightDirection.xyz), 0.1f);
    float3 color = /*vertIn.color **/ colorTexSampler.Sample(vertIn.uv).xyz;
    float3 ambient = color; //* sceneUbo.ambiantColor.xyz;

    return float4(color * lightValue * 1.0f /*ambientsceneUbo.sunlightColor.w + ambient*/, 1.0f);

    //return float4(1.0f, 0.0f, 0.0f, 1.0f);
}