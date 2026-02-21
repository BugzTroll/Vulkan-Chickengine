
struct VSOutput
{
    float4 position : SV_POSITION;
    float3 color : COLOR;
    float2 uv : TEXCOORD;
}

[vk::binding(0, 2)]
Sampler2D textureSampler;

[shader("pixel")]
float4 main(VSOutput vertIn) : SV_Target
{
    float3 color = textureSampler.Sample(vertIn.uv).xyz;
    return float4(color, 1.0f);           // set color
}