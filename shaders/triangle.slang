
struct VSOutput
{
	float4 position : SV_POSITION;
}

[shader("vertex")]
VSOutput vsmain(
uint vertexID : SV_VertexID)
{
	const float3 positions[3] = float3[3]
	(
		float3(1.f, 1.f, 0.f),
		float3(-1.f, 1.f, 0.f),
		float3(0.f, -1.f, 0.f)
	);

	VSOutput output = VSOutput(0);
	output.position = float4(positions[vertexID], 1.0f);
	return output;
}


[shader("pixel")]
float4 psmain() : SV_Target
{
 	return float4(1.f, 0.f, 0.f, 0.f);
}