struct VSOutput
{
	float4 position : SV_POSITION;
	float3 color : COLOR;
}

[shader("vertex")]
VSOutput vsmain(uint vertexID : SV_VertexID)
{
	const float3 positions[3] = float3[3]
	(
		float3(1.f, 1.f, 0.f),
		float3(-1.f, 1.f, 0.f),
		float3(0.f, -1.f, 0.f)
	);

	const float3 color[3] = float3[3]
	(
		float3(1.f, 0.f, 0.f),
		float3(0.f, 1.f, 0.f),
		float3(0.f, 0.f, 1.f)
	);

	VSOutput output = VSOutput(0);
	output.position = float4(positions[vertexID], 1.0f);
	output.color = color[vertexID];
	return output;
}

[shader("pixel")]
float4 psmain(float3 color : COLOR) : SV_Target
{
 	return float4(color.x, color.y, color.z, 0.f);
}