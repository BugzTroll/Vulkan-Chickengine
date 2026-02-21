struct VertexOutput
{
    float4 position : SV_position;
}

[shader("mesh")]
[outputtopology("triangle")]
[numthreads(1, 1, 1)]
void msmain(out indices uint3 triangles[1], out vertices VertexOutput vertices[3], uint3 DipatchThreadId: SV_DispatchThreadID)
{
    SetMeshOutputCounts(3, 1);
    vertices[0].position = float4(0.5, -0.5, 0, 1);
    vertices[1].position = float4(0.5, 0.5, 0, 1);
    vertices[2].position = float4(-0.5, 0.5, 0, 1);
    triangles[0] = uint3(0, 1, 2);
}

[shader("fragment")]
float4 psmain()
{
    return float4(1.0, 0.0, 0.0, 1.0);
}