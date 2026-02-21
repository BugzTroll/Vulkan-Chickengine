#include "renderable.h"

#include "vk_engine.h"

void Node::RefreshTransform(const glm::mat4& parentMatrix)
{
	worldTransform = parentMatrix * localTransform;
	for (auto c : children)
	{
		c->RefreshTransform(worldTransform);
	}
}

void Node::Draw(const glm::mat4& topMatrix, DrawContext& ctx)
{
	for (auto& c : children)
	{
		c->Draw(topMatrix, ctx);
	}
}

void MeshNode::Draw(const glm::mat4& topMatrix, DrawContext& ctx)
{
	glm::mat4 nodeMatrix = topMatrix * worldTransform;

	for (auto& s : mesh->surfaces)
	{
		RenderObject2 def;
		def.indexCount = s.count;
		def.indexBuffer = mesh->meshBuffers._indexBuffer._buffer;
		def.firstIndex = s.startIndex;
		def.materialInstance = &s.material->data;

		def.transform = nodeMatrix;
		def.vertexBufferAddress = mesh->meshBuffers.vertexBufferAddress;
		ctx.OpaqueSurfaces.push_back(def);
	}
}
