#pragma once

#include "glm/glm.hpp"
#include <memory>
#include <vector>
#include "vk_types.h"
#include "vk_mesh.h"

struct DrawContext;

class IRenderable
{
	virtual void Draw(const glm::mat4& topMatrix, DrawContext& ctx) = 0;
};

struct Node : public IRenderable
{
	//parent pointer must be a weak pointer to avoid circular dependencies
	std::weak_ptr<Node> parent;
	std::vector<std::shared_ptr<Node>> children;

	glm::mat4 localTransform;
	glm::mat4 worldTransform;

	void RefreshTransform(const glm::mat4& parentMatrix);
	virtual void Draw(const glm::mat4& topMatrix, DrawContext& ctx);
};

struct MeshNode : public Node
{
	std::shared_ptr<MeshAsset> mesh;
	virtual void Draw(const glm::mat4& topMatrix, DrawContext& ctx);
};