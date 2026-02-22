#pragma once

#include <vk_types.h>
#include <unordered_map>
#include <filesystem>
#include <vk_mesh.h>
#include <string>
#include <vk_descriptors.h>
#include <renderable.h>
#include <memory>

#include <fastgltf/parser.hpp>

struct DrawContext;
class VulkanEngine;

struct LoadedGLTF : public IRenderable 
{
	std::unordered_map<std::string, std::shared_ptr<MeshAsset>> meshes;
	std::unordered_map<std::string, std::shared_ptr<Node>> nodes;
	std::unordered_map<std::string, AllocatedImage> images;
	std::unordered_map<std::string, std::shared_ptr<GLTFMaterial>> materials;

	//nodes that dont have a parent, for iterating trough the file in tree order
	std::vector<std::shared_ptr<Node>> topNodes;
	std::vector<VkSampler> samplers;
	DescriptorAllocatorGrowable descriptorPool;
	AllocatedBuffer materialDataBuffer;
	VulkanEngine* engine;

	~LoadedGLTF() { clearAll(); };

	virtual void Draw(const glm::mat4& topMatrix, DrawContext& ctx) override;

private:
	void clearAll();
};

std::optional<std::vector<std::shared_ptr<MeshAsset>>> loadGltfMeshes(VulkanEngine* engine, std::filesystem::path filePath);
std::optional<std::shared_ptr<LoadedGLTF>> loadGltf(VulkanEngine* engine, std::string_view filePath);
std::optional<AllocatedImage> loadImage(VulkanEngine* engine, fastgltf::Asset& asset, fastgltf::Image& image);

