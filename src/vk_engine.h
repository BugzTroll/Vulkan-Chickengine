// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <vk_types.h>
#include "glm/glm.hpp"

#include "vk_mesh.h"
#include "camera.h"
#include "vk_descriptors.h"
#include "vk_loader.h"
#include "renderable.h"

constexpr unsigned int FRAME_OVERLAP = 2;

class VulkanEngine;

struct GLTFMetallic_Roughness
{
	Material opaque;
	Material transparent;

	VkDescriptorSetLayout materialLayout;

	struct MaterialConstants
	{
		glm::vec4 colorFactors;
		glm::vec4 metalRoughFactors;
		glm::vec4 extra[14];
	};

	struct MaterialResources
	{
		AllocatedImage colorImage;
		VkSampler colorSampler;
		AllocatedImage metalRoughImage;
		VkSampler metalRoughSampler;
		VkBuffer dataBuffer;
		uint32_t dataBufferOffset;
	};

	DescriptorWriter descriptorWriter;

	void buildPipelines(VulkanEngine* engine);
	void clearResources(VkDevice device);

	MaterialInstance writeMaterial(VkDevice device, EMaterialPass pass, const MaterialResources& resources, DescriptorAllocatorGrowable& descriptorAllocator);
};

struct RenderingObject
{
	uint32_t indexCount;
	uint32_t firstIndex;

	VkBuffer indexBuffer;

	MaterialInstance* material;

	glm::mat4 transform;
	VkDeviceAddress vertexBufferAddress;
};

struct DeletionQueue
{
	std::deque<std::function<void()>> deletors;

	void pushFunction(std::function<void()>&& fn)
	{
		deletors.push_back(fn);
	}

	void flush()
	{
		for (auto it = deletors.begin(); it != deletors.end(); it++)
		{
			(*it)();
		}

		deletors.clear();
	}
};

struct Texture
{
	AllocatedImage image;
	VkImageView imageView;
};

struct UploadContext
{
	VkFence _uploadFence;
	VkCommandPool _commandPool;
	VkCommandBuffer commandBuffer;
};

struct GPUObjectData
{
	glm::mat4 modelMatrix;
};

struct GPUCameraData
{
	glm::mat4 view;
	glm::mat4 proj;
	glm::mat4 viewProj;
};

struct GPUSceneData
{
	glm::vec4 fogColor;
	glm::vec4 fogDistances;
	glm::vec4 ambiantColor;
	glm::vec4 sunlightDirection;
	glm::vec4 sunlightColor;

	GPUCameraData cameraData;
};

struct FrameData
{
	//sync structures
	VkSemaphore _presentSemaphore;
	VkFence _renderFence;

	//cmd buffer
	VkCommandPool _commandPool;
	VkCommandBuffer _mainCommandBuffer;
	DeletionQueue _deletionQueue;

	DescriptorAllocatorGrowable _frameDescriptors;

	//objects transform SB
	AllocatedBuffer _objectBuffer;
	VkDescriptorSet _objectDescriptor;
};

struct RenderObject
{
	Mesh* mesh;
	Material* material;
	glm::mat4 transformMatrix;
};

struct RenderObject2
{
	uint32_t indexCount;
	uint32_t firstIndex;
	VkBuffer indexBuffer;
	glm::mat4 transform;

	MaterialInstance* materialInstance;
	VkDeviceAddress vertexBufferAddress;
};

struct DrawContext
{
	std::vector<RenderObject2> OpaqueSurfaces;
	std::vector<RenderObject2> TranslucentSurfaces;
};

struct MeshPushConstants
{
	glm::vec4 data;
	glm::mat4 renderMatrix;
};

class VulkanEngine {
public:

	//vulkan basic setup
	VkInstance _instance;
	VkPhysicalDevice _chosenGPU;
	VkDevice _device;
	VkSurfaceKHR _surface;
	VkDebugUtilsMessengerEXT _debug_messenger; // debug messages will be sent there
	VkPhysicalDeviceProperties _gpuProperties;

	//swapchain setup
	VkSwapchainKHR _swapchain;
	std::vector<VkSemaphore> _renderSemaphores; // one per swapchain image AND NOT ONE PER FRAME!

	VkFormat _swapchainFormat;
	std::vector<VkImage> _swapchainImages;
	std::vector<VkImageView> _swapchainImageViews;

	//command Queue/buffer setup
	VkQueue _graphicsQueue;
	uint32_t _graphicsQueueFamilyIdx;

	//renderpass
	VkRenderPass _renderPass;
	std::vector<VkFramebuffer> _framebuffers; //equivalent to images and imageview array

	//shader selection
	int _selectedShader{0};

	//allocator
	VmaAllocator _allocator;
	DescriptorAllocatorGrowable _globalDescriptorAllocator;

	//Deletors
	DeletionQueue _mainDeletionQueue;

	//Depth buffer
	VkImageView _depthImageView;
	AllocatedImage _depthImage;
	VkFormat _depthFormat;

	//render obj
	std::vector<RenderObject> _renderables;

	//frames
	FrameData _frames[FRAME_OVERLAP];
	FrameData& getCurrentFrame();

	//descriptor layout
	VkDescriptorSetLayout _globalSetLayout;
	VkDescriptorSetLayout _objectSetLayout;
	VkDescriptorSetLayout _singleTextureSetLayout;
	VkDescriptorSetLayout _meshShaderLayout;
	VkDescriptorPool _descriptorPool;

	VkDescriptorSetLayout _materialLayout;

	//Scene
	GPUSceneData _sceneParameters;
	AllocatedBuffer _sceneParameterBuffer;

	// Materials and meshes
	std::unordered_map<std::string, Material> _materials;
	std::unordered_map<std::string, Mesh> _meshes;
	std::unordered_map<std::string, Texture> _textures;

	std::vector<std::shared_ptr<MeshAsset>> testMeshes;

	//upload contect
	UploadContext _uploadContext;

	//camera
	Camera _camera;

	//default images
	AllocatedImage _whiteImg;
	AllocatedImage _blackImg;
	AllocatedImage _greyImg;
	AllocatedImage _errorCheckerBoardImg;

	VkSampler _defaultSamplerLinear;
	VkSampler _defaultSamplerNearest;

	// default Material data
	MaterialInstance _defaultData;
	GLTFMetallic_Roughness _metalRoughMaterial;

	//draw context
	DrawContext mainDrawContext;
	std::unordered_map<std::string, std::shared_ptr<Node>> loadedNodes;
	std::unordered_map < std::string, std::shared_ptr<LoadedGLTF>> loadedScenes;

	//immediate submits, not synced with the renderr loop
	void ImmediateSubmit(std::function<void(VkCommandBuffer cmd)> && function);

	Material* createMaterial(VkPipeline pipeline, VkPipelineLayout layout, const std::string& name);
	Mesh* getMesh(const std::string& name);
	Material* getMaterial(const std::string name);
	void drawObjects(VkCommandBuffer cmd, RenderObject* first, int count);
	size_t padUniformBufferSize(size_t originalSize);
	AllocatedImage createImage(VkExtent3D size, VkFormat format, VkImageUsageFlags usage, bool mipmapped = false);
	AllocatedImage createImage(void* data, VkExtent3D size, VkFormat format, VkImageUsageFlags usage, bool mipmapped = false);
	void destroyImage(const AllocatedImage& img);

	bool _isInitialized{ false };
	int _frameNumber {0};
	bool stop_rendering{ false };
	VkExtent2D _windowExtent{1700, 900};

	struct SDL_Window* _window{ nullptr };

	static VulkanEngine& Get();

	//initializes everything in the engine
	void init();

	//shuts down the engine
	void cleanup();

	//draw loop
	void draw();

	//run main loop
	void run();

	//descriptors
	void initDescriptors();

	//pipeline
	void initPipelines();

	//meshes
	void loadMeshes();

	//meshes 2.0
	GPUMeshBuffers uploadMesh(std::span<uint32_t> indices, std::span<Vertex> vertices);

	void uploadMesh(Mesh& mesh);

	//texture
	void loadImages();

	//scene
	void initScene();

	void updateScene();

	//buffer
	AllocatedBuffer createBuffer(size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage);

	void InitDefaults();


private:

	//initializes vulkan device
	void initVulkan();

	//initializes the swapchain
	void initSwapchain();

	//initializes the command pool
	void initCommands();

	//initializes a renderpass
	void initDefaultRenderpass();

	//initializes the framebuffers
	void initFrameBuffers();

	//initializes the semaphores and fences
	void initSyncStructures();
};
