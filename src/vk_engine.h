// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <vk_types.h>
#include "vk_mesh.h"
#include "glm/glm.hpp"

constexpr unsigned int FRAME_OVERLAP = 2;

struct GPUObjectData
{
	glm::mat4 modelMatrix;
};

struct GPUSceneData
{
	glm::vec4 fogColor;
	glm::vec4 fogDistances;
	glm::vec4 ambiantColor;
	glm::vec4 sunlightDirection;
	glm::vec4 sunlightColor;
};

struct GPUCameraData 
{
	glm::mat4 view;
	glm::mat4 proj;
	glm::mat4 viewProj;
};

struct FrameData
{
	//sync structures
	VkSemaphore _presentSemaphore, _renderSemaphore;
	VkFence _renderFence;

	//cmd buffer
	VkCommandPool _commandPool;
	VkCommandBuffer _mainCommandBuffer;

	//camera UB
	AllocatedBuffer _cameraBuffer;
	VkDescriptorSet _globalDescriptor;

	//objects transform SB
	AllocatedBuffer _objectBuffer;
	VkDescriptorSet _objectDescriptor;
};

struct Material
{
	VkPipeline pipeline;
	VkPipelineLayout pipelineLayout;
};

struct RenderObject
{
	Mesh* mesh;
	Material* material;
	glm::mat4 transformMatrix;
};

struct MeshPushConstants
{
	glm::vec4 data;
	glm::mat4 renderMatrix;
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

class PipelineBuilder 
{
public:
	std::vector<VkPipelineShaderStageCreateInfo> _shaderStages;
	VkPipelineVertexInputStateCreateInfo _vertexInputInfo;
	VkPipelineInputAssemblyStateCreateInfo _inputAssembly;
	VkViewport _viewport;
	VkRect2D _scissor;
	VkPipelineRasterizationStateCreateInfo _rasterizer;
	VkPipelineColorBlendAttachmentState _colorBlendAttachment;
	VkPipelineMultisampleStateCreateInfo _multisampling;
	VkPipelineLayout _pipelineLayout;
	VkPipelineDepthStencilStateCreateInfo _depthStencil;

	VkPipeline buildPipeline(VkDevice device, VkRenderPass pass);
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
	VkDescriptorPool _descriptorPool;

	//Scene
	GPUSceneData _sceneParameters;
	AllocatedBuffer _sceneParameterBuffer;

	std::unordered_map<std::string, Material> _materials;
	std::unordered_map<std::string, Mesh> _meshes;

	Material* createMaterial(VkPipeline pipeline, VkPipelineLayout layout, const std::string& name);
	Mesh* getMesh(const std::string& name);
	Material* getMaterial(const std::string name);
	void drawObjects(VkCommandBuffer cmd, RenderObject* first, int count);
	size_t padUniformBufferSize(size_t originalSize);

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

	//shaders
	bool loadShaderModule(const char* filePath, VkShaderModule* outShaderModule);

	//descriptors
	void initDescriptors();

	//pipeline
	void initPipelines();

	//meshes
	void loadMeshes();

	//scene
	void initScene();

	void uploadMesh(Mesh& mesh);

	AllocatedBuffer createBuffer(size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage);


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
