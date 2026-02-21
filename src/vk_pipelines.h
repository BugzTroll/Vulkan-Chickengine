#pragma once 
#include <vk_types.h>

class PipelineBuilder
{
public:
	std::vector<VkPipelineShaderStageCreateInfo> _shaderStages;
	std::optional<VkPipelineVertexInputStateCreateInfo> _vertexInputInfo;
	std::optional<VkPipelineInputAssemblyStateCreateInfo> _inputAssembly;
	VkViewport _viewport;
	VkRect2D _scissor;
	VkPipelineRasterizationStateCreateInfo _rasterizer;
	VkPipelineColorBlendAttachmentState _colorBlendAttachment;
	VkPipelineMultisampleStateCreateInfo _multisampling;
	VkPipelineLayout _pipelineLayout;
	/*VkPipelineRenderingCreateInfo _renderInfo;*/ //to get rid of renderpass
	VkPipelineDepthStencilStateCreateInfo _depthStencil;
	VkFormat _colorAttachmentFormat;

	PipelineBuilder()
	{
		clear();
	}

	void clear();

	void setShaders(VkShaderModule vertexShader, const char* vsEntry, VkShaderModule fragmentShader, const char* fsEntry);
	void setMeshShaders(VkShaderModule meshShader, const char* meshEntry, VkShaderModule fragmentShader, const char* fsEntry);
	void setInputTopology(VkPrimitiveTopology topology);
	void setPolygonMode(VkPolygonMode polygonMode);
	void setCullMode(VkCullModeFlags cullmode, VkFrontFace frontFace);
	void setMultisamplingNone();
	void disableBlending();
	void enableBlendingAdditive();
	void setColorAttachmentFormat(VkFormat format);
	void setDepthFormat(VkFormat format);
	void setDepthTest(bool bDepthTest, bool bDepthWrite, VkCompareOp compareOp);


	VkPipeline buildPipeline(VkDevice device, VkRenderPass pass);
};