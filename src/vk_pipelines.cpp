#include <vk_pipelines.h>
#include <iostream>

#include "vk_initializers.h"

void PipelineBuilder::clear()
{
    // clear all of the structs we need back to 0 with their correct stype

    if (_inputAssembly.has_value())
    {
        _inputAssembly = { .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
    }

    _rasterizer = { .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };

    _colorBlendAttachment = {};

    _multisampling = { .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };

    _pipelineLayout = {};

    _depthStencil = { .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };

    //_renderInfo = { .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };

    _shaderStages.clear();
}

void PipelineBuilder::setShaders(VkShaderModule vertexShader, const char* vsEntry, VkShaderModule fragmentShader, const char* fsEntry)
{
    _shaderStages.clear();
    _shaderStages.push_back(vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_VERTEX_BIT, vertexShader, vsEntry));
    _shaderStages.push_back(vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_FRAGMENT_BIT, fragmentShader, fsEntry));
}

void PipelineBuilder::setMeshShaders(VkShaderModule meshShader, const char* meshEntry, VkShaderModule fragmentShader, const char* fsEntry)
{
    _shaderStages.clear();
    _shaderStages.push_back(vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_MESH_BIT_EXT, meshShader, meshEntry));
    _shaderStages.push_back(vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_FRAGMENT_BIT, fragmentShader, fsEntry));
}

void PipelineBuilder::setInputTopology(VkPrimitiveTopology topology)
{
    if (!_inputAssembly.has_value())
    {
       _inputAssembly = { .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
    }

    _inputAssembly.value().topology = topology;
    _inputAssembly.value().primitiveRestartEnable = false;
}

void PipelineBuilder::setPolygonMode(VkPolygonMode polygonMode)
{
    _rasterizer.polygonMode = polygonMode;
    _rasterizer.lineWidth = 1.0f;
}

void PipelineBuilder::setCullMode(VkCullModeFlags cullmode, VkFrontFace frontFace)
{
    _rasterizer.cullMode = cullmode;
    _rasterizer.frontFace = frontFace;
}

void PipelineBuilder::setMultisamplingNone()
{
    _multisampling.pNext = nullptr;
    _multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    //1 sample per pixel
    _multisampling.sampleShadingEnable = VK_FALSE;
    _multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    _multisampling.minSampleShading = 1.0f;
    _multisampling.pSampleMask = nullptr;
    //no alphatocoverage
    _multisampling.alphaToCoverageEnable = VK_FALSE;
    _multisampling.alphaToOneEnable = VK_FALSE;
}

void PipelineBuilder::disableBlending()
{
    _colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    _colorBlendAttachment.blendEnable = VK_FALSE;
}

void PipelineBuilder::enableBlendingAdditive()
{
    _colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    _colorBlendAttachment.blendEnable = VK_TRUE;
    _colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    _colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
    _colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
    _colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    _colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    _colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;
}

void PipelineBuilder::setColorAttachmentFormat(VkFormat format)
{
    _colorAttachmentFormat = format;

   // _renderInfo.colorAttachmentCount = 1;
    //_renderInfo.pColorAttachmentFormats = &_colorAttachmentFormat;
}

void PipelineBuilder::setDepthFormat(VkFormat format)
{
    //_renderInfo.depthAttachmentFormat = format;
}

void PipelineBuilder::setDepthTest(bool bDepthTest, bool bDepthWrite, VkCompareOp compareOp)
{
    _depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    _depthStencil.pNext = nullptr;
    _depthStencil.depthCompareOp = bDepthTest ? compareOp : VK_COMPARE_OP_ALWAYS;
    _depthStencil.depthTestEnable = bDepthTest ? VK_TRUE : VK_FALSE;
    _depthStencil.depthWriteEnable = bDepthWrite ? VK_TRUE : VK_FALSE;
    _depthStencil.depthBoundsTestEnable = VK_FALSE;
    _depthStencil.minDepthBounds = 0.f;
    _depthStencil.maxDepthBounds = 1.f;
    _depthStencil.stencilTestEnable = VK_FALSE;
}

VkPipeline PipelineBuilder::buildPipeline(VkDevice device, VkRenderPass pass)
{
    VkPipelineViewportStateCreateInfo viewportState = {};
    viewportState.pNext = nullptr;//&_renderInfo;
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports = &_viewport;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &_scissor;

    VkPipelineColorBlendStateCreateInfo colorBlending = {};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.pNext = nullptr;
    colorBlending.logicOp = VK_LOGIC_OP_COPY;
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &_colorBlendAttachment;

    VkGraphicsPipelineCreateInfo pipelineInfo = {};
    pipelineInfo.pNext = nullptr; //&_renderInfo TODO what is this
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.stageCount = (uint32_t)_shaderStages.size();
    pipelineInfo.pStages = _shaderStages.data();
    pipelineInfo.pVertexInputState = _vertexInputInfo.has_value() ? &_vertexInputInfo.value() : VK_NULL_HANDLE;
    pipelineInfo.pInputAssemblyState = _inputAssembly.has_value() ? &_inputAssembly.value() : VK_NULL_HANDLE;
    pipelineInfo.pMultisampleState = &_multisampling;
    pipelineInfo.pRasterizationState = &_rasterizer;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.layout = _pipelineLayout;
    pipelineInfo.renderPass = pass;
    pipelineInfo.subpass = 0;
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
    pipelineInfo.pDepthStencilState = &_depthStencil;


    VkPipeline newPipeline;

    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &newPipeline) != VK_SUCCESS)
    {
        std::cout << "failed to create graphics pipeline\n";
        return VK_NULL_HANDLE;
    }
    else
    {
        std::cout << "Successfully created graphics pipeline\n";
        return newPipeline;
    }
}