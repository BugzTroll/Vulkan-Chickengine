//> includes
#include "vk_engine.h"

#include <SDL.h>
#include <SDL_vulkan.h>

#include <vk_initializers.h>
#include <vk_types.h>

//bootstrap libraries
#include "VkBootstrap.h"

#include <chrono>
#include <thread>

#include <iostream>
#include <fstream>

#include <set>

#include "glm/glm.hpp"
#include "glm/gtx/transform.hpp"

#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"


#include "vk_texture.h"


using namespace std;

const uint64_t timeout = 1000000000;

VulkanEngine* loadedEngine = nullptr;

FrameData& VulkanEngine::getCurrentFrame()
{
    return _frames[_frameNumber % FRAME_OVERLAP];
}

void VulkanEngine::ImmediateSubmit(std::function<void(VkCommandBuffer cmd)>&& function)
{
    VkCommandBuffer cmd = _uploadContext.commandBuffer;

    VkCommandBufferBeginInfo cmdBeginInfo = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
    VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));

    //execute the function
    function(cmd);

    VK_CHECK(vkEndCommandBuffer(cmd));

    //submit to the gfx queue
    VkSubmitInfo submit = vkinit::submit_info(&cmd);
    VK_CHECK(vkQueueSubmit(_graphicsQueue, 1, &submit, _uploadContext._uploadFence));

    //wait
    vkWaitForFences(_device, 1, &_uploadContext._uploadFence, true, 9999999999);
    vkResetFences(_device, 1, &_uploadContext._uploadFence);

    //reset the command buffers in the command pool
    vkResetCommandPool(_device, _uploadContext._commandPool, 0);
}

Material* VulkanEngine::createMaterial(VkPipeline pipeline, VkPipelineLayout layout, const std::string& name)
{
    Material mat;
    mat.pipeline = pipeline;
    mat.pipelineLayout = layout;
    _materials[name] = mat;
    return &_materials[name];
}

Mesh* VulkanEngine::getMesh(const std::string& name)
{
    auto it = _meshes.find(name);
    if (it != _meshes.end())
    {
        return &it->second;
    }
    else
    {
        return nullptr;
    }
}

Material* VulkanEngine::getMaterial(const std::string name)
{
    auto it = _materials.find(name);
    if (it != _materials.end())
    {
        return &it->second;
    }
    else
    {
        return nullptr;
    }
}

void VulkanEngine::drawObjects(VkCommandBuffer cmd, RenderObject* first, int count)
{
    //cam
    //glm::vec3 camPos = { 0.f, -6.f, -10.f };
    //glm::mat4 view = glm::translate(glm::mat4{ 1.f }, camPos);

    ////proj
    //glm::mat4 projection = glm::perspective(glm::radians(70.f), 1700.f / 900.f, 0.1f, 200.f);
    //projection[1][1] *= -1;

    GPUCameraData camData;
    camData.view = _camera.view;
    camData.proj = _camera.projection;
    camData.viewProj = _camera.viewProjection;

    //scene data
    float framed = _frameNumber / 120.f;
    _sceneParameters.ambiantColor = { sin(framed), 0, cos(framed), 1 };
    _sceneParameters.cameraData = camData;

    int frameIndex = _frameNumber % FRAME_OVERLAP;
    
    //copy scene data to GPU
    char* sceneData;
    vmaMapMemory(_allocator, _sceneParameterBuffer._allocation, (void**)&sceneData);
    sceneData += padUniformBufferSize(sizeof(GPUSceneData)) * frameIndex;
    memcpy(sceneData, &_sceneParameters, sizeof(GPUSceneData)); //We copy the data to the GPU
    vmaUnmapMemory(_allocator, _sceneParameterBuffer._allocation);

    //Object data
    void* objData;
    vmaMapMemory(_allocator, getCurrentFrame()._objectBuffer._allocation, &objData);
    GPUObjectData* objectSSBO = (GPUObjectData*)objData;

    Mesh* lastMesh = nullptr;
    Material* lastMaterial = nullptr;

    for (int i = 0; i < count; i++)
    {
        RenderObject& object = first[i];
        objectSSBO[i].modelMatrix = object.transformMatrix;

        //bind if pipeline is diffent
        if (object.material != lastMaterial)
        {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, object.material->pipeline);
            
            //dynamic descriptor are best suited for data that updates per mesh for example, normal descriptor is best for per frame update
            // game engines often use dynamic uniform ONLY, because they can allocate and write a buffer at runtime while rendering;
            uint32_t uniform_offset = static_cast<uint32_t>(padUniformBufferSize(sizeof(GPUSceneData)) * frameIndex); //only send offset for corresponding dynamic bindings! ( we dont need one for static UB )

            if (object.material != getMaterial("triangle") && object.material != getMaterial("coloredTriangle"))
            {
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, object.material->pipelineLayout, 0, 1, &getCurrentFrame()._globalDescriptor, 1, &uniform_offset);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, object.material->pipelineLayout, 1, 1, &getCurrentFrame()._objectDescriptor, 0, nullptr);
            }
            if (object.material->textureSet != VK_NULL_HANDLE)
            {
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, object.material->pipelineLayout, 2, 1, &object.material->textureSet, 0, nullptr);
            }

            lastMaterial = object.material;
        }

        if (object.material != getMaterial("triangle") && object.material != getMaterial("coloredTriangle"))
        {
            glm::mat4 mesh_matrix = object.transformMatrix;

            MeshPushConstants constants;
            constants.renderMatrix = mesh_matrix;

            // send the transform as push constants
            vkCmdPushConstants(cmd, object.material->pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(MeshPushConstants), &constants);

            VkDeviceSize offset = 0;
            if (object.mesh != lastMesh)
            {
                vkCmdBindVertexBuffers(cmd, 0, 1, &object.mesh->_vertexBuffer._buffer, &offset);
                lastMesh = object.mesh;
            }
        }

        vkCmdDraw(cmd, static_cast<uint32_t>(object.mesh->_vertices.size()), 1, 0, i);
    }

    vmaUnmapMemory(_allocator, getCurrentFrame()._objectBuffer._allocation);
}

size_t VulkanEngine::padUniformBufferSize(size_t originalSize)
{
    size_t minUboAlignment = _gpuProperties.limits.minUniformBufferOffsetAlignment;
    size_t alignedSize = originalSize;
    if (minUboAlignment > 0)
    {
        //alignedSize = (alignedSize + minUboAlignment - 1) & ~(minUboAlignment - 1); //sasha willems https://github.com/SaschaWillems/Vulkan/tree/master/examples/dynamicuniformbuffer
        int division = static_cast<int>(std::ceil(float(originalSize) / float(minUboAlignment)));
        alignedSize = minUboAlignment * division;

    }
    return alignedSize;
}

VulkanEngine& VulkanEngine::Get() { return *loadedEngine; }

void VulkanEngine::init()
{
    // only one engine initialization is allowed with the application.
    assert(loadedEngine == nullptr);
    loadedEngine = this;

    // We initialize SDL and create a window with it.
    SDL_Init(SDL_INIT_VIDEO);

    SDL_WindowFlags window_flags = (SDL_WindowFlags)(SDL_WINDOW_VULKAN);

    _window = SDL_CreateWindow(
        "Vulkan Engine",
        SDL_WINDOWPOS_UNDEFINED,
        SDL_WINDOWPOS_UNDEFINED,
        _windowExtent.width,
        _windowExtent.height,
        window_flags);

    // load the core vulkan structures
    initVulkan();

    // create the swapchain
    initSwapchain();

    //create the commands buffer
    initCommands();

    initDefaultRenderpass();

    initFrameBuffers();

    initSyncStructures();

    initDescriptors();

    initPipelines();

    // load the scene data
    loadMeshes();

    loadImages();

    initScene();

    // everything went fine
    _isInitialized = true;
}

void VulkanEngine::cleanup()
{
    if (_isInitialized) {
        FrameData currentFrame = getCurrentFrame();
        vkWaitForFences(_device, 1, &currentFrame._renderFence, true, 100000000);
        _mainDeletionQueue.flush();

        vkDestroyDevice(_device, nullptr);
        vkDestroySurfaceKHR(_instance, _surface, nullptr);
        vkb::destroy_debug_utils_messenger(_instance, _debug_messenger);
        vkDestroyInstance(_instance, nullptr);

        SDL_DestroyWindow(_window);
    }

    // clear engine pointer
    loadedEngine = nullptr;
}

void VulkanEngine::draw()
{
    FrameData currentFrame = getCurrentFrame();
    VK_CHECK(vkWaitForFences(_device, 1, &currentFrame._renderFence, true, timeout));
    VK_CHECK(vkResetFences(_device, 1, &currentFrame._renderFence));

    uint32_t swapChainImageIndex;
    VK_CHECK(vkAcquireNextImageKHR(_device, _swapchain, timeout, currentFrame._presentSemaphore, nullptr, &swapChainImageIndex)); // blocks the cpu for 1s if no image available

    //commands are done executing, we reset the command buffer
    VK_CHECK(vkResetCommandBuffer(currentFrame._mainCommandBuffer, 0));

    // We begin the command buffer
    VkCommandBufferBeginInfo cmdBeginInfo {};
    cmdBeginInfo.pNext = nullptr;
    cmdBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO; //command buffer will be submitted once, allows driver optims
    cmdBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    cmdBeginInfo.pInheritanceInfo = nullptr; //use for secondary command buffers

    VK_CHECK(vkBeginCommandBuffer(currentFrame._mainCommandBuffer, &cmdBeginInfo)); //Begin cmd buffer

    //we clear the screen
    VkClearValue clearValue;
    float flash = abs(sin(_frameNumber / 120.0f));
    clearValue.color = { {0.0f, 0.0f, flash, 0.0f} };

    //depth clear
    VkClearValue depthClear;
    depthClear.depthStencil.depth = 1.f;

    VkRenderPassBeginInfo rpInfo = {};
    rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpInfo.pNext = nullptr;

    rpInfo.renderPass = _renderPass;
    rpInfo.renderArea.offset.x = 0;
    rpInfo.renderArea.offset.y = 0;
    rpInfo.renderArea.extent.height = _windowExtent.height;
    rpInfo.renderArea.extent.width = _windowExtent.width;
    rpInfo.framebuffer = _framebuffers[swapChainImageIndex];

    VkClearValue clearValues[2] = { clearValue , depthClear };

    rpInfo.clearValueCount = 2;
    rpInfo.pClearValues = &clearValues[0];

    //Drawing
    vkCmdBeginRenderPass(currentFrame._mainCommandBuffer, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);  //Begin render pass

    drawObjects(currentFrame._mainCommandBuffer, &_renderables[0], static_cast<int>(_renderables.size()));

    vkCmdEndRenderPass(currentFrame._mainCommandBuffer);

    VK_CHECK(vkEndCommandBuffer(currentFrame._mainCommandBuffer)); //end command buffer

    //prep submission queue

    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.pNext = nullptr;

    VkPipelineStageFlags waitState = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    submit.pWaitDstStageMask = &waitState;

    submit.waitSemaphoreCount = 1;
    submit.pWaitSemaphores = &currentFrame._presentSemaphore; // We wait for the present semaphore, will be set when the acquireImgKHR is done

    submit.signalSemaphoreCount = 1;
    submit.pSignalSemaphores = &currentFrame._renderSemaphore;// We trigger the render semaphore when we are done rendering

    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &currentFrame._mainCommandBuffer;

    VK_CHECK(vkQueueSubmit(_graphicsQueue, 1, &submit, currentFrame._renderFence));

    VkPresentInfoKHR presentInfo = {};
    presentInfo.pNext = nullptr;
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = &currentFrame._renderSemaphore;
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &_swapchain;
    presentInfo.pImageIndices = &swapChainImageIndex;

    VK_CHECK(vkQueuePresentKHR(_graphicsQueue, &presentInfo)); //Display image to the screen, wait on the render semaphore ( display only when image is done ) 

    _frameNumber++;
}

void VulkanEngine::run()
{
    SDL_Event e;
    bool bQuit = false;
    set<int> keysPressed;

    float cameraSpeedMultiplier = 2.0f;

    // main loop
    while (!bQuit) {
        // Handle events on queue
        while (SDL_PollEvent(&e) != 0) 
        {
            // close the window when user alt-f4s or clicks the X button
            if (e.type == SDL_QUIT)
                bQuit = true;

            if (e.type == SDL_WINDOWEVENT) {
                if (e.window.event == SDL_WINDOWEVENT_MINIMIZED) {
                    stop_rendering = true;
                }
                if (e.window.event == SDL_WINDOWEVENT_RESTORED) {
                    stop_rendering = false;
                }
            }

            //KEY UP
            if (e.type == SDL_KEYUP) 
            {
                keysPressed.erase(e.key.keysym.sym);

                if (e.key.keysym.sym == SDLK_LSHIFT)
                {
                    _camera.movementSpeed /= cameraSpeedMultiplier;
                }
            }

            //KEY DOWN
            if(e.type == SDL_KEYDOWN)
            {
                if (e.key.keysym.sym == SDLK_LSHIFT && !keysPressed.contains(SDLK_LSHIFT))
                {
                    _camera.movementSpeed *= cameraSpeedMultiplier;
                }
                keysPressed.insert(e.key.keysym.sym);

                const char* keyName = SDL_GetKeyName(e.key.keysym.sym);
                std::cout << keyName << std::endl;
            }
        }

        //Send events based on key state
        if (keysPressed.contains(SDLK_w))
        {
            //move camera forward
            _camera.MoveCamera(_camera.cameraForward);
        }
        if (keysPressed.contains(SDLK_s))
        {
            //move camera back
            _camera.MoveCamera(_camera.cameraForward * -1.f);
        }
        if (keysPressed.contains(SDLK_a))
        {
            //move camera left
            _camera.MoveCamera(_camera.cameraRight * -1.f);
        }
        if (keysPressed.contains(SDLK_d))
        {
            //move camera right
            _camera.MoveCamera(_camera.cameraRight);
        }

        if (keysPressed.contains(SDLK_UP))
        {
            _camera.RotateCamera(_camera.radAngleIncrement, 0);
        }
        if (keysPressed.contains(SDLK_DOWN))
        {
            _camera.RotateCamera(-_camera.radAngleIncrement, 0);
        }
        if (keysPressed.contains(SDLK_LEFT))
        {
            _camera.RotateCamera(0, _camera.radAngleIncrement);
        }
        if (keysPressed.contains(SDLK_RIGHT))
        {
            _camera.RotateCamera(0, -_camera.radAngleIncrement);
        }

        // do not draw if we are minimized
        if (stop_rendering) {
            // throttle the speed to avoid the endless spinning
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        draw();
    }
}

bool VulkanEngine::loadShaderModule(const char* filePath, VkShaderModule* outShaderModule)
{
    // open and load the file in uint32 type ( expected by spirv )
    std::ifstream file(filePath, std::ios::ate | std::ios::binary);

    size_t fileSize = (size_t)file.tellg();
    std::vector<uint32_t> buffer(fileSize / sizeof(uint32_t));

    file.seekg(0);
    file.read((char*)buffer.data(), fileSize);
    file.close();

    //create the shader module
    VkShaderModuleCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.pNext = nullptr;

    createInfo.codeSize = buffer.size() * sizeof(uint32_t);
    createInfo.pCode = buffer.data();

    VkShaderModule shaderModule;
    if (vkCreateShaderModule(_device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) //common to fail because of shader issues, so we use somthing more robust
    {
        return false;

    }

    *outShaderModule = shaderModule;
    return true;
}

void VulkanEngine::initDescriptors()
{
    //descriptor layout
    VkDescriptorSetLayoutBinding sceneBufferBinding = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0);
    VkDescriptorSetLayoutBinding objectBufferBinding = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 0); // new layout
    VkDescriptorSetLayoutBinding singleTextureBufferBinding = vkinit::descriptorset_layout_binding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0); // new layout

    //first set is camera and scene
    VkDescriptorSetLayoutCreateInfo setInfo = {};
    setInfo.pNext = nullptr;
    setInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    setInfo.bindingCount = 1;
    setInfo.pBindings = &sceneBufferBinding;
    setInfo.flags = 0;

    vkCreateDescriptorSetLayout(_device, &setInfo, nullptr, &_globalSetLayout);

    //second set is objects data
    VkDescriptorSetLayoutCreateInfo objectSetInfo = {};
    objectSetInfo.pNext = nullptr;
    objectSetInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    objectSetInfo.bindingCount = 1;
    objectSetInfo.pBindings = &objectBufferBinding;
    objectSetInfo.flags = 0;

    vkCreateDescriptorSetLayout(_device, &objectSetInfo, nullptr, &_objectSetLayout);

    //third if for textures
    VkDescriptorSetLayoutCreateInfo textureSetInfo = {};
    textureSetInfo.pNext = nullptr;
    textureSetInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    textureSetInfo.bindingCount = 1;
    textureSetInfo.pBindings = &singleTextureBufferBinding;
    textureSetInfo.flags = 0;

    vkCreateDescriptorSetLayout(_device, &textureSetInfo, nullptr, &_singleTextureSetLayout);

    //descriptor pool of 10 uniform buffer
    std::vector<VkDescriptorPoolSize> sizes =
    {
         {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 10}, //dynamic uniform
         {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 10}, //storage buffer
         {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 10 } //combined image sampler
    };

    //descriptor pool
    VkDescriptorPoolCreateInfo poolInfo = {};
    poolInfo.pNext = nullptr;
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = (uint32_t)sizes.size();
    poolInfo.pPoolSizes = sizes.data();
    poolInfo.maxSets = 10;
    poolInfo.flags = 0;
    
    vkCreateDescriptorPool(_device, &poolInfo, nullptr, &_descriptorPool);

    //Scene parameters buffer
    const size_t sceneParamBufferSize = FRAME_OVERLAP * padUniformBufferSize(sizeof(GPUSceneData));
    _sceneParameterBuffer = createBuffer(sceneParamBufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

    //Camera buffers and descriptors
    for (int i = 0; i < FRAME_OVERLAP; i++)
    {
        const int MAX_OBJECTS = 10000;
        _frames[i]._objectBuffer = createBuffer(sizeof(GPUObjectData) * MAX_OBJECTS, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

        //allocate descriptor set for camera and scene
        VkDescriptorSetAllocateInfo allocInfo = {};
        allocInfo.descriptorPool = _descriptorPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pNext = nullptr;
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.pSetLayouts = &_globalSetLayout;

        vkAllocateDescriptorSets(_device, &allocInfo, &_frames[i]._globalDescriptor);

        //allocate descriptor set for objects
        VkDescriptorSetAllocateInfo objallocInfo = {};
        objallocInfo.descriptorPool = _descriptorPool;
        objallocInfo.descriptorSetCount = 1;
        objallocInfo.pNext = nullptr;
        objallocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        objallocInfo.pSetLayouts = &_objectSetLayout;

        vkAllocateDescriptorSets(_device, &objallocInfo, &_frames[i]._objectDescriptor);

        //scene descriptor info
        VkDescriptorBufferInfo sceneDescriptorInfo;
        sceneDescriptorInfo.offset = 0;// this is needed if uniform are not dynamic : padUniformBufferSize(sizeof(GPUSceneData) * i );
        sceneDescriptorInfo.range = sizeof(GPUSceneData);
        sceneDescriptorInfo.buffer = _sceneParameterBuffer._buffer; //Dynamic buffer ( with data for each frame in separate locations )

       //object buffer info
        VkDescriptorBufferInfo objectDescriptorInfo;
        objectDescriptorInfo.offset = 0;
        objectDescriptorInfo.range = sizeof(GPUObjectData) * MAX_OBJECTS;
        objectDescriptorInfo.buffer = _frames[i]._objectBuffer._buffer;

        VkWriteDescriptorSet sceneWrite = vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, _frames[i]._globalDescriptor, &sceneDescriptorInfo, 0);
        VkWriteDescriptorSet objectWrite = vkinit::write_descriptor_buffer(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, _frames[i]._objectDescriptor, &objectDescriptorInfo, 0);

        VkWriteDescriptorSet setWrites[] = {sceneWrite, objectWrite };

        vkUpdateDescriptorSets(_device, 2, setWrites, 0, nullptr); //one call to update 2 dscriptor set is valid!
        
        _mainDeletionQueue.pushFunction([&]
            {
                vmaDestroyBuffer(_allocator, _frames[i]._objectBuffer._buffer, _frames[i]._objectBuffer._allocation);
            });
    }

    //cleanup
    _mainDeletionQueue.pushFunction([&]
        {
            vkDestroyDescriptorSetLayout(_device, _globalSetLayout, nullptr);
            vkDestroyDescriptorSetLayout(_device, _objectSetLayout, nullptr);
            vkDestroyDescriptorPool(_device, _descriptorPool, nullptr);
            vmaDestroyBuffer(_allocator, _sceneParameterBuffer._buffer, _sceneParameterBuffer._allocation);
        });
}

void VulkanEngine::initPipelines()
{
    VkShaderModule coloredTriangleShaders;
    if (!loadShaderModule("../../shaders/coloredTriangle.spv", &coloredTriangleShaders))
    {
        std::cout << "Error when building the color triangle shaders module" << std::endl;
    }
    else
    {
        std::cout << "color triangle shaders successfully loaded" << std::endl;
    }

    VkShaderModule simpleTriangleShaders;
    if (!loadShaderModule("../../shaders/triangle.spv", &simpleTriangleShaders))
    {
        std::cout << "Error when building the simple triangle shader module";
    }
    else
    {
        std::cout << "simple triangle shader module succesfully loaded" << std::endl;
    }

    VkShaderModule baseMeshShaders;
    if (!loadShaderModule("../../shaders/baseMesh.spv", &baseMeshShaders))
    {
        std::cout << "Error when building the base mesh shaders";
    }
    else
    {
        std::cout << "base mesh shaders module succesfully loaded" << std::endl;
    }

    VkShaderModule texturelitFragShader;
    if (!loadShaderModule("../../shaders/textureLit.spv", &texturelitFragShader))
    {
        std::cout << "Error when building the texture lit frag shader module" << std::endl;
    }
    else
    {
        std::cout << "Texture lit frag shader module succesfully loaded" << std::endl;
    }

    VkDescriptorSetLayout setLayout[] = { _globalSetLayout, _objectSetLayout };
    VkDescriptorSetLayout textureSetLayout[] = { _globalSetLayout, _objectSetLayout, _singleTextureSetLayout };
    VkPipelineLayout meshPipelineLayout, trianglePipelineLayout, texturedPipelineLayout;

    //simple tri pipeline layout
    VkPipelineLayoutCreateInfo pipelineLayoutInfo = vkinit::pipeline_layout_create_info();
    VK_CHECK(vkCreatePipelineLayout(_device, &pipelineLayoutInfo, nullptr, &trianglePipelineLayout));

    //push constants
    VkPushConstantRange push_constant;
    push_constant.offset = 0;
    push_constant.size = sizeof(MeshPushConstants);
    push_constant.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    //mesh pipeline layout
    VkPipelineLayoutCreateInfo meshPipelineLayoutInfo = vkinit::pipeline_layout_create_info();

    meshPipelineLayoutInfo.pPushConstantRanges = &push_constant; //we can have different push constant for vertex and pixel shader, so it's built with ranges
    meshPipelineLayoutInfo.pushConstantRangeCount = 1;
    meshPipelineLayoutInfo.setLayoutCount = 2;
    meshPipelineLayoutInfo.pSetLayouts = setLayout;

    VK_CHECK(vkCreatePipelineLayout(_device, &meshPipelineLayoutInfo, nullptr, &meshPipelineLayout));

    //Texture pipeeline layout
    VkPipelineLayoutCreateInfo textured_pipeline_layout_info = meshPipelineLayoutInfo;
    textured_pipeline_layout_info.setLayoutCount = 3;
    textured_pipeline_layout_info.pSetLayouts = textureSetLayout;
    VK_CHECK(vkCreatePipelineLayout(_device, &textured_pipeline_layout_info, nullptr, &texturedPipelineLayout));

    PipelineBuilder pipelineBuilder;
    pipelineBuilder._shaderStages.push_back(vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_VERTEX_BIT, coloredTriangleShaders, "vsmain"));
    pipelineBuilder._shaderStages.push_back(vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_FRAGMENT_BIT, coloredTriangleShaders, "psmain"));

    pipelineBuilder._vertexInputInfo = vkinit::vertex_input_state_create_info();
    pipelineBuilder._inputAssembly = vkinit::input_assembly_create_info(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    pipelineBuilder._multisampling = vkinit::multisampling_state_create_info();
    pipelineBuilder._rasterizer = vkinit::rasterizaion_state_create_info(VK_POLYGON_MODE_FILL);
    pipelineBuilder._colorBlendAttachment = vkinit::color_blend_attachment_state();
    pipelineBuilder._depthStencil = vkinit::depth_stencil_create_info(true, true, VK_COMPARE_OP_LESS_OR_EQUAL);
    pipelineBuilder._pipelineLayout = trianglePipelineLayout;

    pipelineBuilder._viewport.x = 0;
    pipelineBuilder._viewport.y = 0;
    pipelineBuilder._viewport.width = (float)_windowExtent.width;
    pipelineBuilder._viewport.height = (float)_windowExtent.height;
    pipelineBuilder._viewport.minDepth = 0.0f;
    pipelineBuilder._viewport.maxDepth = 1.0f;

    pipelineBuilder._scissor.offset = { 0, 0 };
    pipelineBuilder._scissor.extent = _windowExtent;

    VkPipeline coloredTrianglePipeline = pipelineBuilder.buildPipeline(_device, _renderPass);

    // init second pipeline
    pipelineBuilder._shaderStages.clear();
    pipelineBuilder._shaderStages.push_back(vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_VERTEX_BIT, simpleTriangleShaders, "vsmain"));
    pipelineBuilder._shaderStages.push_back(vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_FRAGMENT_BIT, simpleTriangleShaders, "psmain"));
    VkPipeline redTrianglePipeline = pipelineBuilder.buildPipeline(_device, _renderPass);

    //init mesh pipeline
    VertexInputDescription vertexDescription = Vertex::getVertexDescription();
    pipelineBuilder._pipelineLayout = meshPipelineLayout;
    pipelineBuilder._vertexInputInfo.pVertexAttributeDescriptions = vertexDescription.attributes.data();
    pipelineBuilder._vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertexDescription.attributes.size());
    pipelineBuilder._vertexInputInfo.pVertexBindingDescriptions = vertexDescription.bindings.data();
    pipelineBuilder._vertexInputInfo.vertexBindingDescriptionCount = static_cast<uint32_t>(vertexDescription.bindings.size());
    pipelineBuilder._shaderStages.clear();
    pipelineBuilder._shaderStages.push_back(vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_VERTEX_BIT, baseMeshShaders, "vsmain"));
    pipelineBuilder._shaderStages.push_back(vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_FRAGMENT_BIT, baseMeshShaders, "psmain"));
    VkPipeline meshPipeline = pipelineBuilder.buildPipeline(_device, _renderPass);


    pipelineBuilder._shaderStages.clear();
    pipelineBuilder._pipelineLayout = texturedPipelineLayout;
    pipelineBuilder._shaderStages.push_back(vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_VERTEX_BIT, baseMeshShaders, "vsmain"));
    pipelineBuilder._shaderStages.push_back(vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_FRAGMENT_BIT, texturelitFragShader));
    VkPipeline texturedPipeline = pipelineBuilder.buildPipeline(_device, _renderPass);

    createMaterial(meshPipeline, meshPipelineLayout, "defaultMesh"); //lit mesh
    createMaterial(redTrianglePipeline, trianglePipelineLayout, "triangle"); //Default red triangle
    createMaterial(coloredTrianglePipeline, trianglePipelineLayout, "coloredTriangle"); // multicolored triangle
    createMaterial(texturedPipeline, texturedPipelineLayout, "texturedMesh"); // textured mesh

    // Cleanup
    vkDestroyShaderModule(_device, baseMeshShaders, nullptr);
    vkDestroyShaderModule(_device, simpleTriangleShaders, nullptr);
    vkDestroyShaderModule(_device, coloredTriangleShaders, nullptr);
    vkDestroyShaderModule(_device, texturelitFragShader, nullptr);

    _mainDeletionQueue.pushFunction([=]
        {
            vkDestroyPipeline(_device, coloredTrianglePipeline, nullptr);
            vkDestroyPipeline(_device, redTrianglePipeline, nullptr);
            vkDestroyPipeline(_device, meshPipeline, nullptr);
            vkDestroyPipeline(_device, texturedPipeline, nullptr);
            vkDestroyPipelineLayout(_device, trianglePipelineLayout, nullptr);
            vkDestroyPipelineLayout(_device, meshPipelineLayout, nullptr);
            vkDestroyPipelineLayout(_device, texturedPipelineLayout, nullptr);
        });
}

void VulkanEngine::loadMeshes()
{
    Mesh triangleMesh;
    triangleMesh._vertices.resize(3);
    triangleMesh._vertices[0].position = { 1.f, 1.f, 0.f };
    triangleMesh._vertices[1].position = { -1.f, 1.f, 0.f };
    triangleMesh._vertices[2].position = { 0.f, -1.f, 0.f };

    //green
    triangleMesh._vertices[0].color = { 0.f, 1.f, 0.f };
    triangleMesh._vertices[1].color = { 0.f, 1.f, 0.f };
    triangleMesh._vertices[2].color = { 0.f, 1.f, 0.f };

    Mesh monkeyMesh;
    monkeyMesh.loadFromObj("../../assets/monkey_smooth.obj");

    Mesh lostEmpireMesh;
    lostEmpireMesh.loadFromObj("../../assets/lost_empire.obj");

    uploadMesh(triangleMesh);
    uploadMesh(monkeyMesh);
    uploadMesh(lostEmpireMesh);

    _meshes["monkey"] = monkeyMesh;
    _meshes["triangle"] = triangleMesh; 
    _meshes["lostEmpire"] = lostEmpireMesh;
}

void VulkanEngine::initScene()
{
    //init camera here
    _camera = Camera();

    //init render objects
    RenderObject monkey;
    monkey.mesh = getMesh("monkey");
    monkey.material = getMaterial("defaultMesh");
    monkey.transformMatrix = glm::mat4(1.0f);

    _renderables.push_back(monkey);

    for (int x = -20; x <= 20; x++)
    {
        for (int y = -20; y <= 20; y++)
        {
            RenderObject tri;
            //tri.mesh = getMesh("triangle");
            tri.mesh = getMesh("triangle");
            tri.material = getMaterial("defaultMesh"); //getMaterial("triangle") //getMaterial("coloredTriangle");
            glm::mat4 translation = glm::translate(glm::mat4{ 1.f }, glm::vec3(x, 0, y));
            glm::mat4 scale = glm::scale(glm::mat4{ 1.f }, glm::vec3(0.2, 0.2, 0.2));
            tri.transformMatrix = translation * scale;

            _renderables.push_back(tri);
        }
    }

    //lost empire
    RenderObject map;
    map.mesh = getMesh("lostEmpire");
    map.material = getMaterial("texturedMesh");
    map.transformMatrix = glm::translate(glm::vec3{ 5, -10, 0 });
    _renderables.push_back(map);

    //Create sampler for the texture
    VkSamplerCreateInfo samplerInfo = vkinit::sampler_create_info(VK_FILTER_NEAREST);
    VkSampler blockySampler;

    vkCreateSampler(_device, &samplerInfo, nullptr, &blockySampler);

    //allocated texture descriptor set
    Material* texturedMat = getMaterial("texturedMesh");
    VkDescriptorSetAllocateInfo allocInfo = {};
    allocInfo.pNext = nullptr;
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = _descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &_singleTextureSetLayout;

    vkAllocateDescriptorSets(_device, &allocInfo, &texturedMat->textureSet);

    VkDescriptorImageInfo imgDscInfo = {};
    imgDscInfo.sampler = blockySampler;
    imgDscInfo.imageView = _textures["empire_diffuse"].imageView;
    imgDscInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkWriteDescriptorSet texture1 = vkinit::write_descriptor_image(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, texturedMat->textureSet, &imgDscInfo, 0);
    vkUpdateDescriptorSets(_device, 1, &texture1, 0, nullptr);
}

void VulkanEngine::uploadMesh(Mesh& mesh)
{
    //new way to upload vertex buffer using copy from CPU to GPU instead of having cpu_gpu memory
    //much faster 
    const size_t bufferSize = mesh._vertices.size() * sizeof(Vertex);

    VkBufferCreateInfo stagingBufferInfo = {};
    stagingBufferInfo.pNext = nullptr;
    stagingBufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    stagingBufferInfo.size = bufferSize;
    stagingBufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT; //only used as source to transfer commands, and not for rendering

    VmaAllocationCreateInfo vmaAllocInfo = {};
    vmaAllocInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY; //ram only

    AllocatedBuffer stagingBuffer;

    VK_CHECK(vmaCreateBuffer(_allocator, &stagingBufferInfo, &vmaAllocInfo, &stagingBuffer._buffer, &stagingBuffer._allocation, nullptr));

    //copy vertex data in CPU side staging buffer
    void* data;
    vmaMapMemory(_allocator, stagingBuffer._allocation, &data); //we need to fill the vertex data on the gpu memory from the cpu
    memcpy(data, mesh._vertices.data(), sizeof(Vertex) * mesh._vertices.size());
    vmaUnmapMemory(_allocator, stagingBuffer._allocation); //driver know we are done writing

    //now create GPU side buffer
    VkBufferCreateInfo vertexBufferInfo = {};
    vertexBufferInfo.pNext = nullptr;
    vertexBufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    vertexBufferInfo.size = bufferSize;
    vertexBufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT; // used to render meshes and transfert data to 

    vmaAllocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY; //Vram only

    VK_CHECK(vmaCreateBuffer(_allocator, &vertexBufferInfo, &vmaAllocInfo, &mesh._vertexBuffer._buffer, &mesh._vertexBuffer._allocation, nullptr));

    //Copy one buffer to the other
    ImmediateSubmit([=](VkCommandBuffer cmd)
        {
            VkBufferCopy copy;
            copy.dstOffset = 0;
            copy.srcOffset = 0;
            copy.size = bufferSize;
            vkCmdCopyBuffer(cmd, stagingBuffer._buffer, mesh._vertexBuffer._buffer, 1, &copy);
        });

    //cleanup

    _mainDeletionQueue.pushFunction([=]()
        {
            vmaDestroyBuffer(_allocator, mesh._vertexBuffer._buffer, mesh._vertexBuffer._allocation);
        });

    vmaDestroyBuffer(_allocator, stagingBuffer._buffer, stagingBuffer._allocation);

    ////old way to send vertex buffer, slower
    //VkBufferCreateInfo bufferInfo = {};
    //bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    //bufferInfo.size = mesh._vertices.size() * sizeof(Vertex);
    //bufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;

    //VmaAllocationCreateInfo vmaAllocInfo = {};
    //vmaAllocInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;

    //VK_CHECK(vmaCreateBuffer(
    //    _allocator, &bufferInfo, &vmaAllocInfo, 
    //    &mesh._vertexBuffer._buffer, &mesh._vertexBuffer._allocation, nullptr)
    //);

    ////buffer is allocated wiht vma, must be destroyed with vma
    //_mainDeletionQueue.pushFunction([=]
    //    {
    //        vmaDestroyBuffer(_allocator, mesh._vertexBuffer._buffer, mesh._vertexBuffer._allocation);
    //    });

    ////copy vertex data to the GPU
    //void* data;
    //vmaMapMemory(_allocator, mesh._vertexBuffer._allocation, &data); //we need to fill the vertex data on the gpu memory from the cpu
    //memcpy(data, mesh._vertices.data(), sizeof(Vertex) * mesh._vertices.size());
    //vmaUnmapMemory(_allocator, mesh._vertexBuffer._allocation); //driver know we are done writing
}

void VulkanEngine::loadImages()
{
    Texture lostEmpire = {};

    bool bLoadedSuccessfully = vkutil::load_image_from_file(*this, "../../assets/lost_empire-RGBA.png", lostEmpire.image);

    if (bLoadedSuccessfully)
    {
        VkImageViewCreateInfo imgViewInfo = vkinit::imageview_create_info(VK_FORMAT_R8G8B8A8_SRGB, lostEmpire.image._image, VK_IMAGE_ASPECT_COLOR_BIT);
        vkCreateImageView(_device, &imgViewInfo, nullptr, &lostEmpire.imageView);

        _textures["empire_diffuse"] = lostEmpire;
    }
}

AllocatedBuffer VulkanEngine::createBuffer(size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage)
{
    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.pNext = nullptr;
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.usage = usage;
    bufferInfo.size = allocSize;

    VmaAllocationCreateInfo vmaAllocInfo = {};
    vmaAllocInfo.usage = memoryUsage;

    AllocatedBuffer newBuffer;

    VK_CHECK(vmaCreateBuffer(_allocator, &bufferInfo, &vmaAllocInfo, &newBuffer._buffer, &newBuffer._allocation, nullptr));

    return newBuffer;
}

void VulkanEngine::initVulkan()
{
    //Instance initialization
    vkb::InstanceBuilder builder;

    // TODO enable validation layer only when -validationLayer is active
    auto inst_ret = builder.set_app_name("Vilain Vulkan Renderer")
       .request_validation_layers(true)
       .require_api_version(1, 3, 0)
       .use_default_debug_messenger()
       .build();

    vkb::Instance vkb_inst = inst_ret.value();

    _instance = vkb_inst.instance;

    _debug_messenger = vkb_inst.debug_messenger;

    //Device initialization
    SDL_Vulkan_CreateSurface(_window, _instance, &_surface);

    vkb::PhysicalDeviceSelector selector{ vkb_inst };

    // Grab watever GPU can render to window ( _surface )
    vkb::PhysicalDevice physicalDevice = selector
        .set_minimum_version(1, 1)
        .set_surface(_surface)
        .select()
        .value();

    vkb::DeviceBuilder deviceBuilder{ physicalDevice };
    VkPhysicalDeviceShaderDrawParametersFeatures shader_draw_paramters_features = {};
    shader_draw_paramters_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DRAW_PARAMETERS_FEATURES;
    shader_draw_paramters_features.pNext = nullptr;
    shader_draw_paramters_features.shaderDrawParameters = VK_TRUE;

    //Adding DRAW_PARAMETERS_FEATURES
    vkb::Device vkbDevice = deviceBuilder.add_pNext(&shader_draw_paramters_features).build().value();

    _device = vkbDevice;
    _chosenGPU = physicalDevice;

    _graphicsQueue = vkbDevice.get_queue(vkb::QueueType::graphics).value();
    _graphicsQueueFamilyIdx = vkbDevice.get_queue_index(vkb::QueueType::graphics).value();

    VmaAllocatorCreateInfo allocatorInfo = {};
    allocatorInfo.physicalDevice = physicalDevice;
    allocatorInfo.device = _device;
    allocatorInfo.instance = _instance;
    vmaCreateAllocator(&allocatorInfo, &_allocator);

    _gpuProperties = vkbDevice.physical_device.properties;

    cout << "GPU has minimum buffer alignement of " << _gpuProperties.limits.minUniformBufferOffsetAlignment << std::endl;
}

void VulkanEngine::initSwapchain()
{
    vkb::SwapchainBuilder swapchainBuilder{ _chosenGPU, _device, _surface };

    vkb::Swapchain vkbSwapchain = swapchainBuilder
        .use_default_format_selection()
        .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
        .set_desired_extent(_windowExtent.width, _windowExtent.height)
        .build()
        .value();

    _swapchain = vkbSwapchain.swapchain;
    _swapchainFormat = vkbSwapchain.image_format;
    _swapchainImages = vkbSwapchain.get_images().value();
    _swapchainImageViews = vkbSwapchain.get_image_views().value();

    _mainDeletionQueue.pushFunction([=]
        {
            vkDestroySwapchainKHR(_device, _swapchain, nullptr);
        });


    VkExtent3D depthImageExtent =
    {
        _windowExtent.width,
        _windowExtent.height,
        1
    };

    _depthFormat = VK_FORMAT_D32_SFLOAT; //32 bit float, supported by most GPU

    VkImageCreateInfo dimg_info = vkinit::image_create_info(_depthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, depthImageExtent);
    VmaAllocationCreateInfo dimg_allocInfo = {};
    dimg_allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY; //fast vram
    dimg_allocInfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT); //force to allocate to VRAM

    //Create image
    VK_CHECK(vmaCreateImage(_allocator, &dimg_info, &dimg_allocInfo, &_depthImage._image, &_depthImage._allocation, nullptr));

    VkImageViewCreateInfo dview_info = vkinit::imageview_create_info(_depthFormat, _depthImage._image, VK_IMAGE_ASPECT_DEPTH_BIT); //used for depth testing

    VK_CHECK(vkCreateImageView(_device, &dview_info, nullptr, &_depthImageView));

    _mainDeletionQueue.pushFunction([=]
        {
            vkDestroyImageView(_device, _depthImageView, nullptr);
            vmaDestroyImage(_allocator, _depthImage._image, _depthImage._allocation);
        });
}

void VulkanEngine::initCommands()
{
    //upload context command pool and command buffer
    VkCommandPoolCreateInfo uploadCommandPoolInfo = vkinit::command_pool_create_info(_graphicsQueueFamilyIdx);
    VK_CHECK(vkCreateCommandPool(_device, &uploadCommandPoolInfo, nullptr, &_uploadContext._commandPool));
    
    _mainDeletionQueue.pushFunction([=]()
        {
            vkDestroyCommandPool(_device, _uploadContext._commandPool, nullptr);
        });

    VkCommandBufferAllocateInfo uploadCommandBuffer = vkinit::command_buffer_allocate_info(_uploadContext._commandPool, 1);
    VK_CHECK(vkAllocateCommandBuffers(_device, &uploadCommandBuffer, &_uploadContext.commandBuffer));
    
    for (int i = 0; i < FRAME_OVERLAP; i++)
    {
        //create command pool
        VkCommandPoolCreateInfo commandPoolInfo = vkinit::command_pool_create_info(_graphicsQueueFamilyIdx, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);

        //create the command pool from commandPoolInfo
        VK_CHECK(vkCreateCommandPool(_device, &commandPoolInfo, nullptr, &_frames[i]._commandPool));

        
        //allocate the default command buffer
        VkCommandBufferAllocateInfo cmdAllocInfo = vkinit::command_buffer_allocate_info(_frames[i]._commandPool, 1);

        VK_CHECK(vkAllocateCommandBuffers(_device, &cmdAllocInfo, &_frames[i]._mainCommandBuffer));

        _mainDeletionQueue.pushFunction([=]
            {
                vkDestroyCommandPool(_device, _frames[i]._commandPool, nullptr);
            });
    }
}

void VulkanEngine::initDefaultRenderpass()
{
    //color attachment description
    VkAttachmentDescription color_attachment = {};
    color_attachment.format = _swapchainFormat;
    color_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
    color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    color_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    color_attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR; // After render pass ends, ready for display;
    
    //color attachment reference
    VkAttachmentReference color_attachment_ref = {};
    color_attachment_ref.attachment = 0; // index in the pAttachment array
    color_attachment_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    //depth attachment descr
    VkAttachmentDescription depth_attachment = {};
    depth_attachment.format = _depthFormat;
    depth_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
    depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depth_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depth_attachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    
    //depth attachment reference
    VkAttachmentReference depth_attachment_ref = {};
    depth_attachment_ref.attachment = 1; //color is 0
    depth_attachment_ref.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    //subpass dependencies
    VkSubpassDependency dependency = {};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    //this tells vulkan that the depth attachment cannot be used before previous renderpasses have finished using it.
    VkSubpassDependency depth_dependency = {};
    depth_dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    depth_dependency.dstSubpass = 0;
    depth_dependency.srcStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    depth_dependency.srcAccessMask = 0;
    depth_dependency.dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    depth_dependency.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    //subpass
    VkSubpassDescription subpass = {};
    subpass.pColorAttachments = &color_attachment_ref;
    subpass.colorAttachmentCount = 1;
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.pDepthStencilAttachment = &depth_attachment_ref;

    //render pass
    VkSubpassDependency dependencies[2] = { dependency , depth_dependency };
    VkAttachmentDescription attachments[2] = { color_attachment, depth_attachment };

    VkRenderPassCreateInfo render_pass_info = {};
    render_pass_info.pNext = nullptr;
    render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    render_pass_info.attachmentCount = 2;
    render_pass_info.pAttachments = &attachments[0];
    render_pass_info.subpassCount = 1;
    render_pass_info.pSubpasses = &subpass;
    render_pass_info.dependencyCount = 2;
    render_pass_info.pDependencies = &dependencies[0];

    VK_CHECK(vkCreateRenderPass(_device, &render_pass_info, nullptr, &_renderPass));

    _mainDeletionQueue.pushFunction([=]
        {
            vkDestroyRenderPass(_device, _renderPass, nullptr);
        });
}

void VulkanEngine::initFrameBuffers()
{
    VkFramebufferCreateInfo fb_info = {};
    fb_info.attachmentCount = 1;
    fb_info.pNext = nullptr;
    fb_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fb_info.renderPass = _renderPass;
    fb_info.height = _windowExtent.height;
    fb_info.width = _windowExtent.width;
    fb_info.layers = 1;

    const size_t swapchain_imageCount = _swapchainImages.size();
    _framebuffers = std::vector<VkFramebuffer>(swapchain_imageCount);

    for (size_t i = 0; i < swapchain_imageCount; i++)
    {
        VkImageView attachments[2] = { _swapchainImageViews[i], _depthImageView }; //only one depth because we can clear an reuse the same for each frame
        fb_info.attachmentCount = 2;
        fb_info.pAttachments = &attachments[0];
        VK_CHECK(vkCreateFramebuffer(_device, &fb_info, nullptr, &_framebuffers[i]));

        _mainDeletionQueue.pushFunction([=]
            {
                vkDestroyFramebuffer(_device, _framebuffers[i], nullptr);
                vkDestroyImageView(_device, _swapchainImageViews[i], nullptr);
            });
    }
}

void VulkanEngine::initSyncStructures()
{
    //upload contect fence
    VkFenceCreateInfo uploadFenceCreateInfo = vkinit::fence_create_info(); //We do not need to wait on it before sending commands
    VK_CHECK(vkCreateFence(_device, &uploadFenceCreateInfo, nullptr, &_uploadContext._uploadFence));
    _mainDeletionQueue.pushFunction([=]()
        {
            vkDestroyFence(_device, _uploadContext._uploadFence, nullptr);
        });


    //render and present fences/semaphores
    for (int i = 0; i < FRAME_OVERLAP; i++)
    {
        //fence
        VkFenceCreateInfo fence_create_info = vkinit::fence_create_info(VK_FENCE_CREATE_SIGNALED_BIT);
        VK_CHECK(vkCreateFence(_device, &fence_create_info, nullptr, &_frames[i]._renderFence));

        //semaphore
        VkSemaphoreCreateInfo semaphore_create_info = vkinit::semaphore_create_info();
        VK_CHECK(vkCreateSemaphore(_device, &semaphore_create_info, nullptr, &_frames[i]._presentSemaphore));
        VK_CHECK(vkCreateSemaphore(_device, &semaphore_create_info, nullptr, &_frames[i]._renderSemaphore));

        _mainDeletionQueue.pushFunction([=]
            {
                vkDestroySemaphore(_device, _frames[i]._presentSemaphore, nullptr);
                vkDestroySemaphore(_device, _frames[i]._renderSemaphore, nullptr);
            });
    }
}

VkPipeline PipelineBuilder::buildPipeline(VkDevice device, VkRenderPass pass)
{
    VkPipelineViewportStateCreateInfo viewportState = {};
    viewportState.pNext = nullptr;
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
    pipelineInfo.pNext = nullptr;
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.stageCount = (uint32_t)_shaderStages.size();
    pipelineInfo.pStages = _shaderStages.data();
    pipelineInfo.pVertexInputState = &_vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &_inputAssembly;
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
        cout << "failed to create graphics pipeline\n";
        return VK_NULL_HANDLE;
    }
    else
    {
        cout << "Successfully created graphics pipeline\n";
        return newPipeline;
    }
}
