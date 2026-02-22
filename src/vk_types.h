// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.
#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>
#include <span>
#include <array>
#include <functional>
#include <deque>

#include <vulkan/vulkan.h>
#include <vulkan/vk_enum_string_helper.h>
#include <vk_mem_alloc.h>


#include <fmt/core.h>

#include <glm/mat4x4.hpp>
#include <glm/vec4.hpp>

struct AllocatedBuffer 
{
    VkBuffer _buffer;
    VmaAllocation _allocation;
    VmaAllocationInfo _info;
};

struct AllocatedImage
{
    VkImage _image;
    VmaAllocation _allocation;
    VkFormat _imageFormat;
    VkExtent3D _imageExtent;
    VkImageView _imageView;
};

enum class EMaterialPass
{
    MainColor,
    Transparent,
    Other
};

struct Material
{
    VkDescriptorSet textureSet{ VK_NULL_HANDLE }; //texture defaulted to null
    VkPipeline pipeline;
    VkPipelineLayout layout;
};

struct MaterialInstance
{
    Material material;
    VkDescriptorSet materialSet;
    EMaterialPass passType;
};


struct GLTFMaterial
{
    MaterialInstance data;
};

struct GeoSurface
{
    uint32_t startIndex;
    uint32_t count;
    std::shared_ptr<GLTFMaterial> material;
};

#define VK_CHECK(x)                                                  \
    do                                                               \
    {                                                                \
        VkResult err = x;                                            \
        if (err)                                                     \
        {                                                            \
            std::cout << "Oh ohhh vilain Vulkan error detected" << err << std::endl; \
            abort();                                                 \
        }                                                            \
    } while(0)                                                       \