#include "vk_utils.h"

#include <iostream>
#include <fstream>

bool vk_utils::loadShaderModule(VkDevice device, const char* filePath, VkShaderModule* outShaderModule)
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
    if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) //common to fail because of shader issues, so we use somthing more robust
    {
        return false;

    }

    *outShaderModule = shaderModule;
    return true;
}
