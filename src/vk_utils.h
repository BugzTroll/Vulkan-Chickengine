#pragma once

#include <vk_types.h>

class vk_utils
{
public:
	//shaders
	static bool loadShaderModule(VkDevice device, const char* filePath, VkShaderModule* outShaderModule);
};