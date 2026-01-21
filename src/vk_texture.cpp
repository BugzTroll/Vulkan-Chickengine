#include "vk_texture.h"
#include "iostream"
#include "vk_initializers.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

bool vkutil::load_image_from_file(VulkanEngine& engine, const char* file, AllocatedImage &outImage)
{
	int texWidth, texHeight, texChannels;

	stbi_uc* pixels = stbi_load(file, &texWidth, &texHeight, &texChannels, STBI_rgb_alpha); //load RGBA into cpu memory
	
	if (!pixels)
	{
		std::cout << "Error loading file %s" << file << std::endl;
		return false;
	}

	//allocate to GPU
	void* pixelptr = pixels;
	size_t imgSize = texWidth * texHeight * 4; //4 bytes per pixels

	//staging buffer used for GPU copy ( staging buffer is CPU )
	AllocatedBuffer stagingBuffer = engine.createBuffer(imgSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);
	
	//Copy image data to staging buffer
	void* data;
	vmaMapMemory(engine._allocator, stagingBuffer._allocation, &data);
	memcpy(data, pixelptr, imgSize);
	vmaUnmapMemory(engine._allocator, stagingBuffer._allocation);

	stbi_image_free(pixels);

	VkFormat imageFormat = VK_FORMAT_R8G8B8A8_SRGB;

	//We now create the GPU image
	VkExtent3D imageExtent;
	imageExtent.width = static_cast<uint32_t>(texWidth);
	imageExtent.height = static_cast<uint32_t>(texHeight);
	imageExtent.depth = 1;

	VkImageCreateInfo imgCreateInfo = vkinit::image_create_info(imageFormat, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, imageExtent); // we transfer from cpu to gpu and then sample it in the shaders

	AllocatedImage newImage;

	VmaAllocationCreateInfo imgAllocInfo = {};
	imgAllocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY; //allocated on VRAM

	//allocate and create the image ( still empty )
	vmaCreateImage(engine._allocator, &imgCreateInfo, &imgAllocInfo, &newImage._image, &newImage._allocation, nullptr);

	//transfer the image from CPU to GPU ( staging buffer -> NewImage )
	engine.ImmediateSubmit([&](VkCommandBuffer cmdBuffer)
	{
			// Tells what part of the image we will transform
			VkImageSubresourceRange range;
			range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			range.baseMipLevel = 0;
			range.levelCount = 1; // just one mip
			range.baseArrayLayer = 0;
			range.layerCount = 1; // just one layer

			// we need an image memory barrier to transform the image to the correct formats and layout
			VkImageMemoryBarrier imageBarrier_ToTransfer = {};
			imageBarrier_ToTransfer.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
			imageBarrier_ToTransfer.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			imageBarrier_ToTransfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; // prepare image into the layout to be a destination of memory transfer
			imageBarrier_ToTransfer.image = newImage._image; //points to the image
			imageBarrier_ToTransfer.subresourceRange = range; //points to the subressource
			imageBarrier_ToTransfer.srcAccessMask = 0;
			imageBarrier_ToTransfer.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

			//VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT is the earliest possible point in the pipeline "pseudo-stage"
			//We create a pipeline barrier
			vkCmdPipelineBarrier(cmdBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &imageBarrier_ToTransfer);

			VkBufferImageCopy copyRegion = {};
			copyRegion.bufferOffset = 0;
			copyRegion.bufferRowLength = 0;
			copyRegion.bufferImageHeight = 0;
			copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			copyRegion.imageSubresource.mipLevel = 0;
			copyRegion.imageSubresource.baseArrayLayer = 0;
			copyRegion.imageSubresource.layerCount = 1;
			copyRegion.imageExtent = imageExtent;

			//Actual copy
			vkCmdCopyBufferToImage(cmdBuffer, stagingBuffer._buffer, newImage._image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);

			//Then we need to put back the image layout to make it shader readable
			VkImageMemoryBarrier imageBarrier_toReadable = imageBarrier_ToTransfer;
			imageBarrier_toReadable.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; //how the memory is organized //ex:tiling
			imageBarrier_toReadable.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			imageBarrier_toReadable.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; //how the memory is actually used //garentees memory visibility and ordering
			imageBarrier_toReadable.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

			//barrier to make it readable from fragment shaders
			vkCmdPipelineBarrier(cmdBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &imageBarrier_toReadable);
	});

	engine._mainDeletionQueue.pushFunction([=]
		{
			vmaDestroyImage(engine._allocator, newImage._image, newImage._allocation);
		});

	vmaDestroyBuffer(engine._allocator, stagingBuffer._buffer, stagingBuffer._allocation);

	std::cout << "Texture loaded successfully !" << file << std::endl;

	outImage = newImage;

	return true;

}
