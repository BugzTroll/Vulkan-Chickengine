#include <vk_descriptors.h>
#include "iostream"

void DescriptorAllocatorGrowable::init(VkDevice device, uint32_t initialSets, std::span<PoolSizeRatio> poolRatios)
{
    ratios.clear();

    for (auto r : poolRatios)
    {
        ratios.push_back(r);
    }

    VkDescriptorPool newPool = createPool(device, initialSets, poolRatios);
    setsPerPool = initialSets *= 1.5f; //grow it in the next allocation
    readyPools.push_back(newPool);
}

void DescriptorAllocatorGrowable::clearPools(VkDevice device)
{
    for (auto p : readyPools)
    {
        vkResetDescriptorPool(device, p, 0);
    }

    for (auto p : fullPools)
    {
        vkResetDescriptorPool(device, p, 0);
        readyPools.push_back(p);
    }

    fullPools.clear();
}

void DescriptorAllocatorGrowable::destroyPools(VkDevice device)
{
}

VkDescriptorSet DescriptorAllocatorGrowable::allocate(VkDevice device, VkDescriptorSetLayout layout, void* pNext)
{
    VkDescriptorPool poolToUse = getOrCreatePool(device);
    
    VkDescriptorSetAllocateInfo allocInfo = {};
    allocInfo.pNext = pNext;
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = poolToUse;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &layout;

    VkDescriptorSet descriptorSet;
    VkResult result = vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet);

    //Try one more time on error
    if (result == VK_ERROR_OUT_OF_POOL_MEMORY || result == VK_ERROR_FRAGMENTED_POOL)
    {
        fullPools.push_back(poolToUse);
        poolToUse = getOrCreatePool(device);
        allocInfo.descriptorPool = poolToUse;

        VK_CHECK(vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet));
    }

    readyPools.push_back(poolToUse);
    return descriptorSet;


}

VkDescriptorPool DescriptorAllocatorGrowable::getOrCreatePool(VkDevice device)
{
    VkDescriptorPool pool;
	
    if (readyPools.size() != 0)
    {
        pool = readyPools.back();
        readyPools.pop_back();
    }
    else
    {
        pool = createPool(device, setsPerPool, ratios);
        setsPerPool = uint32_t(setsPerPool * 1.5);
        if (setsPerPool > 4092)
        {
            setsPerPool = 4092;
        }
    }

    return pool;
}

VkDescriptorPool DescriptorAllocatorGrowable::createPool(VkDevice device, uint32_t setCount, std::span<PoolSizeRatio> poolRatios)
{
    std::vector<VkDescriptorPoolSize> poolSizes;
    for (PoolSizeRatio ratio : poolRatios)
    {
        poolSizes.push_back(VkDescriptorPoolSize{ 
            .type = ratio.type,
            .descriptorCount = uint32_t(ratio.ratio * setCount) });
    }

    //descriptor pool
    VkDescriptorPoolCreateInfo poolInfo = {};
    poolInfo.pNext = nullptr;
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = (uint32_t)poolSizes.size();
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = 10;
    poolInfo.flags = 0;

    VkDescriptorPool newPool;
    vkCreateDescriptorPool(device, &poolInfo, nullptr, &newPool);
    return newPool;
}

void DescriptorWriter::writeImage(int binding, VkImageView image, VkSampler sampler, VkImageLayout layout, VkDescriptorType type, VkDescriptorSet& set)
{
    VkDescriptorImageInfo& info = imageInfos.emplace_back(VkDescriptorImageInfo{ 
        .sampler = sampler,
        .imageView = image,
        .imageLayout = layout
        });


    VkWriteDescriptorSet write = { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
    write.pNext = nullptr;
    write.dstBinding = binding;
    write.dstSet = set; // check
    write.descriptorCount = 1;
    write.descriptorType = type;
    write.pImageInfo = &info;

    writes.push_back(write);
}

void DescriptorWriter::writeBuffer(int binding, VkBuffer buffer, size_t size, size_t offset, VkDescriptorType type, VkDescriptorSet& set)
{
    VkDescriptorBufferInfo& info = bufferInfos.emplace_back(VkDescriptorBufferInfo
        { .buffer = buffer,
          .offset = offset,
          .range = size
        });

    VkWriteDescriptorSet write = { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
    write.dstBinding = binding;
    write.dstSet = set;
    write.descriptorCount = 1;
    write.descriptorType = type;
    write.pBufferInfo = &info;

    writes.push_back(write);
}

void DescriptorWriter::clear()
{
    imageInfos.clear();
    writes.clear();
    bufferInfos.clear();
}

void DescriptorWriter::updateSet(VkDevice device)
{
    vkUpdateDescriptorSets(device, (uint32_t)writes.size(), writes.data(), 0, nullptr);
}

void DescriptorLayoutBuilder::addBinding(uint32_t binding, VkDescriptorType type)
{
    bindings.push_back(VkDescriptorSetLayoutBinding
        {
            .binding = binding,
            .descriptorType = type,
            .descriptorCount = 1
        });
}

void DescriptorLayoutBuilder::clear()
{
    bindings.clear();
}

VkDescriptorSetLayout DescriptorLayoutBuilder::build(VkDevice device, VkShaderStageFlags shaderStages, void* next, VkDescriptorSetLayoutCreateFlags flags)
{
    for (auto& b : bindings) {
        b.stageFlags |= shaderStages;
    }

    VkDescriptorSetLayoutCreateInfo layoutInfo = {};
    layoutInfo.pNext = next;
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = uint32_t(bindings.size());
    layoutInfo.pBindings = bindings.data();
    layoutInfo.flags = flags;

    VkDescriptorSetLayout descriptorLayout;
    VK_CHECK(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorLayout));
    return descriptorLayout;
}
