//
// Created by Felix on 27/08/2025.
//

#include "betterAsBuilder.hpp"

#include <cstdint>
#include <vector>
#include <vulkan/vulkan_core.h>

#include "acceleration_structures.hpp"
#include "buffers_vk.hpp"
#include "resourceallocator_vk.hpp"

//--------------------------------------------------------------------------------------------------
// Initializing the allocator and querying the raytracing properties
//
void nvvk::BetterRtBuilder::setup(const VkDevice &device, nvvk::ResourceAllocator *allocator) {
    m_device     = device;
    m_debug.setup(device);
    m_alloc = allocator;
}

//--------------------------------------------------------------------------------------------------
// Destroying all allocations
//
void nvvk::BetterRtBuilder::destroy() {
    if (m_alloc != nullptr) {
        for (auto &b: m_blas) {
                m_alloc->destroy(b);
        }
        m_alloc->destroy(m_tlas);
        m_alloc->destroy(m_scratchBuffer);
        m_alloc->destroy(m_instanceBuffer);
        m_alloc->destroy(m_tlasScratchBuffer);
        m_destroyer.empty();
    }

    m_blas.clear();
}

//--------------------------------------------------------------------------------------------------
// Returning the constructed top-level acceleration structure
//
VkAccelerationStructureKHR nvvk::BetterRtBuilder::getAccelerationStructure() const {
    return m_tlas.accel;
}

//--------------------------------------------------------------------------------------------------
// Return the device address of a Blas previously created.
//
VkDeviceAddress nvvk::BetterRtBuilder::getBlasDeviceAddress(const uint32_t blasId) const {
    assert(static_cast<size_t>(blasId) < m_blas.size());
    VkAccelerationStructureDeviceAddressInfoKHR addressInfo{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
    addressInfo.accelerationStructure = m_blas[blasId].accel;
    return vkGetAccelerationStructureDeviceAddressKHR(m_device, &addressInfo);
}



//--------------------------------------------------------------------------------------------------
// Create all the BLAS from the vector of BlasInput
// - There will be one BLAS per input-vector entry
// - There will be as many BLAS as input.size()
// - The resulting BLAS (along with the inputs used to build) are stored in m_blas,
//   and can be referenced by index.
//
void nvvk::BetterRtBuilder::buildBlas(const VkCommandBuffer &cmdBuf,
                                           const std::vector<BlasInput> &input,
                                           const size_t size,
                                           VkBuildAccelerationStructureFlagsKHR flags) {

    auto numBlas = static_cast<uint32_t>(size);
    VkDeviceSize maxScratchSize{0}; // Largest scratch size

    std::vector<nvvk::AccelerationStructureBuildData> blasBuildData(numBlas);
    m_blas.resize(numBlas); // Resize to hold all the BLAS
    for (uint32_t idx = 0; idx < numBlas; idx++) {
        blasBuildData[idx].asType           = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        blasBuildData[idx].asGeometry       = input[idx].asGeometry;
        blasBuildData[idx].asBuildRangeInfo = input[idx].asBuildOffsetInfo;

        auto sizeInfo  = blasBuildData[idx].finalizeGeometry(m_device, input[idx].flags | flags);
        maxScratchSize = std::max(maxScratchSize, sizeInfo.buildScratchSize);
    }

    VkDeviceSize hintMaxBudget{256'000'000}; // 256 MB

    nvvk::BlasBuilder blasBuilder(m_alloc, m_device);

    uint32_t minAlignment = 128; /*m_rtASProperties.minAccelerationStructureScratchOffsetAlignment*/
    // 1) finding the largest scratch size
    VkDeviceSize scratchSize = blasBuilder.getScratchSize(hintMaxBudget, blasBuildData, minAlignment);
    // 2) allocating the scratch buffer

    m_scratchBuffer = m_alloc->createBuffer(scratchSize,
                                                           VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    // 3) getting the device address for the scratch buffer
    std::vector<VkDeviceAddress> scratchAddresses;
    blasBuilder.getScratchAddresses(hintMaxBudget, blasBuildData, m_scratchBuffer.address, scratchAddresses,
                                    minAlignment);


    bool finished = false;
    do {
        {
            finished = blasBuilder.cmdCreateParallelBlas(cmdBuf, blasBuildData, m_blas, scratchAddresses, hintMaxBudget);
            //Already Barrier in there
        }
    } while (!finished);

    //Cleanup
    //Might break stuff
    m_alloc->finalizeAndReleaseStaging();
}

//--------------------------------------------------------------------------------------------------
// Low level of Tlas creation - see buildTlas
//
void nvvk::BetterRtBuilder::cmdCreateTlas(VkCommandBuffer cmdBuf,
                                               uint32_t countInstance,
                                               VkDeviceAddress instBufferAddr,
                                               nvvk::Buffer &scratchBuffer,
                                               VkBuildAccelerationStructureFlagsKHR flags) {
    nvvk::AccelerationStructureBuildData tlasBuildData;
    tlasBuildData.asType = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;

    nvvk::AccelerationStructureGeometryInfo geo = tlasBuildData.makeInstanceGeometry(countInstance, instBufferAddr);
    tlasBuildData.addGeometry(geo);

    auto sizeInfo = tlasBuildData.finalizeGeometry(m_device, flags);

    // Allocate the scratch memory
    VkDeviceSize scratchSize = sizeInfo.buildScratchSize;
    scratchBuffer            = m_alloc->createBuffer(scratchSize,
                                                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                     VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    VkDeviceAddress scratchAddress = nvvk::getBufferDeviceAddress(m_device, scratchBuffer.buffer);
    NAME_VK(scratchBuffer.buffer);

    // Create and build the acceleration structure
    VkAccelerationStructureCreateInfoKHR createInfo = tlasBuildData.makeCreateInfo();

    m_tlas = m_alloc->createAcceleration(createInfo);
    NAME_VK(m_tlas.accel);
    NAME_VK(m_tlas.buffer.buffer);
    //Barrier inside, but possibly the wrong one
    tlasBuildData.cmdBuildAccelerationStructure(cmdBuf, m_tlas.accel, scratchAddress);
}