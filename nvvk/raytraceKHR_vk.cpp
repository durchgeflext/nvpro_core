/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <numeric>

#include "nvvk/buffers_vk.hpp"
#include "nvvk/raytraceKHR_vk.hpp"
#include "acceleration_structures.hpp"

//--------------------------------------------------------------------------------------------------
// Initializing the allocator and querying the raytracing properties
//
void nvvk::RaytracingBuilderKHR::setup(const VkDevice &device, nvvk::ResourceAllocator *allocator,
                                       uint32_t queueIndex) {
    m_device     = device;
    m_queueIndex = queueIndex;
    m_debug.setup(device);
    m_alloc = allocator;
}

//--------------------------------------------------------------------------------------------------
// Destroying all allocations
//
void nvvk::RaytracingBuilderKHR::destroy() {
    if (m_alloc) {
        for (auto &b: m_blas) {
            m_alloc->destroy(b);
        }
        m_alloc->destroy(m_tlas);
    }
    m_blas.clear();
}

//--------------------------------------------------------------------------------------------------
// Returning the constructed top-level acceleration structure
//
VkAccelerationStructureKHR nvvk::RaytracingBuilderKHR::getAccelerationStructure() const {
    return m_tlas.accel;
}

//--------------------------------------------------------------------------------------------------
// Return the device address of a Blas previously created.
//
VkDeviceAddress nvvk::RaytracingBuilderKHR::getBlasDeviceAddress(uint32_t blasId) {
    assert(size_t(blasId) < m_blas.size());
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
// - if flag has the 'Compact' flag, the BLAS will be compacted
//
void nvvk::RaytracingBuilderKHR::buildBlas(const std::vector<BlasInput> &input,
                                           VkBuildAccelerationStructureFlagsKHR flags) {
    m_cmdPool.init(m_device, m_queueIndex);
    auto numBlas = static_cast<uint32_t>(input.size());
    VkDeviceSize asTotalSize{0}; // Memory size of all allocated BLAS
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

    // Allocate the scratch buffers holding the temporary data of the acceleration structure builder
    nvvk::Buffer blasScratchBuffer;

    bool hasCompaction = hasFlag(flags, VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR);

    nvvk::BlasBuilder blasBuilder(m_alloc, m_device);

    uint32_t minAlignment = 128; /*m_rtASProperties.minAccelerationStructureScratchOffsetAlignment*/
    // 1) finding the largest scratch size
    VkDeviceSize scratchSize = blasBuilder.getScratchSize(hintMaxBudget, blasBuildData, minAlignment);
    // 2) allocating the scratch buffer
    blasScratchBuffer =
            m_alloc->createBuffer(scratchSize,
                                  VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    // 3) getting the device address for the scratch buffer
    std::vector<VkDeviceAddress> scratchAddresses;
    blasBuilder.getScratchAddresses(hintMaxBudget, blasBuildData, blasScratchBuffer.address, scratchAddresses,
                                    minAlignment);


    bool finished = false;
    do {
        {
            VkCommandBuffer cmd = m_cmdPool.createCommandBuffer();
            finished = blasBuilder.cmdCreateParallelBlas(cmd, blasBuildData, m_blas, scratchAddresses, hintMaxBudget);
            m_cmdPool.submitAndWait(cmd);
        }
        if (hasCompaction) {
            VkCommandBuffer cmd = m_cmdPool.createCommandBuffer();
            blasBuilder.cmdCompactBlas(cmd, blasBuildData, m_blas);
            m_cmdPool.submitAndWait(cmd); // Submit command buffer and call vkQueueWaitIdle
            blasBuilder.destroyNonCompactedBlas();
        }
    } while (!finished);
    if (hasCompaction) {
        LOGI("%s\n", blasBuilder.getStatistics().toString().c_str());
    }

    // Clean up
    m_alloc->finalizeAndReleaseStaging();
    m_alloc->destroy(blasScratchBuffer);
    m_cmdPool.deinit();
}

void nvvk::RaytracingBuilderKHR::buildTlas(const std::vector<VkAccelerationStructureInstanceKHR> &instances,
                                           VkBuildAccelerationStructureFlagsKHR flags
                                           /*= VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR*/,
                                           bool update /*= false*/) {
    buildTlas(instances, flags, update, false);
}

#ifdef VK_NV_ray_tracing_motion_blur
void nvvk::RaytracingBuilderKHR::buildTlas(const std::vector<VkAccelerationStructureMotionInstanceNV> &instances,
                                           VkBuildAccelerationStructureFlagsKHR flags
                                           /*= VK_BUILD_ACCELERATION_STRUCTURE_MOTION_BIT_NV*/,
                                           bool update /*= false*/) {
    buildTlas(instances, flags, update, true);
}
#endif

void nvvk::RaytracingBuilderKHR::buildTlasFromMultipleBlases(const std::vector<VkAccelerationStructureInstanceKHR>& instances, VkBuildAccelerationStructureFlagsKHR flags, bool update) {
    assert((m_tlas.accel == VK_NULL_HANDLE) && "TLAS is not empty");
    assert(!m_blas.empty() && "No BLAS available to build TLAS");
    assert(instances.size() == m_blas.size() && "Passed unequal amounts of instances compared to the existing as structures");

    const uint32_t countInstance = static_cast<uint32_t>(instances.size());

    // Create instance buffer
    nvvk::CommandPool cmdPool(m_device, m_queueIndex);
    VkCommandBuffer cmdBuf = cmdPool.createCommandBuffer();

    nvvk::Buffer instanceBuffer = m_alloc->createBuffer(
        cmdBuf,
        instances,
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
    NAME_VK(instanceBuffer.buffer);

    VkBufferDeviceAddressInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    bufferInfo.buffer = instanceBuffer.buffer;
    VkDeviceAddress instBufferAddress = vkGetBufferDeviceAddress(m_device, &bufferInfo);

    //MemoryBarrier to ensure instance Buffer is complete
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    vkCmdPipelineBarrier(cmdBuf,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                         0, 1, &barrier, 0, nullptr, 0, nullptr);

    VkAccelerationStructureGeometryKHR asGeom{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
        .pNext = nullptr,
        .geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR,
        .flags = VK_GEOMETRY_OPAQUE_BIT_KHR,
    };

    asGeom.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    asGeom.geometry.instances.arrayOfPointers = VK_FALSE;
    asGeom.geometry.instances.data.deviceAddress = instBufferAddress;

    // Build geometry info
    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    buildInfo.pNext         = nullptr;
    buildInfo.type          = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    buildInfo.flags         = flags;
    buildInfo.mode          = update
                              ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR
                              : VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfo.srcAccelerationStructure = VK_NULL_HANDLE;
    //dstAccelerationsStructure is declared later
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries   = &asGeom;
    buildInfo.ppGeometries  = nullptr;
    //scratch data is initialized later

    // Query size requirements
    VkAccelerationStructureBuildSizesInfoKHR sizeInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    vkGetAccelerationStructureBuildSizesKHR(m_device,
                                            VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                            &buildInfo,
                                            &countInstance,
                                            &sizeInfo);

    // Allocate or update TLAS
    if(!update)
    {
        VkAccelerationStructureCreateInfoKHR createInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
        createInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
        createInfo.size = sizeInfo.accelerationStructureSize;

        m_tlas = m_alloc->createAcceleration(createInfo);
        NAME_VK(m_tlas.accel);
    }
    buildInfo.dstAccelerationStructure = m_tlas.accel;

    //Scratch Data
    nvvk::Buffer scratchBuffer = m_alloc->createBuffer(
     update ? sizeInfo.updateScratchSize : sizeInfo.buildScratchSize,
     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

    VkDeviceAddress scratchAddress = nvvk::getBufferDeviceAddress(m_device, scratchBuffer.buffer);
    buildInfo.scratchData.deviceAddress = scratchAddress;

    // Build range info
    VkAccelerationStructureBuildRangeInfoKHR buildRangeInfo{};
    buildRangeInfo.primitiveCount  = countInstance;
    buildRangeInfo.primitiveOffset = 0;
    buildRangeInfo.firstVertex     = 0;
    buildRangeInfo.transformOffset = 0;
    const VkAccelerationStructureBuildRangeInfoKHR* pbuildRangeInfo = &buildRangeInfo;

    //infoCount: number of as to be build
    //pInfos: pointer to buildInformation for the as to be built
    //ppRangeInfos: pointer to infoCount pointers. Each pointer points to an array of buildRangeInfo
    //              each array has lengthpInfos[i].geometryCount
    vkCmdBuildAccelerationStructuresKHR(cmdBuf, 1, &buildInfo, &pbuildRangeInfo);

    // Submit work
    cmdPool.submitAndWait(cmdBuf);

    // Cleanup
    m_alloc->finalizeAndReleaseStaging();
    m_alloc->destroy(scratchBuffer);
    m_alloc->destroy(instanceBuffer);
}


//--------------------------------------------------------------------------------------------------
// Low level of Tlas creation - see buildTlas
//
void nvvk::RaytracingBuilderKHR::cmdCreateTlas(VkCommandBuffer cmdBuf,
                                               uint32_t countInstance,
                                               VkDeviceAddress instBufferAddr,
                                               nvvk::Buffer &scratchBuffer,
                                               VkBuildAccelerationStructureFlagsKHR flags,
                                               bool update,
                                               bool motion) {
    nvvk::AccelerationStructureBuildData tlasBuildData;
    tlasBuildData.asType = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;

    nvvk::AccelerationStructureGeometryInfo geo = tlasBuildData.makeInstanceGeometry(countInstance, instBufferAddr);
    tlasBuildData.addGeometry(geo);

    auto sizeInfo = tlasBuildData.finalizeGeometry(m_device, flags);

    // Allocate the scratch memory
    VkDeviceSize scratchSize = update ? sizeInfo.updateScratchSize : sizeInfo.buildScratchSize;
    scratchBuffer            = m_alloc->createBuffer(scratchSize,
                                                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                     VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    VkDeviceAddress scratchAddress = nvvk::getBufferDeviceAddress(m_device, scratchBuffer.buffer);
    NAME_VK(scratchBuffer.buffer);

    if (update) {
        // Update the acceleration structure
        tlasBuildData.asGeometry[0].geometry.instances.data.deviceAddress = instBufferAddr;
        tlasBuildData.cmdUpdateAccelerationStructure(cmdBuf, m_tlas.accel, scratchAddress);
    } else {
        // Create and build the acceleration structure
        VkAccelerationStructureCreateInfoKHR createInfo = tlasBuildData.makeCreateInfo();

#ifdef VK_NV_ray_tracing_motion_blur
        VkAccelerationStructureMotionInfoNV motionInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_MOTION_INFO_NV};
        motionInfo.maxInstances = countInstance;

        if (motion) {
            createInfo.createFlags = VK_ACCELERATION_STRUCTURE_CREATE_MOTION_BIT_NV;
            createInfo.pNext       = &motionInfo;
        }
#endif
        m_tlas = m_alloc->createAcceleration(createInfo);
        NAME_VK(m_tlas.accel);
        NAME_VK(m_tlas.buffer.buffer);
        tlasBuildData.cmdBuildAccelerationStructure(cmdBuf, m_tlas.accel, scratchAddress);
    }
}

//--------------------------------------------------------------------------------------------------
// Refit BLAS number blasIdx from updated buffer contents.
//
void nvvk::RaytracingBuilderKHR::updateBlas(uint32_t blasIdx, BlasInput &blas,
                                            VkBuildAccelerationStructureFlagsKHR flags) {
    assert(size_t(blasIdx) < m_blas.size());

    nvvk::AccelerationStructureBuildData buildData{VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR};
    buildData.asGeometry       = blas.asGeometry;
    buildData.asBuildRangeInfo = blas.asBuildOffsetInfo;
    auto sizeInfo              = buildData.finalizeGeometry(m_device, flags);

    // Allocate the scratch buffer and setting the scratch info
    nvvk::Buffer scratchBuffer = m_alloc->createBuffer(sizeInfo.updateScratchSize,
                                                       VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                                                       | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    // Update the instance buffer on the device side and build the TLAS
    nvvk::CommandPool genCmdBuf(m_device, m_queueIndex);
    VkCommandBuffer cmdBuf = genCmdBuf.createCommandBuffer();
    buildData.cmdUpdateAccelerationStructure(cmdBuf, m_blas[blasIdx].accel, scratchBuffer.address);
    genCmdBuf.submitAndWait(cmdBuf);

    m_alloc->destroy(scratchBuffer);
}
