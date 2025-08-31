#pragma once
#include <cstdint>
#include <stdexcept>

#include "commands_vk.hpp"
#include "debug_util_vk.hpp"
#include "raytraceKHR_vk.hpp"
#include "resourceallocator_vk.hpp"

namespace nvvk {
    class BetterRtBuilder {
        private:
            struct InstanceDestroyer {
                std::vector<nvvk::Buffer> buffers;
                BetterRtBuilder* builder;

                explicit InstanceDestroyer(BetterRtBuilder *builder) : builder(builder) {}

                void empty() {
                    for (auto buf : buffers) {
                        builder->m_alloc->destroy(buf);
                    }
                    buffers.clear();
                }

            };
        protected:
            std::vector<std::vector<nvvk::AccelKHR> > m_blas; // Bottom-level acceleration structure
            std::vector<nvvk::AccelKHR> m_tlas; // Top-level acceleration structure
            // Setup
            VkDevice m_device{VK_NULL_HANDLE};
            nvvk::ResourceAllocator *m_alloc{nullptr};
            nvvk::DebugUtil m_debug;

            std::vector<nvvk::Buffer> m_scratchBuffers;
            std::vector<nvvk::Buffer> m_instanceBuffers;
            std::vector<nvvk::Buffer> m_tlasScratchBuffers;

            InstanceDestroyer m_instDest = InstanceDestroyer(this);

        public:
            // Inputs used to build Bottom-level acceleration structure.
            // You manage the lifetime of the buffer(s) referenced by the VkAccelerationStructureGeometryKHRs within.
            // In particular, you must make sure they are still valid and not being modified when the BLAS is built or updated.
            struct BlasInput {
                // Data used to build acceleration structure geometry
                std::vector<VkAccelerationStructureGeometryKHR> asGeometry;
                std::vector<VkAccelerationStructureBuildRangeInfoKHR> asBuildOffsetInfo;
                VkBuildAccelerationStructureFlagsKHR flags{0};
            };

            // Initializing the allocator and querying the raytracing properties
            void setup(const VkDevice &device, nvvk::ResourceAllocator *allocator, uint32_t imageCount);

            // Destroying all allocations
            void destroy();

            // Returning the constructed top-level acceleration structure
            VkAccelerationStructureKHR getAccelerationStructure(const uint32_t frame) const;

            void clearBlas(const uint32_t frame) {

                //I do not think it is in use. The buffer can only be started when the previous frame has been cleared
                m_alloc->destroy(m_scratchBuffers[frame]);
                for (auto &blas: m_blas[frame]) {
                    m_alloc->destroy(blas);
                }
                m_blas[frame].clear();
                m_alloc->finalizeAndReleaseStaging();
            }

            void clearTlas(const uint32_t frame) {

                //Wait for fence possibly not needed
                m_instDest.empty();
                m_instDest.buffers.push_back(m_instanceBuffers[frame]);
                m_alloc->destroy(m_tlasScratchBuffers[frame]);
                if (m_tlas[frame].accel != VK_NULL_HANDLE) {
                    m_alloc->destroy(m_tlas[frame]);
                    m_tlas[frame] = {};
                }
                m_alloc->finalizeAndReleaseStaging();
            }

            // Return the Acceleration Structure Device Address of a BLAS Id
            VkDeviceAddress getBlasDeviceAddress(uint32_t blasId, uint32_t frame) const;


            void buildBlas(const VkCommandBuffer &cmdBuf, uint32_t curFrame, const std::vector<BlasInput> &input,
                           size_t size,
                           VkBuildAccelerationStructureFlagsKHR flags =
                                   VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR);


            void cmdCreateTlas(VkCommandBuffer cmdBuf,
                               uint32_t curFrame,
                               uint32_t countInstance,
                               VkDeviceAddress instBufferAddr,
                               nvvk::Buffer &scratchBuffer,
                               VkBuildAccelerationStructureFlagsKHR flags);

            // Build TLAS from an array of VkAccelerationStructureInstanceKHR
            // - Use motion=true with VkAccelerationStructureMotionInstanceNV
            // - The resulting TLAS will be stored in m_tlas
            // - update is to rebuild the Tlas with updated matrices, flag must have the 'allow_update'
            template<class T>
            void buildTlas(const VkCommandBuffer &cmdBuf, uint32_t curFrame,
                           const std::vector<T> &instances,
                           const size_t size,
                           VkBuildAccelerationStructureFlagsKHR flags =
                                   VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
                           bool update = false,
                           bool motion = false) {

                // Cannot call buildTlas twice except to update.
                assert(m_tlas[curFrame].accel == VK_NULL_HANDLE || update);
                uint32_t countInstance = static_cast<uint32_t>(size);

                // Create a buffer holding the actual instance data (matrices++) for use by the AS builder
                m_instanceBuffers[curFrame] = m_alloc->createBuffer(cmdBuf, instances,
                                                        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                                                        | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
                NAME_VK(m_instanceBuffers[curFrame].buffer);
                VkBufferDeviceAddressInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, nullptr,
                                                     m_instanceBuffers[curFrame].buffer};
                VkDeviceAddress instBufferAddr = vkGetBufferDeviceAddress(m_device, &bufferInfo);

                // Make sure the copy of the instance buffer are copied before triggering the acceleration structure build
                VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
                vkCmdPipelineBarrier(cmdBuf, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                                     VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1, &barrier, 0, nullptr,
                                     0, nullptr);

                // Creating the TLAS
                cmdCreateTlas(cmdBuf, curFrame, countInstance, instBufferAddr, m_tlasScratchBuffers[curFrame], flags);

                //Add barrier
                VkMemoryBarrier tlasBarrier {VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                                            nullptr,
                                            VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                                            VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR,

                };
                vkCmdPipelineBarrier(cmdBuf, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                                    VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                                    0, 1, &tlasBarrier, 0,
                                    nullptr, 0, nullptr);
            }


    };
}
