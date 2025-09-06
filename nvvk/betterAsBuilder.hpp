#pragma once
#include <cstdint>

#include "commands_vk.hpp"
#include "debug_util_vk.hpp"
#include "raytraceKHR_vk.hpp"
#include "resourceallocator_vk.hpp"

namespace nvvk {
    class BetterRtBuilder {
        private:
            struct Destroyer {
                std::vector<nvvk::Buffer> buffers;
                std::vector<nvvk::AccelKHR> accel;
                BetterRtBuilder* builder;

                explicit Destroyer(BetterRtBuilder *builder) : builder(builder) {}

                void empty() {
                    for (auto &buf : buffers) {
                        builder->m_alloc->destroy(buf);
                    }
                    for (auto &acc : accel) {
                        builder->m_alloc->destroy(acc);
                    }
                    buffers.clear();
                    accel.clear();
                }

            };
        protected:
            std::vector<nvvk::AccelKHR> m_blas; // Bottom-level acceleration structure
            nvvk::AccelKHR m_tlas; // Top-level acceleration structure
            // Setup
            VkDevice m_device{VK_NULL_HANDLE};
            nvvk::ResourceAllocator *m_alloc{nullptr};
            nvvk::DebugUtil m_debug;

            nvvk::Buffer m_scratchBuffer;
            nvvk::Buffer m_instanceBuffer;
            nvvk::Buffer m_tlasScratchBuffer;

            Destroyer m_destroyer = Destroyer(this);

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
            void setup(const VkDevice &device, nvvk::ResourceAllocator *allocator);

            // Destroying all allocations
            void destroy();

            // Returning the constructed top-level acceleration structure
            VkAccelerationStructureKHR getAccelerationStructure() const;

            void clearBlas() {

                m_destroyer.buffers.push_back(m_scratchBuffer);
                for (auto &blas: m_blas) {
                    m_destroyer.accel.push_back(blas);
                }
                m_blas.clear();
            }

            void clearTlas() {

                m_destroyer.buffers.push_back(m_instanceBuffer);
                m_destroyer.buffers.push_back(m_tlasScratchBuffer);
                if (m_tlas.accel != VK_NULL_HANDLE) {
                    m_destroyer.accel.push_back(m_tlas);
                    m_tlas = {};
                }
            }

            void clearLeftovers() {
                m_destroyer.empty();
            }

            // Return the Acceleration Structure Device Address of a BLAS Id
            VkDeviceAddress getBlasDeviceAddress(uint32_t blasId) const;


            void buildBlas(const VkCommandBuffer &cmdBuf, const std::vector<BlasInput> &input,
                           size_t size,
                           VkBuildAccelerationStructureFlagsKHR flags =
                                   VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR);


            void cmdCreateTlas(VkCommandBuffer cmdBuf,
                               uint32_t countInstance,
                               VkDeviceAddress instBufferAddr,
                               nvvk::Buffer &scratchBuffer,
                               VkBuildAccelerationStructureFlagsKHR flags);

            // Build TLAS from an array of VkAccelerationStructureInstanceKHR
            // - Use motion=true with VkAccelerationStructureMotionInstanceNV
            // - The resulting TLAS will be stored in m_tlas
            // - update is to rebuild the Tlas with updated matrices, flag must have the 'allow_update'
            template<class T>
            void buildTlas(const VkCommandBuffer &cmdBuf,
                           const std::vector<T> &instances,
                           const size_t size,
                           VkBuildAccelerationStructureFlagsKHR flags =
                                   VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
                           bool update = false,
                           bool motion = false) {

                // Cannot call buildTlas twice except to update.
                assert(m_tlas.accel == VK_NULL_HANDLE || update);
                uint32_t countInstance = static_cast<uint32_t>(size);

                // Create a buffer holding the actual instance data (matrices++) for use by the AS builder
                m_instanceBuffer = m_alloc->createBuffer(cmdBuf, instances,
                                                        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                                                        | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
                NAME_VK(m_instanceBuffer.buffer);
                VkBufferDeviceAddressInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, nullptr,
                                                     m_instanceBuffer.buffer};
                VkDeviceAddress instBufferAddr = vkGetBufferDeviceAddress(m_device, &bufferInfo);

                // Make sure the copy of the instance buffer are copied before triggering the acceleration structure build
                VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
                vkCmdPipelineBarrier(cmdBuf, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                                     VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1, &barrier, 0, nullptr,
                                     0, nullptr);

                // Creating the TLAS
                cmdCreateTlas(cmdBuf, countInstance, instBufferAddr, m_tlasScratchBuffer, flags);

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
