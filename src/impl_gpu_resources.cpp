#include "impl_gpu_resources.hpp"

#include <daxa/daxa.inl>

namespace daxa
{
    auto GPUResourceId::is_empty() const -> bool
    {
        return version == 0;
    }

    auto ImageId::default_view() const -> ImageViewId
    {
        return ImageViewId{{.index = index, .version = version}};
    }

    auto to_string(GPUResourceId const & id) -> std::string
    {
        return std::format("index: {}, version: {}", static_cast<u32>(id.index), static_cast<u32>(id.version));
    }

    auto to_string(ImageViewType const & type) -> std::string_view
    {
        switch (type)
        {
        case ImageViewType::REGULAR_1D: return "REGULAR_1D";
        case ImageViewType::REGULAR_2D: return "REGULAR_2D";
        case ImageViewType::REGULAR_3D: return "REGULAR_3D";
        case ImageViewType::CUBE: return "CUBE";
        case ImageViewType::REGULAR_1D_ARRAY: return "REGULAR_1D_ARRAY";
        case ImageViewType::REGULAR_2D_ARRAY: return "REGULAR_2D_ARRAY";
        case ImageViewType::CUBE_ARRAY: return "CUBE_ARRAY";
        default: return "NONE";
        }
    }

    auto GPUShaderResourceTable::initialize(u32 max_buffers, u32 max_images, u32 max_samplers, u32 max_acceleration_structures,
                                            VkDevice device, VkBuffer device_address_buffer,
                                            PFN_vkSetDebugUtilsObjectNameEXT vkSetDebugUtilsObjectNameEXT) -> daxa_Result
    {
        daxa_Result result = DAXA_RESULT_SUCCESS;
        defer
        {
            if (result != DAXA_RESULT_SUCCESS)
            {
                if (this->vk_descriptor_pool)
                {
                    vkDestroyDescriptorPool(device, this->vk_descriptor_pool, nullptr);
                }
                if (this->vk_descriptor_set_layout)
                {
                    vkDestroyDescriptorSetLayout(device, this->vk_descriptor_set_layout, nullptr);
                }
            }
        };

        bool const ray_tracing_enabled = max_acceleration_structures != (~0u);

        buffer_slots.max_resources = max_buffers;
        image_slots.max_resources = max_images;
        sampler_slots.max_resources = max_samplers;
        if (ray_tracing_enabled)
        {
            tlas_slots.max_resources = max_acceleration_structures;
            blas_slots.max_resources = 1'000'000; // TODO(Raytracing): Should we have a smarter limit?
        }

        VkDescriptorPoolSize const buffer_descriptor_pool_size{
            .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = buffer_slots.max_resources + 1,
        };

        VkDescriptorPoolSize const storage_image_descriptor_pool_size{
            .type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .descriptorCount = image_slots.max_resources,
        };

        VkDescriptorPoolSize const sampled_image_descriptor_pool_size{
            .type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
            .descriptorCount = image_slots.max_resources,
        };

        VkDescriptorPoolSize const sampler_descriptor_pool_size{
            .type = VK_DESCRIPTOR_TYPE_SAMPLER,
            .descriptorCount = sampler_slots.max_resources,
        };

        auto pool_sizes = std::vector{
            buffer_descriptor_pool_size,
            storage_image_descriptor_pool_size,
            sampled_image_descriptor_pool_size,
            sampler_descriptor_pool_size,
        };
        if (ray_tracing_enabled)
        {
            VkDescriptorPoolSize const as_descriptor_pool_size{
                .type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
                .descriptorCount = tlas_slots.max_resources,
            };
            pool_sizes.push_back(as_descriptor_pool_size);
        }

        VkDescriptorPoolCreateInfo const vk_descriptor_pool_create_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT | VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT,
            .maxSets = 1,
            .poolSizeCount = static_cast<u32>(pool_sizes.size()),
            .pPoolSizes = pool_sizes.data(),
        };

        result = static_cast<daxa_Result>(vkCreateDescriptorPool(device, &vk_descriptor_pool_create_info, nullptr, &this->vk_descriptor_pool));
        _DAXA_RETURN_IF_ERROR(result, result)

        if (vkSetDebugUtilsObjectNameEXT != nullptr)
        {
            auto const * descriptor_pool_name = "mega descriptor pool";
            VkDebugUtilsObjectNameInfoEXT const descriptor_pool_name_info{
                .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
                .pNext = nullptr,
                .objectType = VK_OBJECT_TYPE_DESCRIPTOR_POOL,
                .objectHandle = std::bit_cast<uint64_t>(vk_descriptor_pool),
                .pObjectName = descriptor_pool_name,
            };
            vkSetDebugUtilsObjectNameEXT(device, &descriptor_pool_name_info);
        }

        VkDescriptorSetLayoutBinding const buffer_descriptor_set_layout_binding{
            .binding = DAXA_STORAGE_BUFFER_BINDING,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = static_cast<u32>(buffer_slots.max_resources),
            .stageFlags = VK_SHADER_STAGE_ALL,
            .pImmutableSamplers = nullptr,
        };

        VkDescriptorSetLayoutBinding const storage_image_descriptor_set_layout_binding{
            .binding = DAXA_STORAGE_IMAGE_BINDING,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .descriptorCount = static_cast<u32>(image_slots.max_resources),
            .stageFlags = VK_SHADER_STAGE_ALL,
            .pImmutableSamplers = nullptr,
        };

        VkDescriptorSetLayoutBinding const sampled_image_descriptor_set_layout_binding{
            .binding = DAXA_SAMPLED_IMAGE_BINDING,
            .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
            .descriptorCount = static_cast<u32>(image_slots.max_resources),
            .stageFlags = VK_SHADER_STAGE_ALL,
            .pImmutableSamplers = nullptr,
        };

        VkDescriptorSetLayoutBinding const sampler_descriptor_set_layout_binding{
            .binding = DAXA_SAMPLER_BINDING,
            .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER,
            .descriptorCount = static_cast<u32>(sampler_slots.max_resources),
            .stageFlags = VK_SHADER_STAGE_ALL,
            .pImmutableSamplers = nullptr,
        };

        VkDescriptorSetLayoutBinding const buffer_address_buffer_descriptor_set_layout_binding{
            .binding = DAXA_BUFFER_DEVICE_ADDRESS_BUFFER_BINDING,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_ALL,
            .pImmutableSamplers = nullptr,
        };

        auto descriptor_set_layout_bindings = std::vector{
            buffer_descriptor_set_layout_binding,
            storage_image_descriptor_set_layout_binding,
            sampled_image_descriptor_set_layout_binding,
            sampler_descriptor_set_layout_binding,
            buffer_address_buffer_descriptor_set_layout_binding,
        };

        if (ray_tracing_enabled)
        {
            VkDescriptorSetLayoutBinding const as_descriptor_set_layout_binding{
                .binding = DAXA_ACCELERATION_STRUCTURE_BINDING,
                .descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
                .descriptorCount = static_cast<u32>(tlas_slots.max_resources),
                .stageFlags = VK_SHADER_STAGE_ALL,
                .pImmutableSamplers = nullptr,
            };
            descriptor_set_layout_bindings.push_back(as_descriptor_set_layout_binding);
        }

        auto vk_descriptor_binding_flags = std::vector{
            VkDescriptorBindingFlags{VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT},
            VkDescriptorBindingFlags{VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT},
            VkDescriptorBindingFlags{VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT},
            VkDescriptorBindingFlags{VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT},
            VkDescriptorBindingFlags{VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT},
        };
        if (ray_tracing_enabled)
        {
            vk_descriptor_binding_flags.push_back({VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT});
        }

        VkDescriptorSetLayoutBindingFlagsCreateInfo vk_descriptor_set_layout_binding_flags_create_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
            .pNext = nullptr,
            .bindingCount = static_cast<u32>(vk_descriptor_binding_flags.size()),
            .pBindingFlags = vk_descriptor_binding_flags.data(),
        };

        VkDescriptorSetLayoutCreateInfo const vk_descriptor_set_layout_create_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .pNext = &vk_descriptor_set_layout_binding_flags_create_info,
            .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
            .bindingCount = static_cast<u32>(descriptor_set_layout_bindings.size()),
            .pBindings = descriptor_set_layout_bindings.data(),
        };

        result = static_cast<daxa_Result>(vkCreateDescriptorSetLayout(device, &vk_descriptor_set_layout_create_info, nullptr, &this->vk_descriptor_set_layout));
        _DAXA_RETURN_IF_ERROR(result, result)

        if (vkSetDebugUtilsObjectNameEXT != nullptr)
        {
            auto const * name = "mega descriptor set layout";
            VkDebugUtilsObjectNameInfoEXT const name_info{
                .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
                .pNext = nullptr,
                .objectType = VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT,
                .objectHandle = std::bit_cast<uint64_t>(vk_descriptor_set_layout),
                .pObjectName = name,
            };
            vkSetDebugUtilsObjectNameEXT(device, &name_info);
        }

        VkDescriptorSetAllocateInfo const vk_descriptor_set_allocate_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .pNext = nullptr,
            .descriptorPool = this->vk_descriptor_pool,
            .descriptorSetCount = 1,
            .pSetLayouts = &this->vk_descriptor_set_layout,
        };

        result = static_cast<daxa_Result>(vkAllocateDescriptorSets(device, &vk_descriptor_set_allocate_info, &this->vk_descriptor_set));
        _DAXA_RETURN_IF_ERROR(result, result)

        if (vkSetDebugUtilsObjectNameEXT != nullptr)
        {
            auto const * name = "mega descriptor set";
            VkDebugUtilsObjectNameInfoEXT const name_info{
                .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
                .pNext = nullptr,
                .objectType = VK_OBJECT_TYPE_DESCRIPTOR_SET,
                .objectHandle = std::bit_cast<uint64_t>(vk_descriptor_set),
                .pObjectName = name,
            };
            vkSetDebugUtilsObjectNameEXT(device, &name_info);
        }

        auto vk_descriptor_set_layouts = std::array{this->vk_descriptor_set_layout};
        VkPipelineLayoutCreateInfo vk_pipeline_create_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .pNext = nullptr,
            .flags = {},
            .setLayoutCount = static_cast<u32>(vk_descriptor_set_layouts.size()),
            .pSetLayouts = vk_descriptor_set_layouts.data(),
            .pushConstantRangeCount = 0,
            .pPushConstantRanges = nullptr,
        };

        result = static_cast<daxa_Result>(vkCreatePipelineLayout(device, &vk_pipeline_create_info, nullptr, pipeline_layouts.data()));
        _DAXA_RETURN_IF_ERROR(result, result)

        if (vkSetDebugUtilsObjectNameEXT != nullptr)
        {
            auto const * name = "pipeline layout (push constant size 0)";
            VkDebugUtilsObjectNameInfoEXT const name_info{
                .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
                .pNext = nullptr,
                .objectType = VK_OBJECT_TYPE_PIPELINE_LAYOUT,
                .objectHandle = std::bit_cast<uint64_t>(*pipeline_layouts.data()),
                .pObjectName = name,
            };
            vkSetDebugUtilsObjectNameEXT(device, &name_info);
        }

        for (usize i = 1; i < PIPELINE_LAYOUT_COUNT; ++i)
        {
            VkPushConstantRange const vk_push_constant_range{
                .stageFlags = VK_SHADER_STAGE_ALL,
                .offset = 0,
                .size = static_cast<u32>(i * 4),
            };
            vk_pipeline_create_info.pushConstantRangeCount = 1;
            vk_pipeline_create_info.pPushConstantRanges = &vk_push_constant_range;
            result = static_cast<daxa_Result>(vkCreatePipelineLayout(device, &vk_pipeline_create_info, nullptr, &pipeline_layouts.at(i)));
            _DAXA_RETURN_IF_ERROR(result, result)

            if (vkSetDebugUtilsObjectNameEXT != nullptr)
            {
                auto name = std::format("pipeline layout (push constant size {})", i * 4);
                VkDebugUtilsObjectNameInfoEXT const name_info{
                    .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
                    .pNext = nullptr,
                    .objectType = VK_OBJECT_TYPE_PIPELINE_LAYOUT,
                    .objectHandle = std::bit_cast<uint64_t>(pipeline_layouts.at(i)),
                    .pObjectName = name.c_str(),
                };
                vkSetDebugUtilsObjectNameEXT(device, &name_info);
            }
        }

        VkDescriptorBufferInfo const write_buffer{
            .buffer = device_address_buffer,
            .offset = 0,
            .range = VK_WHOLE_SIZE,
        };

        VkWriteDescriptorSet const write{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = nullptr,
            .dstSet = this->vk_descriptor_set,
            .dstBinding = DAXA_BUFFER_DEVICE_ADDRESS_BUFFER_BINDING,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pImageInfo = nullptr,
            .pBufferInfo = &write_buffer,
            .pTexelBufferView = nullptr,
        };

        vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);

        return result;
    }

    void GPUShaderResourceTable::cleanup(VkDevice device)
    {
        [[maybe_unused]] auto print_remaining = [&](std::string prefix, auto & pages)
        {
            std::string ret{prefix + "\nthis can happen due to not waiting for the gpu to finish executing, as daxa defers destruction. List of survivors:\n"};
            for (auto & page : pages)
            {
                if (page)
                {
                    for (auto & slot : *page)
                    {
                        bool handle_invalid = {};
                        if constexpr (std::is_same_v<decltype(slot.first), ImplBufferSlot>)
                        {
                            handle_invalid = slot.first.vk_buffer == VK_NULL_HANDLE;
                        }
                        if constexpr (std::is_same_v<decltype(slot.first), ImplImageSlot>)
                        {
                            handle_invalid = slot.first.vk_image == VK_NULL_HANDLE;
                        }
                        if constexpr (std::is_same_v<decltype(slot.first), ImplSamplerSlot>)
                        {
                            handle_invalid = slot.first.vk_sampler == VK_NULL_HANDLE;
                        }
                        if (!handle_invalid)
                        {
                            ret += std::format("debug name : \"{}\"", r_cast<SmallString const *>(&slot.first.info.name)->view());
                            ret += "\n";
                        }
                    }
                }
            }
            return ret;
        };
        DAXA_DBG_ASSERT_TRUE_M(buffer_slots.free_index_stack.size() == buffer_slots.next_index, print_remaining("Detected leaked buffers; not all buffers have been destroyed before destroying the device;", buffer_slots.pages));
        DAXA_DBG_ASSERT_TRUE_M(image_slots.free_index_stack.size() == image_slots.next_index, print_remaining("Detected leaked images; not all images have been destroyed before destroying the device;", image_slots.pages));
        DAXA_DBG_ASSERT_TRUE_M(sampler_slots.free_index_stack.size() == sampler_slots.next_index, print_remaining("Detected leaked samplers; not all samplers have been destroyed before destroying the device;", sampler_slots.pages));
        for (usize i = 0; i < PIPELINE_LAYOUT_COUNT; ++i)
        {
            vkDestroyPipelineLayout(device, pipeline_layouts.at(i), nullptr);
        }
        vkDestroyDescriptorSetLayout(device, this->vk_descriptor_set_layout, nullptr);
        vkResetDescriptorPool(device, this->vk_descriptor_pool, {});
        vkDestroyDescriptorPool(device, this->vk_descriptor_pool, nullptr);
    }

    void write_descriptor_set_sampler(VkDevice vk_device, VkDescriptorSet vk_descriptor_set, VkSampler vk_sampler, u32 index)
    {
        VkDescriptorImageInfo const vk_descriptor_image_info{
            .sampler = vk_sampler,
            .imageView = VK_NULL_HANDLE,
            .imageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        };

        VkWriteDescriptorSet const vk_write_descriptor_set_storage{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = nullptr,
            .dstSet = vk_descriptor_set,
            .dstBinding = DAXA_SAMPLER_BINDING,
            .dstArrayElement = index,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER,
            .pImageInfo = &vk_descriptor_image_info,
            .pBufferInfo = nullptr,
            .pTexelBufferView = nullptr,
        };

        vkUpdateDescriptorSets(vk_device, 1, &vk_write_descriptor_set_storage, 0, nullptr);
    }

    void write_descriptor_set_buffer(VkDevice vk_device, VkDescriptorSet vk_descriptor_set, VkBuffer vk_buffer, VkDeviceSize offset, VkDeviceSize range, u32 index)
    {
        VkDescriptorBufferInfo const vk_descriptor_image_info{
            .buffer = vk_buffer,
            .offset = offset,
            .range = range,
        };

        VkWriteDescriptorSet const vk_write_descriptor_set{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = nullptr,
            .dstSet = vk_descriptor_set,
            .dstBinding = DAXA_STORAGE_BUFFER_BINDING,
            .dstArrayElement = index,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pImageInfo = nullptr,
            .pBufferInfo = &vk_descriptor_image_info,
            .pTexelBufferView = nullptr,
        };

        vkUpdateDescriptorSets(vk_device, 1, &vk_write_descriptor_set, 0, nullptr);
    }

    void write_descriptor_set_image(VkDevice vk_device, VkDescriptorSet vk_descriptor_set, VkImageView vk_image_view, ImageUsageFlags usage, u32 index)
    {
        u32 descriptor_set_write_count = 0;
        std::array<VkWriteDescriptorSet, 2> descriptor_set_writes = {};

        VkDescriptorImageInfo const vk_descriptor_image_info{
            .sampler = VK_NULL_HANDLE,
            .imageView = vk_image_view,
            .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
        };

        VkWriteDescriptorSet const vk_write_descriptor_set{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = nullptr,
            .dstSet = vk_descriptor_set,
            .dstBinding = DAXA_STORAGE_IMAGE_BINDING,
            .dstArrayElement = index,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .pImageInfo = &vk_descriptor_image_info,
            .pBufferInfo = nullptr,
            .pTexelBufferView = nullptr,
        };

        if ((usage & ImageUsageFlagBits::SHADER_STORAGE) != ImageUsageFlagBits::NONE)
        {
            descriptor_set_writes.at(descriptor_set_write_count++) = vk_write_descriptor_set;
        }

        VkDescriptorImageInfo const vk_descriptor_image_info_sampled{
            .sampler = VK_NULL_HANDLE,
            .imageView = vk_image_view,
            .imageLayout = VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL,
        };

        VkWriteDescriptorSet const vk_write_descriptor_set_sampled{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = nullptr,
            .dstSet = vk_descriptor_set,
            .dstBinding = DAXA_SAMPLED_IMAGE_BINDING,
            .dstArrayElement = index,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
            .pImageInfo = &vk_descriptor_image_info_sampled,
            .pBufferInfo = nullptr,
            .pTexelBufferView = nullptr,
        };

        if ((usage & ImageUsageFlagBits::SHADER_SAMPLED) != ImageUsageFlagBits::NONE)
        {
            descriptor_set_writes.at(descriptor_set_write_count++) = vk_write_descriptor_set_sampled;
        }

        vkUpdateDescriptorSets(vk_device, descriptor_set_write_count, descriptor_set_writes.data(), 0, nullptr);
    }

    void write_descriptor_set_acceleration_structure(VkDevice vk_device, VkDescriptorSet vk_descriptor_set, VkAccelerationStructureKHR vk_acceleration_structure, u32 index)
    {
        VkWriteDescriptorSetAccelerationStructureKHR vk_write_descriptor_set_as = {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
            .pNext = nullptr,
            .accelerationStructureCount = 1,
            .pAccelerationStructures = &vk_acceleration_structure,
        };

        VkWriteDescriptorSet const vk_write_descriptor_set{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = &vk_write_descriptor_set_as,
            .dstSet = vk_descriptor_set,
            .dstBinding = DAXA_ACCELERATION_STRUCTURE_BINDING,
            .dstArrayElement = index,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
            .pImageInfo = nullptr,
            .pBufferInfo = nullptr,
            .pTexelBufferView = nullptr,
        };

        vkUpdateDescriptorSets(vk_device, 1, &vk_write_descriptor_set, 0, nullptr);
    }
} // namespace daxa
