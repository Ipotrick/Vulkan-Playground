#include <daxa/daxa.hpp>
#include <iostream>

namespace tests
{
    using namespace daxa::types;

    void simplest(daxa::Instance & instance)
    {
        auto device = instance.create_device_2(instance.choose_device({}, {}));
    }
    void device_selection(daxa::Instance & instance)
    {
        try
        {
            // When creating a device you have to fill out DeviceInfo2.
            // Choose a name, resource limits, explicit features.
            // After that is done, list the physical devices and choose one of them.

            daxa::DeviceInfo2 device_info = {.name = "my device"};

            std::span<daxa::DeviceProperties const> device_properties_list = instance.list_devices_properties();
            for (u32 i = 0; i < device_properties_list.size(); ++i)
            {
                // The device properties contain all information needed to choose a device:
                // limits, feature limits, queue count, explicit features, implicit features.
                daxa::DeviceProperties const & properties = device_properties_list[i];

                // The properties also include an enum declaring if a required feature is not present.
                if (properties.missing_required_feature != daxa::MissingRequiredVkFeature::NONE)
                {
                    continue;
                }

                // We can also check if the device has all the features we need:
                // Explicit features are ones the have to be manually enabled, otherwise the remain disabled!
                daxa::ExplicitFeatureFlags required_explicit_features = {};
                // Implicit features are ones that always get enabled when present.
                daxa::ImplicitFeatureFlags required_implicit_features = {};
                if (!(properties.explicit_features & required_explicit_features) || !(properties.implicit_features & required_implicit_features))
                {
                    continue;
                }
                
                device_info.physical_device_index = i;
                break;
            }
            
            // Note there are a number of other qualities one should check for to choose a device other then missing features!
            // For this purpose daxa has an automatic device choose function: instance.choose_device() which takes required features and a device info to pick the first fit.
            // auto device = instance.create_device_2(instance.choose_device({}/* required implicit features */, device_info));

            // We will create the device manually here as an example:
            auto device = instance.create_device_2(device_info);

            // once the device is created, you can query its properties, such
            // as its name and much more! These are the same properties we used
            // to discriminate in the GPU selection.
            std::cout << device.properties().device_name << std::endl;
        }
        catch (std::runtime_error error)
        {
            std::cout << "failed test \"device_selection\": " << error.what() << std::endl;
            exit(-1);
        }
    }
    constexpr daxa::BufferInfo test_buffer_info = {
        .size = 64,
        .name = "test buffer",
    };
    constexpr daxa::ImageInfo test_image_info = {
        .size = {64, 64, 1},
        .usage = daxa::ImageUsageFlagBits::SHADER_STORAGE,
        .name = "test image",
    };
    constexpr daxa::TlasInfo test_tlas_info = {
        .size = 256,
        .name = "test tlas",
    };
    constexpr daxa::BlasInfo test_blas_info = {
        .size = 256,
        .name = "test blas",
    };
    void sro_creation(daxa::Instance & instance)
    {
        try
        {
            auto device = instance.create_device_2(instance.choose_device({}, {}));
            auto test_buffer = device.create_buffer(test_buffer_info);
            auto test_image = device.create_image(test_image_info);
            auto test_image_view = device.create_image_view({
                .type = daxa::ImageViewType::REGULAR_2D_ARRAY,
                .image = test_image,
                .name = "test image view",
            });
            auto test_sampler = device.create_sampler({
                .name = "test sampler",
            });
            device.destroy_sampler(test_sampler);
            device.destroy_image_view(test_image_view);
            device.destroy_image(test_image);
            device.destroy_buffer(test_buffer);
        }
        catch (std::runtime_error error)
        {
            std::cout << "failed test \"sro_creation\": " << error.what() << std::endl;
            exit(-1);
        }
    }
    void sro_aliased_suballocation(daxa::Instance & instance)
    {
        try
        {
            auto device = instance.create_device_2(instance.choose_device({}, {}));
            auto test_buffer_mem_req = device.memory_requirements(test_buffer_info);
            auto test_image_mem_req = device.memory_requirements(test_image_info);
            auto pessimised_mem_req = daxa::MemoryRequirements{
                .size = std::max(test_buffer_mem_req.size, test_image_mem_req.size),
                .alignment = std::max(test_buffer_mem_req.alignment, test_image_mem_req.alignment),
                .memory_type_bits = test_buffer_mem_req.memory_type_bits & test_image_mem_req.memory_type_bits,
            };
            auto test_memory_block = device.create_memory({
                .requirements = pessimised_mem_req,
            });
            auto memory_block_test_buffer = device.create_buffer_from_memory_block({
                .buffer_info = test_buffer_info,
                .memory_block = test_memory_block,
            });
            auto memory_block_test_image = device.create_image_from_memory_block({
                .image_info = test_image_info,
                .memory_block = test_memory_block,
            });
            device.destroy_image(memory_block_test_image);
            device.destroy_buffer(memory_block_test_buffer);
        }
        catch (std::runtime_error error)
        {
            std::cout << "failed test \"sro_aliased_suballocation\": " << error.what() << std::endl;
            exit(-1);
        }
    }
    void acceleration_structure_creation(daxa::Instance & instance)
    {
        try
        {
            daxa::Device device;
            try
            {
                device = instance.create_device_2(instance.choose_device(daxa::ImplicitFeatureFlagBits::BASIC_RAY_TRACING, {}));
            }
            catch (std::runtime_error error)
            {
                std::cout << "Test skipped. No present device supports raytracing!" << std::endl;
                return;
            }

            auto test_tlas = device.create_tlas(test_tlas_info);
            auto test_blas = device.create_blas(test_blas_info);
            device.destroy_blas(test_blas);
            device.destroy_tlas(test_tlas);
        }
        catch (std::runtime_error error)
        {
            std::cout << "failed test \"acceleration_structure_creation\": " << error.what() << std::endl;
            exit(-1);
        }
    }
} // namespace tests

auto main() -> int
{
    auto instance = daxa::create_instance({});
    tests::simplest(instance);
    tests::device_selection(instance);
    tests::sro_creation(instance);
    tests::sro_aliased_suballocation(instance);
    tests::acceleration_structure_creation(instance);
    std::cout << "completed all tests successfully!" << std::endl;
}
