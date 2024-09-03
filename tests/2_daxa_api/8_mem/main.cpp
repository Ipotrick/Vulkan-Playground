#include <daxa/daxa.hpp>
using namespace daxa::types;

#include <daxa/utils/mem.hpp>

#include <iostream>

static inline constexpr usize ITERATION_COUNT = {1000};
static inline constexpr usize ELEMENT_COUNT = {17};

auto main() -> int
{
    daxa::Instance daxa_ctx = daxa::create_instance({});
    daxa::Device device = daxa_ctx.create_device_2(daxa_ctx.choose_device({},{}));
    daxa::TransferMemoryPool tmem{daxa::TransferMemoryPoolInfo{
        .device = device,
        .capacity = 256,
        .name = "transient memory pool",
    }};
    daxa::TimelineSemaphore gpu_timeline = device.create_timeline_semaphore({
        .name = "timeline semaphpore",
    });
    usize global_submit_timeline = 1;
    daxa::BufferId result_buffer = device.create_buffer({
        .size = sizeof(u32) * ELEMENT_COUNT * ITERATION_COUNT,
        .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
        .name = "result",
    });

    for (u32 iteration = 0; iteration < ITERATION_COUNT; ++iteration)
    {
        [[maybe_unused]] auto _timeout = gpu_timeline.wait_for_value(global_submit_timeline - 1);
        daxa::CommandRecorder cmd = device.create_command_recorder({});
        cmd.pipeline_barrier({
            .src_access = daxa::AccessConsts::TRANSFER_READ_WRITE | daxa::AccessConsts::HOST_WRITE,
            .dst_access = daxa::AccessConsts::TRANSFER_READ_WRITE,
        });

        // Can allocate anywhere in the frame with imediately available staging memory.
        daxa::TransferMemoryPool::Allocation alloc = tmem.allocate(ELEMENT_COUNT, 8).value();
        for (u32 i = 0; i < ELEMENT_COUNT; ++i)
        {
            // The Allocation provides a host pointer to the memory.
            reinterpret_cast<u32 *>(alloc.host_address)[i] = iteration * 100 + i;
        }
        // ALl the allocations are from a single internal buffer.
        // The allocation contains an integer offset into that buffer.
        // It also contains a device address that can be passed to a shader directly.
        cmd.copy_buffer_to_buffer({
            .src_buffer = tmem.buffer(),
            .dst_buffer = result_buffer,
            .src_offset = alloc.buffer_offset,
            .dst_offset = sizeof(u32) * ELEMENT_COUNT * iteration,
            .size = sizeof(u32) * ELEMENT_COUNT,
        });

        // Need to specify give every subnmit using the mme util timeline semaphore and its value on submission.
        // This is nessecary for internal tracking.

        auto signals = std::array{
            std::pair{gpu_timeline, global_submit_timeline},
            std::pair{tmem.timeline_semaphore(), tmem.timeline_value()},
        };
        device.submit_commands({
            .command_lists = std::array{cmd.complete_current_commands()},
            .signal_timeline_semaphores = signals,
        });
        global_submit_timeline += 1;
    }

    daxa::CommandRecorder cmd = device.create_command_recorder({});
    cmd.pipeline_barrier({
        .src_access = daxa::AccessConsts::TRANSFER_WRITE,
        .dst_access = daxa::AccessConsts::HOST_READ,
    });

    device.submit_commands({
        .command_lists = std::array{cmd.complete_current_commands()},
    });
    cmd.~CommandRecorder();

    device.wait_idle();

    u32 const * elements = device.buffer_host_address_as<u32>(result_buffer).value();
    for (u32 iteration = 0; iteration < ITERATION_COUNT; ++iteration)
    {
        for (u32 element = 0; element < ELEMENT_COUNT; ++element)
        {
            std::cout << "value: " << elements[iteration * ELEMENT_COUNT + element] / 100 << " " << elements[iteration * ELEMENT_COUNT + element] % 100 << "\n";
        }
    }
    device.destroy_buffer(result_buffer);
    device.collect_garbage();
    std::cout << std::flush;
}
