#include <daxa/daxa.hpp>

struct App
{
    daxa::Instance daxa_ctx = daxa::create_instance({});
    daxa::Device device = daxa_ctx.create_device_2(daxa_ctx.choose_device({}, {}));
};

namespace tests
{
    using namespace daxa::types;

    void binary_semaphore(App & app)
    {
        auto recorder = app.device.create_command_recorder({});
        auto commands1 = recorder.complete_current_commands();
        auto commands2 = recorder.complete_current_commands();
        auto commands3 = recorder.complete_current_commands();
        auto commands4 = recorder.complete_current_commands();
        recorder.~CommandRecorder();

        auto binary_semaphore_1 = app.device.create_binary_semaphore({});
        auto binary_semaphore_2 = app.device.create_binary_semaphore({});

        // This semaphore is useful in the future, it can be used to make submits wait on each other,
        // or to make a present wait for a submit to finish
        app.device.submit_commands({
            .command_lists = std::array{commands1},
            .signal_binary_semaphores = std::array{binary_semaphore_1},
        });

        app.device.submit_commands({
            .command_lists = std::array{commands2},
            .wait_binary_semaphores = std::array{binary_semaphore_1},
            .signal_binary_semaphores = std::array{binary_semaphore_2},
        });

        // Binary semaphores can be reused ONLY after they have been signaled.
        app.device.submit_commands({
            .command_lists = std::array{commands3},
            .wait_binary_semaphores = std::array{binary_semaphore_2},
            .signal_binary_semaphores = std::array{binary_semaphore_1},
        });

        app.device.submit_commands({
            .command_lists = std::array{commands4},
            .wait_binary_semaphores = std::array{binary_semaphore_1},
        });
    }

    void memory_barriers(App & app)
    {
        auto recorder = app.device.create_command_recorder({});
        auto commands = recorder.complete_current_commands();

        app.device.submit_commands({
            .command_lists = std::array{commands},
        });
    }
} // namespace tests

auto main() -> int
{
    App app = {};
    tests::binary_semaphore(app);
    // Useless for now
    tests::memory_barriers(app);
}
