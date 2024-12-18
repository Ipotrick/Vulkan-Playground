#pragma once

#include <daxa/types.hpp>
#include <daxa/gpu_resources.hpp>

namespace daxa
{
    struct Device;

    struct MemoryBarrierInfo
    {
        Access src_access = AccessConsts::NONE;
        Access dst_access = AccessConsts::NONE;
    };

    [[nodiscard]] auto to_string(MemoryBarrierInfo const & info) -> std::string;

    struct ImageMemoryBarrierInfo
    {
        Access src_access = AccessConsts::NONE;
        Access dst_access = AccessConsts::NONE;
        ImageLayout src_layout = ImageLayout::UNDEFINED;
        ImageLayout dst_layout = ImageLayout::UNDEFINED;
        ImageMipArraySlice image_slice = {};
        ImageId image_id = {};
    };

    [[nodiscard]] DAXA_EXPORT_CXX auto to_string(ImageMemoryBarrierInfo const & info) -> std::string;

    struct BinarySemaphoreInfo
    {
        SmallString name = {};
    };

    struct DAXA_EXPORT_CXX BinarySemaphore final : ManagedPtr<BinarySemaphore, daxa_BinarySemaphore>
    {
        BinarySemaphore() = default;

        /// THREADSAFETY:
        /// * reference MUST NOT be read after the device is destroyed.
        /// @return reference to info of object.
        [[nodiscard]] auto info() const -> BinarySemaphoreInfo const &;

      protected:
        template <typename T, typename H_T>
        friend struct ManagedPtr;
        static auto inc_refcnt(ImplHandle const * object) -> u64;
        static auto dec_refcnt(ImplHandle const * object) -> u64;
    };

    struct TimelineSemaphoreInfo
    {
        u64 initial_value = {};
        SmallString name = {};
    };

    struct DAXA_EXPORT_CXX TimelineSemaphore final : ManagedPtr<TimelineSemaphore, daxa_TimelineSemaphore>
    {
        TimelineSemaphore() = default;

        /// THREADSAFETY:
        /// * reference MUST NOT be read after the object is destroyed.
        /// @return reference to info of object.
        [[nodiscard]] auto info() const -> TimelineSemaphoreInfo const &;

        [[nodiscard]] auto value() const -> u64;
        void set_value(u64 value);
        [[nodiscard]] auto wait_for_value(u64 value, u64 timeout_nanos = ~0ull) -> bool;

      protected:
        template <typename T, typename H_T>
        friend struct ManagedPtr;
        static auto inc_refcnt(ImplHandle const * object) -> u64;
        static auto dec_refcnt(ImplHandle const * object) -> u64;
    };

    struct EventInfo
    {
        SmallString name = {};
    };

    struct DAXA_EXPORT_CXX Event final : ManagedPtr<Event, daxa_Event>
    {
        Event() = default;

        /// THREADSAFETY:
        /// * reference MUST NOT be read after the object is destroyed.
        /// @return reference to info of object.
        [[nodiscard]] auto info() const -> EventInfo const &;

      protected:
        template <typename T, typename H_T>
        friend struct ManagedPtr;
        static auto inc_refcnt(ImplHandle const * object) -> u64;
        static auto dec_refcnt(ImplHandle const * object) -> u64;
    };

    struct EventSignalInfo
    {
        daxa::Span<MemoryBarrierInfo const> memory_barriers = {};
        daxa::Span<ImageMemoryBarrierInfo const> image_barriers = {};
        Event & event;
    };

    using EventWaitInfo = EventSignalInfo;
} // namespace daxa