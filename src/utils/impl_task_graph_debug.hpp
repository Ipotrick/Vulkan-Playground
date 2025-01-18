#pragma once

#include "impl_task_graph.hpp"
#include <format>
namespace daxa
{
    enum struct Level
    {
        BASIC,
        DETAILED,
        VERBOSE,
        MAX_ENUM = 0x7fffffff,
    };

    void print_separator_to(std::string & out, std::string & indent)
    {
        char const last = indent.back();
        indent.back() = '#';
        std::format_to(std::back_inserter(out), "{}--------------------------------\n", indent);
        indent.back() = last;
    }

    void begin_indent(std::string & out, std::string & indent, bool contained = false)
    {
        indent.push_back(' ');
        indent.push_back(' ');
        indent.push_back(' ');
        indent.push_back(contained ? '|' : ' ');
        if (contained)
        {
            print_separator_to(out, indent);
        }
    }

    void end_indent(std::string &, std::string & indent)
    {
        indent.pop_back();
        indent.pop_back();
        indent.pop_back();
        indent.pop_back();
    }

    struct FormatIndent
    {
        std::string & out;
        std::string & indent;
        FormatIndent(std::string & a_out, std::string & a_indent, bool contained = false) : out{a_out}, indent{a_indent}
        {
            begin_indent(out, indent, contained);
        }
        ~FormatIndent()
        {
            end_indent(out, indent);
        }
    };

    void validate_not_compiled(ImplTaskGraph & impl)
    {
        DAXA_DBG_ASSERT_TRUE_M(!impl.compiled, "Completed task graphs can not record new tasks");
    }

    template <typename BufferBlasTlasAttachmentT>
    void validate_buffer_blas_tlas_task_view(ITask const & task, u32 attach_index, BufferBlasTlasAttachmentT const & attach)
    {
        DAXA_DBG_ASSERT_TRUE_M(
            !attach.view.is_empty(),
            std::format("Detected unassigned task buffer view for attachment \"{}\" (index: {}, access: {}) in task \"{}\"\n",
                        attach.name, attach_index, to_string(attach.access), task.name()));
    }

    void validate_image_task_view(ITask const & task, u32 attach_index, TaskImageAttachmentInfo const & attach)
    {
        DAXA_DBG_ASSERT_TRUE_M(
            !attach.view.is_empty(),
            std::format("Detected unassigned task image view for attachment \"{}\" (index: {}, access: {}) in task \"{}\"\n",
                        attach.name, attach_index, to_string(attach.access), task.name()));
    }

    void validate_overlapping_attachment_views(ImplTaskGraph const & impl, ITask const * task)
    {
#if DAXA_VALIDATION
        for_each(
            task->attachments(),
            [&](u32 index_a, auto const & a)
            {
                if (a.view.is_null())
                    return;
                for_each(
                    task->attachments(),
                    [&](u32 index_b, auto const & b)
                    {
                        if (b.view.is_null())
                            return;
                        if (index_a == index_b)
                            return;
                        [[maybe_unused]] bool const overlapping = a.view == b.view;
                        DAXA_DBG_ASSERT_TRUE_M(
                            !overlapping,
                            std::format(
                                "Detected overlapping attachment buffer views;\n"
                                "Attachments \"{}\" and \"{}\" both refer to the same task buffer \"{}\" in task \"{}\";\n"
                                "All buffer attachments must refer to different buffers within each task!",
                                a.name, b.name,
                                impl.global_buffer_infos[a.view.index].get_name(),
                                task->name()));
                    },
                    [&](u32, TaskImageAttachmentInfo const &) {});
            },
            [&](u32 index_a, TaskImageAttachmentInfo const & a)
            {
                if (a.view.is_null())
                    return;
                for_each(
                    task->attachments(),
                    [&](u32, auto const &) {},
                    [&](u32 index_b, TaskImageAttachmentInfo const & b)
                    {
                        if (b.view.is_null())
                            return;
                        if (index_a == index_b)
                            return;
                        [[maybe_unused]] auto const intersect = a.view == b.view && a.view.slice.intersects(b.view.slice);
                        DAXA_DBG_ASSERT_TRUE_M(
                            !intersect,
                            std::format(
                                "Detected overlapping attachment image views.\n"
                                "Attachments \"{}\" and \"{}\" refer overlapping slices ({} and {}) to the same task image \"{}\" in task \"{}\";"
                                "All task image attachment views and their slices must refer to disjoint parts of images within each task!",
                                a.name, b.name, to_string(a.view.slice), to_string(b.view.slice),
                                impl.global_image_infos.at(b.view.index).get_name(),
                                task->name()));
                    });
            });
#endif
    }

    template <typename BufferBlasTlasT>
    void validate_task_buffer_blas_tlas_runtime_data(ImplTask & task, BufferBlasTlasT const & attach)
    {
#if DAXA_VALIDATION
        if constexpr (std::is_same_v<BufferBlasTlasT, TaskBufferAttachmentInfo>)
        {
            DAXA_DBG_ASSERT_TRUE_M(
                attach.ids.size() >= attach.shader_array_size,
                std::format("Detected invalid runtime buffer count.\n"
                            "Attachment \"{}\" in task \"{}\" requires {} runtime buffer(s), but only {} runtime buffer(s) are present when executing task.\n"
                            "Attachment runtime buffers must be at least as many as its shader array size!",
                            attach.name, task.base_task->name(), attach.shader_array_size, attach.ids.size()));
        }
#endif
    }

    void validate_task_image_runtime_data(ImplTask & task, TaskImageAttachmentInfo const & attach)
    {
#if DAXA_VALIDATION
        if (attach.shader_array_type == TaskHeadImageArrayType::MIP_LEVELS)
        {
            DAXA_DBG_ASSERT_TRUE_M(
                attach.ids.size() >= 1,
                std::format("Detected invalid runtime image count.\n"
                            "Attachment \"{}\" in task \"{}\" requires at least 1 runtime image, but no runtime images are present when executing task.\n"
                            "Attachment runntime image count must be at least one for mip-array attachments!",
                            attach.name, task.base_task->name(), attach.shader_array_size, attach.ids.size()));
        }
        else // arg.shader_array_type == TaskHeadImageArrayType::RUNTIME_ARRAY
        {
            DAXA_DBG_ASSERT_TRUE_M(
                attach.ids.size() >= attach.shader_array_size,
                std::format("Detected invalid runtime image count.\n"
                            "Attachment \"{}\" in task \"{}\" requires at least {} runtime image(s), but only {} runtime images are present when executing task.\n"
                            "Attachment runntime image count must be at least the shader array size for array attachments!",
                            attach.name, task.base_task->name(), attach.shader_array_size, attach.ids.size()));
        }
#endif
    }
    // void validate_
} // namespace daxa