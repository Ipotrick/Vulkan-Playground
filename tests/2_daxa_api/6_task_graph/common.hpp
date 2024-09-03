#pragma once

#include <0_common/window.hpp>
#include <iostream>
#include <thread>

#include <daxa/utils/pipeline_manager.hpp>
#include <daxa/utils/task_graph.hpp>

#define APPNAME "Daxa API Sample TaskGraph"
#define APPNAME_PREFIX(x) ("[" APPNAME "] " x)

using namespace daxa::types;

#include "shaders/shared.inl"

#include <daxa/utils/imgui.hpp>
#include <imgui_impl_glfw.h>

struct AppContext
{
    daxa::Instance daxa_ctx = daxa::create_instance({});
    daxa::Device device = daxa_ctx.create_device_2(daxa_ctx.choose_device({}, {}));
};
