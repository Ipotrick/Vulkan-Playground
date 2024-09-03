#pragma once

#include "daxa/daxa.inl"
#include "daxa/utils/task_graph.inl"

DAXA_DECL_TASK_HEAD_BEGIN(DrawTri)
DAXA_TH_IMAGE(COLOR_ATTACHMENT, REGULAR_2D, resolve_target)
DAXA_TH_IMAGE(COLOR_ATTACHMENT, REGULAR_2D, render_target)
DAXA_DECL_TASK_HEAD_END