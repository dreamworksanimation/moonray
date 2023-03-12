// Copyright 2023 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//
#pragma once

namespace moonray {
namespace engine_tool {

enum class FrameStatus : int {
    STARTED = 0,
    RENDERING,
    FINISHED,
    CANCELLED,
    ERROR
};

} // namespace engine_tool
} // namespace moonray

