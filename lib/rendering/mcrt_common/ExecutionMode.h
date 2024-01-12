// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
#pragma once
#ifndef EXECUTIONMODE_H
#define EXECUTIONMODE_H

namespace moonray {
namespace mcrt_common {

enum ExecutionMode
{
    // Run vectorized unless there are scalar only features in the scene file.
    AUTO,

    // Run vectorized despite there being scalar only features in the scene file.
    VECTORIZED,

    // Run in scalar mode.
    SCALAR,

    // Run in gpu-assisted (XPU) mode
    XPU,

    NUM_MODES,
};

} // mcrt_common
} // namespace moonray

#endif // EXECUTIONMODE_H
