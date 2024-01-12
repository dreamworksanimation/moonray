// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
#pragma once

#include <scene_rdl2/common/platform/Platform.h>
#include <tbb/parallel_for.h>

namespace moonray {
namespace rndr {

// Mimics "for (unsigned i = start; i != end; ++i)" behavior.
// Calling this version allows the caller to decide at runtime whether they
// want it parallelized or not.
template <typename T, typename FUNC>
finline void
simpleLoop(bool parallel, T start, T end, const FUNC &func)
{
    if (parallel) {
        tbb::parallel_for (start, end, [&](unsigned iter) {
            func(iter);
        });
    } else {
        for (unsigned iter = start; iter != end; ++iter) {
            func(iter);
        }
    }
}

// This is a simillar idea of simpleLoop but parallel option is solved at compile time.
template <bool parallel, typename T, typename FUNC>
finline void
simpleLoop2(T start, T end, const FUNC &func)
{
    if (parallel) {
        tbb::parallel_for (start, end, [&](unsigned iter) {
            func(iter);
        });
    } else {
        for (unsigned iter = start; iter != end; ++iter) {
            func(iter);
        }
    }
}


} // namespace rndr
} // namespace moonray


