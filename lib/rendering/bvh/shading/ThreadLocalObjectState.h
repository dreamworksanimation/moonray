// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file ThreadLocalObjectState.h
///

#pragma once
#include <moonray/common/mcrt_util/Average.h>
#include <scene_rdl2/render/logging/logging.h>

namespace moonray {
namespace shading {

///
/// State that is local to both the thread and the object. There
/// is one of these per-thread per-rdl object. Writing object-specific
/// data to a ThreadLocalObjectState is always thread-safe. 
///
/// A ThreadLocalState object (see below) contains a vector of these.
///

// This is cache-line aligned so as to avoid false sharing when multiple
// threads write to adjacent ThreadLocalObjectStates in RDL objects

class CACHE_ALIGN ThreadLocalObjectState
{
 public:
    static ThreadLocalObjectState *alignedAlloc(int num)
    {
        void *memptr;
        if (auto err = posix_memalign(&memptr, 64, sizeof(ThreadLocalObjectState)*num)) {
            throw std::bad_alloc();
        }
        ThreadLocalObjectState *result = static_cast<ThreadLocalObjectState *>(memptr);
        {
            for (int i = 0; i < num; i++) {
                new (result + i) ThreadLocalObjectState;
            }
        }
        return result;
    }

    static void deallocate(int num, ThreadLocalObjectState *tlos)
    {
        if (tlos == nullptr) return;
        for (int i = 0; i < num; i++) {
            tlos[i].~ThreadLocalObjectState();
        }
        free(tlos);
    }

    moonray::util::InclusiveExclusiveAverage<int64> mShaderCallStat;

    void clear()
    {
        mShaderCallStat.reset();
    }
};

} // namespace shading 
} // namespace moonray

