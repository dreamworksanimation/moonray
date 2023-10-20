// Copyright 2023 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//
#pragma once
#include <moonray/rendering/pbr/Types.h>
#include <moonray/rendering/mcrt_common/Bundle.h>
#include <moonray/rendering/shading/Types.h>

#include <scene_rdl2/common/math/BBox.h>
#include <scene_rdl2/common/math/Vec3.h>
#include <scene_rdl2/common/math/Vec4.h>

// Turned off by default since it turns out not to be a net win overall.
// Intersection is generally a few % faster, but the gains are lost by the cost
// of sorting.
//#define RAY_SORTING

namespace moonray {
namespace pbr {

class RayState;

#pragma warning(push)
#pragma warning(disable:444) // destructor for base class isn't virtual

#ifdef RAY_SORTING
typedef mcrt_common::LocalQueue<WrappedRayState, true, offsetof(shading::SortedRayState, mSortKey)> BaseIncoherentRayQueue;
#else
typedef mcrt_common::LocalQueue<WrappedRayState, false, offsetof(shading::SortedRayState, mSortKey)> BaseIncoherentRayQueue;
#endif

class IncoherentRayQueue : public BaseIncoherentRayQueue
{
public:
    IncoherentRayQueue();
    ~IncoherentRayQueue();

    void init(unsigned queueSize);

    void addEntries(mcrt_common::ThreadLocalState *tls,
                    unsigned numEntries, RayState **entries,
                    scene_rdl2::alloc::Arena *arena);

    // Setting this can give more accurate positional sorting.
    static void setBounds(const scene_rdl2::math::BBox3f &bbox);

private:
    EntryType * mEntries;

    // Used to locate which voxel a position is located in. Shared between all threads.
    static scene_rdl2::math::Vec3f sPositionScale;
    static scene_rdl2::math::Vec3f sPositionOffset;
};

#pragma warning(pop)

} // namespace pbr
} // namespace moonray

