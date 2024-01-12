// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <moonray/rendering/geom/prim/Primitive.h>

#include <moonray/rendering/rt/rt.h>
#include <moonray/rendering/geom/prim/BVHUserData.h>
#include <moonray/rendering/mcrt_common/Ray.h>

#include <scene_rdl2/common/platform/Platform.h>
#include <scene_rdl2/scene/rdl2/SceneContext.h>
#include <scene_rdl2/scene/rdl2/Layer.h>

#include <embree4/rtcore.h>

namespace moonray {
namespace rt {

typedef std::vector<std::unique_ptr<geom::internal::BVHUserData>> BVHUserDataList;

class EmbreeAccelerator
{
public:
    explicit EmbreeAccelerator(const AcceleratorOptions& options);
    ~EmbreeAccelerator();
    /// Copy is disabled
    EmbreeAccelerator(const EmbreeAccelerator& other) = delete;
    const EmbreeAccelerator &operator=(const EmbreeAccelerator& other) = delete;

    void build(OptimizationTarget accelMode, ChangeFlag changeFlag,
            const scene_rdl2::rdl2::Layer *layer,
            const scene_rdl2::rdl2::SceneContext::GeometrySetVector& geometrySets,
            const scene_rdl2::rdl2::Layer::GeometryToRootShadersMap* g2s = nullptr);

    void intersect(mcrt_common::Ray& ray) const;

    bool occluded(mcrt_common::Ray& ray) const;

    scene_rdl2::math::BBox3f getBounds() const;

    size_t getMemory() const {
        return mBVHMemory;
    }

    finline const RTCDevice& getRTCDevice(void) const {
        return mDevice;
    }

    void addMemoryUsage(const ssize_t bytes)
    {
        mBVHMemory += bytes;
    }

    //------------------------------

    double mBvhBuildProceduralTime;
    double mRtcCommitTime;

private:
    /// An Embree scene that contains all geometry and instances
    RTCScene mRootScene;
    RTCDevice mDevice;
    // container for userdata so that they can be safely deleted.
    BVHUserDataList mBVHUserData;
    std::atomic<ssize_t> mBVHMemory;
};

} // namespace rt
} // namespace moonray



