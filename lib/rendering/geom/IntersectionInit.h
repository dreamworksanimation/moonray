// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <scene_rdl2/common/math/Vec2.h>
#include <scene_rdl2/common/math/Vec3.h>
#include <scene_rdl2/scene/rdl2/Layer.h>

namespace moonray {

namespace shading { class Intersection; }

namespace mcrt_common {
    class ThreadLocalState;
    class Ray;
}

namespace geom {

/// Initialize the geometry properties of an intersection.
/// The ray and resulting intersection are in render space.
void initIntersectionPhase1(shading::Intersection &isect,
                            mcrt_common::ThreadLocalState *tls,
                            const mcrt_common::Ray         &ray,
                            const scene_rdl2::rdl2::Layer *pRdlLayer);


/// Initialize the type of path we are on for an intersection
void initIntersectionPhase2(shading::Intersection &isect,
                            mcrt_common::ThreadLocalState *tls,
                            int mirrorDepth,
                            int glossyDepth,
                            int diffuseDepth,
                            bool isSubsurfaceAllowed,
                            const scene_rdl2::math::Vec2f &minRoughness,
                            const scene_rdl2::math::Vec3f &wo);

/// Fully initialize an intersection. The ray and resulting
/// intersection are in render space.
void initIntersectionFull(shading::Intersection &isect,
                          mcrt_common::ThreadLocalState *tls,
                          const mcrt_common::Ray         &ray,
                          const scene_rdl2::rdl2::Layer *pRdlLayer,
                          int mirrorDepth,
                          int glossyDepth,
                          int diffuseDepth,
                          bool isSubsurfaceAllowed,
                          const scene_rdl2::math::Vec2f &minRoughness,
                          const scene_rdl2::math::Vec3f &wo);

} // namespace geom
} // namespace moonray

