// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <moonray/rendering/mcrt_common/Ray.h>
#include <moonray/rendering/pbr/core/PbrTLState.h>
#include <moonray/rendering/pbr/integrator/PathIntegrator.h>
#include <moonray/rendering/pbr/integrator/PathIntegratorUtil.h>
#include <scene_rdl2/common/math/Color.h>

namespace moonray {
namespace pbr {

void accumLightAovs(pbr::TLState* pbrTls, const BundledOcclRay& occlRay, const FrameState& fs,
                    int numItems, const scene_rdl2::math::Color& matchMultiplier,
                    const scene_rdl2::math::Color* nonMatchMultiplier, int flags);

void accumVisibilityAovs(pbr::TLState* pbrTls, const BundledOcclRay& occlRay,
                         const FrameState& fs, const int numItems, float value);

void CPP_accumVisibilityAovs(float value, const pbr::TLState& pbrTls, const FrameState& fs, const BsdfSampler& bSampler, 
                         const LightSetSampler& lSampler, const PathVertex& pv, const RayState& rs);

void accumVisibilityAovsHit(pbr::TLState* pbrTls, const BundledOcclRay& occlRay,
                            const FrameState& fs, const int numItems);

void accumVisibilityAovsOccluded(pbr::TLState* pbrTls, const BundledOcclRay& occlRay,
                                 const FrameState& fs, const int numItems);

void fillBundledRadiance(pbr::TLState* pbrTls, BundledRadiance* dst, const BundledOcclRay& occlRay);

// Volume transmittance along the occlusion ray. Call this function after doing the
// occlusion ray test. We only need to know the transmittance if the ray is not occluded.
scene_rdl2::math::Color getTransmittance(pbr::TLState* pbrTls, const BundledOcclRay& occlRay);

} // namespace pbr
} // namespace moonray

