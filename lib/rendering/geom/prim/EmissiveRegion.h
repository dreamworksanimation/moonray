// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

///
/// @file EmissiveRegion.h
///

#pragma once

#include <moonray/rendering/geom/prim/EmissionDistribution.h>
#include <moonray/rendering/geom/prim/GeomTLState.h>
#include <moonray/rendering/geom/prim/Primitive.h>

#include <scene_rdl2/scene/rdl2/rdl2.h>
#include <scene_rdl2/scene/rdl2/VisibilityFlags.h>

namespace moonray {
namespace geom {
namespace internal {

// EmissiveRegion is a primitive with emissive volume assignment
// The integrator will integrate the emission contribution from EmissiveRegion
// through importance sampling like regular light source so that we can
// use emissive volume to illuminate the scene with better converge rate
class EmissiveRegion
{
public:
    // This constructor is used by emissive regions that do
    // not point to instanced volume primitives.  So it can just use
    // the transform from the emission distribution unmodified.
    EmissiveRegion(const Primitive* primitive,
            const scene_rdl2::rdl2::VolumeShader* volumeShader,
            int volumeId, int visibilityMask):
        mVolumeId(volumeId),
        mVisibilityMask(visibilityMask),
        mEmissionDistribution(primitive->computeEmissionDistribution(volumeShader)),
        mTransform(mEmissionDistribution->getTransform())
    {}
    // This constructor is used by emissive regions that point
    // to shared volume primitives.  It needs to use a distToRender transform
    // that is different than the one stored in the emission distribution.
    EmissiveRegion(int volumeId, int visibilityMask,
            std::shared_ptr<EmissionDistribution> emissionDistribution,
            const EmissionDistribution::Transform &transform):
        mVolumeId(volumeId),
        mVisibilityMask(visibilityMask),
        mEmissionDistribution(emissionDistribution),
        mTransform(transform)
    {
    }

    int size() const
    {
        return mEmissionDistribution->count();
    }

    void sample(const Vec3f& p, float u1, float u2, float u3,
            Vec3f& wi, float& pdfWi, float& tEnd, float time) const
    {
        mEmissionDistribution->sample(mTransform, p, u1, u2, u3, wi, pdfWi, tEnd, time);
    }

    float pdf(const Vec3f& p, const Vec3f& wi, float& tEnd, float time) const
    {
        return mEmissionDistribution->pdf(mTransform, p, wi, tEnd, time);
    }

    Vec3f getEquiAngularPivot(float u1, float u2, float u3, float time) const
    {
        return mEmissionDistribution->sample(mTransform, u1, u2, u3, time);
    }

    bool canIlluminateSSS() const
    {
        // treat bssrdf as a special case of diffuse reflection
        return (mVisibilityMask & scene_rdl2::rdl2::DIFFUSE_REFLECTION) != 0;
    }

    bool canIlluminateVolume() const
    {
        // PHASE_REFLECTION and PHASE_TRANSMISSION always on and off in pair
        // so we don't need to verify both
        return (mVisibilityMask & scene_rdl2::rdl2::PHASE_REFLECTION) != 0;
    }

    int mVolumeId;
    int mVisibilityMask;
    std::shared_ptr<EmissionDistribution> mEmissionDistribution;
    EmissionDistribution::Transform mTransform;
};

} // namespace internal
} // namespace geom
} // namespace moonray

