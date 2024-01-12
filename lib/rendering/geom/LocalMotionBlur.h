// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#ifndef LOCALMOTIONBLUR_H
#define LOCALMOTIONBLUR_H

#include <moonray/rendering/bvh/shading/PrimitiveAttribute.h>
#include <moonray/rendering/bvh/shading/Xform.h>
#include <moonray/rendering/geom/Api.h>
#include <moonray/rendering/geom/ProceduralContext.h>

#include <scene_rdl2/common/math/Xform.h>
#include <scene_rdl2/common/math/Vec3.h>
#include <scene_rdl2/scene/rdl2/rdl2.h>

#include <vector>

#pragma once

namespace moonray {
namespace local_motion_blur {

// Attributes in the points file that will be used
// to contruct the regions
static moonray::shading::AttributeKeySet sLocalMotionBlurAttributes = {
    moonray::shading::TypedAttributeKey<float>("radius"),
    moonray::shading::TypedAttributeKey<float>("inner_radius"),
    moonray::shading::TypedAttributeKey<float>("multiplier")
};

struct MbRegion
{
    MbRegion(scene_rdl2::math::Xform3f xform,
             float radius,
             float innerRadius,
             float multiplier) :
        mXform(xform),
        mRadius(radius),
        mInnerRadius(innerRadius),
        mMultiplier(multiplier) {}

    scene_rdl2::math::Xform3f mXform;
    float mRadius;
    float mInnerRadius;
    float mMultiplier;
};

// This class creates spherical regions which locally modulate motion blur.
// The regions are defined by a list of transforms and parameters.
class LocalMotionBlur
{
public:
    LocalMotionBlur(const moonray::geom::GenerateContext& generateContext,
                    const std::vector<moonray::shading::XformSamples>& regionXforms,
                    const moonray::shading::PrimitiveAttributeTable& pointsAttributes,
                    const bool useLocalCameraMotionBlur=false,
                    const float strengthMult=1.0f,
                    const float radiusMult=1.0f);

    template <typename AttributeType> void
    apply(const scene_rdl2::rdl2::MotionBlurType mbType,
          const moonray::shading::XformSamples& parent2root,
          moonray::geom::VertexBuffer<AttributeType, moonray::geom::InterleavedTraits>& vertices,
          moonray::shading::PrimitiveAttributeTable& primitiveAttributeTable) const;

    virtual ~LocalMotionBlur() = default;

private:
    float getMultiplier(const scene_rdl2::math::Vec3f& P,
                        const moonray::shading::XformSamples& parent2root) const;

    std::vector<MbRegion> mRegions;
    const scene_rdl2::rdl2::Geometry* mRdlGeometry;
    moonray::shading::XformSamples mNodeXform;
    moonray::shading::XformSamples mCameraXform;
    float mShutterInterval;
    float mFps;
    bool mUseLocalCameraMotionBlur;
};


} // namespace local_motion_blur
} // namespace moonray 

#endif // LOCALMOTIONBLUR_H
