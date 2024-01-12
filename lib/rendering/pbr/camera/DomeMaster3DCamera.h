// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <moonray/rendering/texturing/sampler/TextureSampler.h>

#include <scene_rdl2/common/math/Mat4.h>
#include <scene_rdl2/common/math/Vec3.h>

#include <OpenImageIO/imageio.h>
#include <OpenImageIO/imagebuf.h>
#include <OpenImageIO/imagebufalgo.h>

#include "Camera.h"

namespace moonray {
namespace pbr {

class DomeMaster3DCamera : public Camera
{
public:
    /// Constructor
    explicit DomeMaster3DCamera(const scene_rdl2::rdl2::Camera* rdlCamera);

private:
    void initAttributeKeys(const scene_rdl2::rdl2::SceneClass& sceneClass);

    bool getIsDofEnabledImpl() const override;

    void updateImpl(const scene_rdl2::math::Mat4d& world2render) override;

    void createRayImpl(mcrt_common::RayDifferential* dstRay,
                       float x,
                       float y,
                       float time,
                       float lensU,
                       float lensV) const override;

    StereoView getStereoViewImpl() const override;

    inline scene_rdl2::math::Vec3f createDirection(const scene_rdl2::math::Vec3f& camOrigin, float x, float y) const;
    // utility helper function
    inline void computePhiAndTheta(float x,
                                   float y, 
                                   float& sinPhi,
                                   float& cosPhi,
                                   float& sinTheta,
                                   float& cosTheta) const;
    inline void flipXVector(scene_rdl2::math::Vec3f& vec) const;
    inline void flipYVector(scene_rdl2::math::Vec3f& vec) const;
    inline void applyParallax(scene_rdl2::math::Vec3f& vec) const;

    // Camera focal point in camera space (non-zero for stereo views L/R)
    float mInterocularOffset;
    float mImageResolutionWidthReciprocal;   // 1.0 / x pixel resolution (width)
    float mImageResolutionHeightReciprocal;  // 1.0 / y pixel resolution (height)

    static bool                            sAttributeKeyInitialized;

    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Int>   sStereoViewKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> sStereoInterocularDistanceKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> sParallaxDistanceKey;

    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> sFOVVerticalAngleKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> sFOVHorizontalAngleKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Bool>  sFlipRayXKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Bool>  sFlipRayYKey;

    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::String> sCameraSeparationMapFileNameKey;

    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> sHeadTiltMapKey;
    static scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Bool>  sZenithModeKey;

    StereoView mStereoView;
    float mFOVHorizontalAngleRadians;
    float mFOVVerticalAngleRadians;
    bool mZenithMode;
    bool mFlipRayX;
    bool mFlipRayY;
    float mParallaxDistance;
    std::string mInterocularDistanceFileName;

    // OIIO Functionality used to sample mInterocularDistance map distance
    texture::TextureHandle* mTextureHandle;
    OIIO::TextureSystem* mOIIOTextureSystem;
    OIIO::ImageSpec mImageSpec;
    OIIO::TextureOpt mTextureOption;
};

} // namespace pbr
} // namespace moonray


