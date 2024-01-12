// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "DomeMaster3DCamera.h"

#include <moonray/common/mcrt_macros/moonray_static_check.h>

#include <scene_rdl2/scene/rdl2/rdl2.h>
#include <scene_rdl2/common/math/MathUtil.h>

namespace moonray {
namespace pbr {

    using namespace scene_rdl2::math;

    bool                            DomeMaster3DCamera::sAttributeKeyInitialized = false;
    scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Int>   DomeMaster3DCamera::sStereoViewKey;
    scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> DomeMaster3DCamera::sStereoInterocularDistanceKey;
    scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> DomeMaster3DCamera::sParallaxDistanceKey;

    scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> DomeMaster3DCamera::sFOVVerticalAngleKey;
    scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> DomeMaster3DCamera::sFOVHorizontalAngleKey;
    scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Bool>  DomeMaster3DCamera::sFlipRayXKey;
    scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Bool>  DomeMaster3DCamera::sFlipRayYKey;

    scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::String> DomeMaster3DCamera::sCameraSeparationMapFileNameKey;
    scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Float> DomeMaster3DCamera::sHeadTiltMapKey;
    scene_rdl2::rdl2::AttributeKey<scene_rdl2::rdl2::Bool>  DomeMaster3DCamera::sZenithModeKey;

    namespace {
        // Compute signed interocular offset due to stereo (translation in camera
        // space along x axis).
        // Warning: This function doesn't work in CENTER mode.
        finline float
        computeInterocularOffset(StereoView stereoView,
                                 float interocularDistance)
        {
            MNRY_ASSERT(stereoView != StereoView::CENTER);

            float interocularOffset =
                (stereoView == StereoView::LEFT ? -0.5f : 0.5f) * interocularDistance;

            return interocularOffset;
        }
    } // namespace

    DomeMaster3DCamera::DomeMaster3DCamera(const scene_rdl2::rdl2::Camera* rdlCamera) :
        Camera(rdlCamera), mInterocularOffset(6.5), mImageResolutionWidthReciprocal(1.0f / 1024.0f),
        mImageResolutionHeightReciprocal(1.0f / 1024.0f), mStereoView( StereoView::CENTER ),
        mFOVHorizontalAngleRadians( M_PI * 2.0f ),
        mFOVVerticalAngleRadians( M_PI ), 
        mZenithMode( false ),
        mFlipRayX( false ),
        mFlipRayY( false ),
        mParallaxDistance( 360.0 ),
        mInterocularDistanceFileName(""),
        mTextureHandle( nullptr ),
        mOIIOTextureSystem( nullptr )
    {
        initAttributeKeys(rdlCamera->getSceneClass());
    }

    void DomeMaster3DCamera::initAttributeKeys(const scene_rdl2::rdl2::SceneClass& sceneClass)
    {
        if (sAttributeKeyInitialized) {
            return;
        }

        MOONRAY_START_NON_THREADSAFE_STATIC_WRITE

        sAttributeKeyInitialized = true;

        sFOVVerticalAngleKey = sceneClass.getAttributeKey<scene_rdl2::rdl2::Float>("FOV_vertical_angle");
        sFOVHorizontalAngleKey = sceneClass.getAttributeKey<scene_rdl2::rdl2::Float>("FOV_horizontal_angle");
        sFlipRayXKey = sceneClass.getAttributeKey<scene_rdl2::rdl2::Bool>("flip_ray_x");
        sFlipRayYKey = sceneClass.getAttributeKey<scene_rdl2::rdl2::Bool>("flip_ray_y");

        sStereoViewKey = sceneClass.getAttributeKey<scene_rdl2::rdl2::Int>("stereo_view");
        sParallaxDistanceKey = sceneClass.getAttributeKey<scene_rdl2::rdl2::Float>("stereo_convergence_distance");
        sStereoInterocularDistanceKey = sceneClass.getAttributeKey<scene_rdl2::rdl2::Float>(
            "stereo_interocular_distance");
        sCameraSeparationMapFileNameKey = sceneClass.getAttributeKey<scene_rdl2::rdl2::String>(
            "interocular_distance_map_file_name");

        sHeadTiltMapKey = sceneClass.getAttributeKey<scene_rdl2::rdl2::Float>("head_tilt_map");
        sZenithModeKey = sceneClass.getAttributeKey<scene_rdl2::rdl2::Bool>("zenith_mode");

        MOONRAY_FINISH_NON_THREADSAFE_STATIC_WRITE
    }

    bool DomeMaster3DCamera::getIsDofEnabledImpl() const
    {
        return false;
    }

    void DomeMaster3DCamera::updateImpl(const Mat4d& world2render)
    {
        mStereoView = getStereoViewImpl();
        mInterocularOffset = (mStereoView == StereoView::CENTER) ? 0 :
            computeInterocularOffset(mStereoView, getRdlCamera()->get(sStereoInterocularDistanceKey));

        mImageResolutionWidthReciprocal = 1.0f / getApertureWindowWidth();
        mImageResolutionHeightReciprocal = 1.0f / getApertureWindowHeight();

        // Convert to Radians to use w/ trig functions...
        mFOVHorizontalAngleRadians = scene_rdl2::math::degreesToRadians(getRdlCamera()->get(sFOVHorizontalAngleKey));
        mFOVVerticalAngleRadians = scene_rdl2::math::degreesToRadians(getRdlCamera()->get(sFOVVerticalAngleKey));

        mZenithMode = getRdlCamera()->get( sZenithModeKey );
        mFlipRayX = getRdlCamera()->get( sFlipRayXKey );
        mFlipRayY = getRdlCamera()->get( sFlipRayYKey );

        mParallaxDistance = getRdlCamera()->get( sParallaxDistanceKey );
        mInterocularDistanceFileName = getRdlCamera()->get(sCameraSeparationMapFileNameKey);

        texture::TextureSampler* textureSampler = texture::getTextureSampler();
        mTextureOption.swrap = OIIO::TextureOpt::Wrap::WrapClamp;
        mTextureOption.twrap = OIIO::TextureOpt::Wrap::WrapClamp;
        mTextureOption.interpmode = OIIO::TextureOpt::InterpMode::InterpBilinear;
        mOIIOTextureSystem = textureSampler->getTextureSystem();
        std::string errorString;
        if (mInterocularDistanceFileName != "") {
            mTextureHandle =
                textureSampler->getHandle(mInterocularDistanceFileName,
                                          errorString,
                                          mOIIOTextureSystem->get_perthread_info() );
            if (mTextureHandle == nullptr) {
                getRdlCamera()->error("FATAL: DomeMaster3DCamera failed to open texture file \"" ,
                                      mInterocularDistanceFileName, "\" (" , errorString , 
                                      ") at line: %d", __LINE__);
            }
        }
    }

    void 
    DomeMaster3DCamera::computePhiAndTheta(float x, 
                                           float y, 
                                           float& sinPhi,
                                           float& cosPhi,
                                           float& sinTheta,
                                           float& cosTheta) const {
        // remap pixel coordinates to new domain: [-1,-1] and [1,1]...
        x = 2.0 * x * mImageResolutionWidthReciprocal - 1.0;
        y = 2.0 * y * mImageResolutionHeightReciprocal - 1.0;

        double phi, theta;
        // convert to spherical coordinates...(bounded by Horizontal & Vertical FOV)
        phi = x * (mFOVHorizontalAngleRadians / 2.0);
        if (mZenithMode) {
            theta = M_PI_2 - y * (mFOVVerticalAngleRadians / 2.0);
        } else {
            theta = y * (mFOVVerticalAngleRadians / 2.0);
        }
        sincos(phi, &sinPhi, &cosPhi);
        sincos(theta, &sinTheta, &cosTheta);
    }

    void 
    DomeMaster3DCamera::flipXVector(Vec3f& vec) const {
        if (mFlipRayX) vec.x = - vec.x;
    }

    void 
    DomeMaster3DCamera::flipYVector(Vec3f& vec) const {
        if (mFlipRayY) {
            if (mZenithMode)
                vec.z = -vec.z;
            else
                vec.y = - vec.y;
        }
    }

    void 
    DomeMaster3DCamera::applyParallax(Vec3f& vec) const {
        if ((mStereoView != StereoView::CENTER) && (mParallaxDistance > 0.0)) {
            vec *= mParallaxDistance;
        }
    }

    void DomeMaster3DCamera::createRayImpl(mcrt_common::RayDifferential* dstRay,
                                           float x,
                                           float y,
                                           float time,
                                           float /*lensU*/,
                                           float /*lensV*/) const
    {
        // Compute transforms
        Mat4f ct2render;    // "camera space at ray time" --> render space
        if (getMotionBlur()) {
            ct2render = computeCamera2Render(time);
        } else {
            time = 0.0f;
            ct2render = getCamera2Render();
        }

        float sinPhi, cosPhi, sinTheta, cosTheta, tmp;
        computePhiAndTheta(x, y, sinPhi, cosPhi, sinTheta, cosTheta);

        float interocularMapDistanceColor[3] = { 1.0, 1.0, 1.0 };
        if (mTextureHandle != nullptr) {
            mOIIOTextureSystem->texture (mTextureHandle, 
                                         mOIIOTextureSystem->get_perthread_info(),
                                         const_cast<OIIO::TextureOpt&>(mTextureOption),
                                         ( x * mImageResolutionWidthReciprocal ),
                                         ( y * mImageResolutionHeightReciprocal ),
                                         mImageResolutionWidthReciprocal, 0.0, 
                                         0.0, mImageResolutionHeightReciprocal,
                                         mImageSpec.nchannels, interocularMapDistanceColor );
        }
        Vec3f eyeOrigin( mInterocularOffset * interocularMapDistanceColor[0], 0.0, 0.0 );
        if (mStereoView != StereoView::CENTER) {
            // This section causes the left & right eyes to "rotate" about the up-axis
            // Eg. As we look 90 degrees to left, the left eye comes to the origin in X
            // and moves backward in Z by .5 the interocular distance...
            if (mZenithMode) {
                tmp = eyeOrigin.x * cosPhi - eyeOrigin.y * sinPhi;
                eyeOrigin.y = (eyeOrigin.y * cosPhi + eyeOrigin.x * sinPhi);
                eyeOrigin.x = tmp;
            } else {
                tmp = eyeOrigin.x * cosPhi - eyeOrigin.z * sinPhi;
                eyeOrigin.z = (eyeOrigin.z * cosPhi + eyeOrigin.x * sinPhi);
                eyeOrigin.x = tmp;
            }
        }

#if 0
        Vec3f rayDirection = createDirection(eyeOrigin, x, y);
        // Head Tilt Support
        Vec3f headTarget( sinPhi * sinTheta, 
                          -cosPhi * sinTheta, 
                          rayDirection.z );
        float headTilt = getRdlCamera()->get(sHeadTiltMapKey);
        headTilt = (headTilt - 0.5) * M_PI;
        Mat3f tilt;
        tilt.setToRotation(headTarget, headTilt);
        eyeOrigin = scene_rdl2::math::transformVector(tilt, eyeOrigin);
#endif

        flipXVector(eyeOrigin);
        flipYVector(eyeOrigin);

        const Vec3f cameraOrigin = transformPoint(ct2render, eyeOrigin);
        *dstRay = mcrt_common::RayDifferential(cameraOrigin,
                                               transformVector(ct2render, createDirection(eyeOrigin, x, y)),
                                               cameraOrigin, transformVector(ct2render, createDirection(eyeOrigin,
                                                                                                        x + 1.0f, y)),
                                               cameraOrigin, transformVector(ct2render, createDirection(eyeOrigin,
                                                                                                        x, y + 1.0f)),
                                               getNear(), getFar(), time, 0);
    }

    StereoView DomeMaster3DCamera::getStereoViewImpl() const
    {
        return static_cast<StereoView>(getRdlCamera()->get(sStereoViewKey));
    }

    Vec3f DomeMaster3DCamera::createDirection(const Vec3f& camOrigin, float x, float y) const
    {
        float sinPhi, cosPhi, sinTheta, cosTheta;
        computePhiAndTheta(x, y,
                           sinPhi, cosPhi, 
                           sinTheta, cosTheta);
        Vec3f rayDirection;
        if (mZenithMode) {
            rayDirection = Vec3f(sinPhi * sinTheta, - cosPhi * sinTheta, -cosTheta);
        } else {
            rayDirection = Vec3f(sinPhi * cosTheta, sinTheta, - cosPhi * cosTheta);
        }

        applyParallax(rayDirection);

        flipXVector(rayDirection);
        flipYVector(rayDirection);

        rayDirection -= camOrigin;
        rayDirection.normalize();

        return rayDirection;
    }

} // namespace pbr
} // namespace moonray


