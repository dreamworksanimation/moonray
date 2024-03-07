// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#include "Light.h"
#include <moonray/rendering/pbr/core/Util.h>
#include <moonray/rendering/pbr/lightfilter/LightFilter.h>
#include <moonray/rendering/pbr/light/Light_ispc_stubs.h>

#include <moonray/common/mcrt_macros/moonray_static_check.h>

#include <scene_rdl2/scene/rdl2/rdl2.h>
#include <scene_rdl2/common/except/exceptions.h>

using namespace scene_rdl2;
using namespace scene_rdl2::math;

namespace moonray {
namespace pbr {

const Mat4f Light::sIdentity = Mat4f( 1.0f, 0.0f, 0.0f, 0.0f,
                                      0.0f, 1.0f, 0.0f, 0.0f,
                                      0.0f, 0.0f, 1.0f, 0.0f,
                                      0.0f, 0.0f, 0.0f, 1.0f );

/// Used to transform from space where light is pointing down the +Z axis in
/// local space, to pointing down the -Z axis in render space, just like the
/// camera.
const Mat4f Light::sRotateX180 = Mat4f( 1.0f, 0.0f, 0.0f, 0.0f,
                                        0.0f,-1.0f, 0.0f, 0.0f,
                                        0.0f, 0.0f,-1.0f, 0.0f,
                                        0.0f, 0.0f, 0.0f, 1.0f );

//----------------------------------------------------------------------------

HUD_VALIDATOR(Light);

Light::Light(const scene_rdl2::rdl2::Light* rdlLight) :
    mCanIlluminateFn(nullptr),
    mIntersectFn(nullptr),
    mSampleFn(nullptr),
    mEvalFn(nullptr),
    mRdlLight(rdlLight),
    mOn(false),
    mIsVisibleInCamera(false),
    mIsOpaqueInAlpha(true),
    mMb(LIGHT_MB_NONE),
    mArea(1.0f),
    mInvArea(1.0f),
    mPosition{zero, zero},
    mOrientation{zero, zero},
    mDirection(zero),
    mRadiance(zero),
    mDistribution(nullptr),
    mDistributionMapping(Distribution2D::Mapping::NONE),
    mLabelId(-1),
    mIsRayTerminator(false),
    mTextureFilter(TEXTURE_FILTER_NEAREST),
    mClearRadius(0.f),
    mClearRadiusFalloffDistance(0.f),
    mClearRadiusInterpolation(0)
{
    MNRY_ASSERT(rdlLight);
}

Vec3f
Light::lerpPosition(float time) const
{
    if (!(mMb & LIGHT_MB_TRANSLATION)) return mPosition[0];

    return math::lerp(mPosition[0], mPosition[1], time);
}

Vec3f
Light::slerpDirection(float time) const
{
    if (!(mMb & LIGHT_MB_ROTATION)) return mDirection;

    const math::Quaternion3f q = math::slerp(mOrientation[0], mOrientation[1], time);
    // row 2 of Mat3(q) - see Mat3.h
    return math::Vec3f(2.0f * (q.i * q.k + q.r * q.j),
                       2.0f * (q.j * q.k - q.r * q.i),
                       1.0 - 2.0f * (q.i * q.i + q.j *q.j));
}

BBox3f
Light::getBounds() const
{
    MNRY_ASSERT(!"We shouldn't get here. If isBounded() returns true,"
                " the light needs getBounds() defined.");
    return BBox3f(scene_rdl2::util::empty);
}


void
Light::updateVisibilityFlags()
{
    // visible in camera flag
    VisibleInCamera visible = static_cast<VisibleInCamera>(
                    mRdlLight->get(scene_rdl2::rdl2::Light::sVisibleInCameraKey));
    if (visible == VISIBLE_IN_CAMERA_USE_GLOBAL) {
        const scene_rdl2::rdl2::SceneContext *rdlSceneContext =
                mRdlLight->getSceneClass().getSceneContext();
        mIsVisibleInCamera = rdlSceneContext->getSceneVariables().get(
                scene_rdl2::rdl2::SceneVariables::sLightsVisibleInCameraKey);
    } else {
        mIsVisibleInCamera = (visible == VISIBLE_IN_CAMERA_ON);
    }

    // lobe visibility mask
    mVisibilityMask = mRdlLight->getVisibilityMask();
}

bool
Light::updateImageMap(Distribution2D::Mapping distributionMapping)
{
    // Re-creating the image distribution below is expensive, so let's make
    // sure we really need this
    if (!mRdlLight->hasChanged(scene_rdl2::rdl2::Light::sTextureKey)
        && (mDistributionMapping == distributionMapping)
        && !mRdlLight->hasChanged(scene_rdl2::rdl2::Light::sGammaKey)
        && !mRdlLight->hasChanged(scene_rdl2::rdl2::Light::sContrastKey)
        && !mRdlLight->hasChanged(scene_rdl2::rdl2::Light::sSaturationKey)
        && !mRdlLight->hasChanged(scene_rdl2::rdl2::Light::sGainKey)
        && !mRdlLight->hasChanged(scene_rdl2::rdl2::Light::sOffsetKey)
        && !mRdlLight->hasChanged(scene_rdl2::rdl2::Light::sTemperatureKey)
        && !mRdlLight->hasChanged(scene_rdl2::rdl2::Light::sTextureRotationKey)
        && !mRdlLight->hasChanged(scene_rdl2::rdl2::Light::sTextureTranslationKey)
        && !mRdlLight->hasChanged(scene_rdl2::rdl2::Light::sTextureCoverageKey)
        && !mRdlLight->hasChanged(scene_rdl2::rdl2::Light::sTextureRepsUKey)
        && !mRdlLight->hasChanged(scene_rdl2::rdl2::Light::sTextureRepsVKey)
        && !mRdlLight->hasChanged(scene_rdl2::rdl2::Light::sTextureMirrorUKey)
        && !mRdlLight->hasChanged(scene_rdl2::rdl2::Light::sTextureMirrorVKey)
        && !mRdlLight->hasChanged(scene_rdl2::rdl2::Light::sTextureBorderColorKey)) {
        return true;
    }

    mDistributionMapping = distributionMapping;

    delete mDistribution;
    mDistribution = nullptr;

    const std::string & mapFilename = mRdlLight->get(scene_rdl2::rdl2::Light::sTextureKey);
    if (mapFilename.empty()) {
        return true;
    }

    // TODO: Keep and share image distributions in a scene-global resource manager
    try {
        mDistribution = new ImageDistribution(mapFilename, mDistributionMapping,
            mRdlLight->get(scene_rdl2::rdl2::Light::sGammaKey), 
            mRdlLight->get(scene_rdl2::rdl2::Light::sContrastKey), 
            mRdlLight->get(scene_rdl2::rdl2::Light::sSaturationKey),
            mRdlLight->get(scene_rdl2::rdl2::Light::sGainKey),
            mRdlLight->get(scene_rdl2::rdl2::Light::sOffsetKey),
            mRdlLight->get(scene_rdl2::rdl2::Light::sTemperatureKey),
            mRdlLight->get(scene_rdl2::rdl2::Light::sTextureRotationKey),
            mRdlLight->get(scene_rdl2::rdl2::Light::sTextureTranslationKey),
            mRdlLight->get(scene_rdl2::rdl2::Light::sTextureCoverageKey),
            mRdlLight->get(scene_rdl2::rdl2::Light::sTextureRepsUKey),
            mRdlLight->get(scene_rdl2::rdl2::Light::sTextureRepsVKey),
            mRdlLight->get(scene_rdl2::rdl2::Light::sTextureMirrorUKey),
            mRdlLight->get(scene_rdl2::rdl2::Light::sTextureMirrorVKey),
            mRdlLight->get(scene_rdl2::rdl2::Light::sTextureBorderColorKey));
    } catch (scene_rdl2::except::KeyError &e) {
        mRdlLight->error(e.what());
    }

    if (mDistribution == nullptr  ||  !mDistribution->isValid()) {
        delete mDistribution;
        mDistribution = nullptr;
        mRadiance = math::sBlack;
        mOn = false;
        return false;
    }

    return true;
}

void
Light::setLabelId(int32_t labelId)
{
    mLabelId = labelId;
}

void
Light::updatePresenceShadows()
{
    PresenceShadows shadows = static_cast<PresenceShadows>(
                    mRdlLight->get(scene_rdl2::rdl2::Light::sPresenceShadowsKey));
    if (shadows == PRESENCE_SHADOWS_USE_GLOBAL) {
        const scene_rdl2::rdl2::SceneContext *rdlSceneContext =
                mRdlLight->getSceneClass().getSceneContext();
        mPresenceShadows = rdlSceneContext->getSceneVariables().get(
                scene_rdl2::rdl2::SceneVariables::sEnablePresenceShadows);
    } else {
        mPresenceShadows = (shadows == PRESENCE_SHADOWS_ON);
    }
}

void
Light::updateRayTermination()
{
    mIsRayTerminator = getRdlLight()->get(scene_rdl2::rdl2::Light::sRayTerminationKey);
}

void
Light::updateTextureFilter()
{
    mTextureFilter = static_cast<TextureFilterType>(getRdlLight()->get(scene_rdl2::rdl2::Light::sTextureFilterKey));
}

void
Light::updateMaxShadowDistance()
{
    mMaxShadowDistance = getRdlLight()->get(scene_rdl2::rdl2::Light::sMaxShadowDistanceKey);
}

//----------------------------------------------------------------------------

HUD_VALIDATOR(LocalParamLight);

LocalParamLight::LocalParamLight(const scene_rdl2::rdl2::Light* rdlLight) :
    Light(rdlLight)
{
    updateParamAndTransforms(Mat4f(one), Mat4f(one), 1.0f, 1.0f);
}

bool
LocalParamLight::updateTransforms(const Mat4f &local2Render, int ti)
{
    // update transforms at time index (ti)
    // ti == 0 is at normalized rayTime = 0.f
    // ti == 1 is at normalized rayTime = 1.f;
    MNRY_ASSERT(getRdlLight());

    if (!extractUniformScale(local2Render, &mLocal2RenderScale[ti])) {
        getRdlLight()->warn("Non-uniform scale on light - setting to off.");
        mOn = false;
        return false;
    }
    mRender2LocalScale[ti] = 1.0f / math::max(sEpsilon, mLocal2RenderScale[ti]);

    Mat4f local2RenderRot = local2Render * Mat4f::scale(Vec4f(mRender2LocalScale[ti]));
    if (!isOne(math::abs(local2RenderRot.det()), 0.01f)) {
        getRdlLight()->warn("Invalid transform on light - setting to off.");
        mOn = false;
        return false;
    }

    mLocal2Render[ti] = Xform3f(Mat3f(asVec3(local2Render.row0()),
                                      asVec3(local2Render.row1()),
                                      asVec3(local2Render.row2())),
                                      asVec3(local2Render.row3()));
    // WARNING: this can be inaccurate, may lead to float precision jitter
    mRender2Local[ti] = mLocal2Render[ti].inverse();

    // WARNING: Includes translation, ok because only used with transformVector
    mLocal2RenderRot[ti] = Xform3f(Mat3f(asVec3(local2RenderRot.row0()),
                                         asVec3(local2RenderRot.row1()),
                                         asVec3(local2RenderRot.row2())),
                                         asVec3(local2RenderRot.row3()));

    // WARNING: Includes translation, ok because only used with transformVector
    // and also getXformRender2Local which uses the translation.
    mRender2LocalRot[ti] = mRender2Local[ti] * Xform3f::scale(Vec3f(mLocal2RenderScale[ti]));

    return true;
}

bool
LocalParamLight::updateParamAndTransforms(const Mat4f &local2Render0,
                                          const Mat4f &local2Render1,
                                          float halfWidth,
                                          float halfHeight)
{
    if (!updateTransforms(local2Render0, /* ti = */ 0)) return false;

    mPosition[0] = mLocal2Render[0].row3();
    mOrientation[0] = normalize(math::Quaternion3f(mLocal2RenderRot[0].row0(),
        mLocal2RenderRot[0].row1(), mLocal2RenderRot[0].row2()));
    mDirection = mLocal2RenderRot[0].row2();

    // setup mMb
    mMb = LIGHT_MB_NONE;
    const scene_rdl2::rdl2::SceneVariables &vars =
        getRdlLight()->getSceneClass().getSceneContext()->getSceneVariables();
    const bool mb  = vars.get(scene_rdl2::rdl2::SceneVariables::sEnableMotionBlur) &&
        getRdlLight()->get(scene_rdl2::rdl2::Light::sMbKey) &&
        !isEqual(local2Render0, local2Render1);

    if (mb) {
        if (!updateTransforms(local2Render1, /* ti = */ 1)) return false;

        mPosition[1] = mLocal2Render[1].row3();
        mOrientation[1] = normalize(math::Quaternion3f(mLocal2RenderRot[1].row0(),
            mLocal2RenderRot[1].row1(), mLocal2RenderRot[1].row2()));
        if (dot(mOrientation[0], mOrientation[1]) < 0) {
            mOrientation[1] *= -1.f;
        }

        // set specific mb bits
        if (!isEqual(mPosition[0], mPosition[1]))                   mMb |= LIGHT_MB_TRANSLATION;
        if (!isEqual(mLocal2RenderRot[0].l, mLocal2RenderRot[1].l)) mMb |= LIGHT_MB_ROTATION;
        if (!isEqual(mLocal2RenderScale[0], mLocal2RenderScale[1])) mMb |= LIGHT_MB_SCALE;
    }

    // Uv mapping
    getScaleOffset(-halfWidth,  halfWidth,  1.0f, 0.0f, &mUvScale.x, &mUvOffset.x);
    getScaleOffset(-halfHeight, halfHeight, 1.0f, 0.0f, &mUvScale.y, &mUvOffset.y);

    return true;
}

//-----------------------------------------------------------------------------

// point transformations
Vec3f
LocalParamLight::slerpPointLocal2Render(const Vec3f &p, float time) const
{
    // transformPoint(Xform3f x, Vec3f p) is  p * x.l + x.t
    const float s = (mMb & LIGHT_MB_SCALE) ?
        lerp(mLocal2RenderScale[0], mLocal2RenderScale[1], time) :
        mLocal2RenderScale[0];
    const Mat3f r = (mMb & LIGHT_MB_ROTATION) ?
        Mat3f(slerp(mOrientation[0], mOrientation[1], time)) :
        mLocal2RenderRot[0].l;
    const Vec3f t = (mMb & LIGHT_MB_TRANSLATION) ?
        lerp(mPosition[0], mPosition[1], time) :
        mPosition[0];
    
    return (p * s) * r + t;
}

Vec3f
LocalParamLight::slerpPointRender2Local(const Vec3f &p, float time) const
{
    // inverse of Local2Render is (p + t) * r * s
    // where t = -position, r = render2localRot.l, s = render2localScale
    const Vec3f t = (mMb & LIGHT_MB_TRANSLATION) ?
        -1.f * lerp(mPosition[0], mPosition[1], time) :
        -1.f * mPosition[0];
    const Mat3f r = (mMb & LIGHT_MB_ROTATION)?
        Mat3f(slerp(mOrientation[0], mOrientation[1], time)).transposed() : // inverse is transpose
        mRender2LocalRot[0].l;
    const float s = (mMb & LIGHT_MB_SCALE) ?
        rcp(lerp(mLocal2RenderScale[0], mLocal2RenderScale[1], time)) :
        mRender2LocalScale[0];

    return ((p + t) * r) * s;
}

Vec3f
LocalParamLight::slerpVectorLocal2Render(const Vec3f &v, float time) const
{
    // transformVector is v * s * r
    // where r is local2renderRot.l and s is the local2renderScale
    const Mat3f r = (mMb & LIGHT_MB_ROTATION) ?
        Mat3f(slerp(mOrientation[0], mOrientation[1], time)) :
        mLocal2RenderRot[0].l;
    const float s = (mMb & LIGHT_MB_SCALE) ?
        lerp(mLocal2RenderScale[0], mLocal2RenderScale[1], time) :
        mLocal2RenderScale[0];

    return (v * s) * r;
}

// vector transformations
Vec3f
LocalParamLight::slerpVectorRender2Local(const Vec3f &v, float time) const
{
    // inverse of Local2Render is v * r * s
    // where r = render2localRot.l, s = render2localScale
    const Mat3f r = (mMb & LIGHT_MB_ROTATION) ?
        Mat3f(slerp(mOrientation[0], mOrientation[1], time)).transposed() : // inverse is transpose
        mRender2LocalRot[0].l;
    const float s = (mMb & LIGHT_MB_SCALE) ?
        rcp(lerp(mLocal2RenderScale[0], mLocal2RenderScale[1], time)) :
        mRender2LocalScale[0];

    return (v * r) * s;
}

Vec3f
LocalParamLight::slerpVectorLocal2RenderRot(const Vec3f &v, float time) const
{
    // transformVector(Xform3f x, Vec3f v) is v * x.l
    const Mat3f r = (mMb & LIGHT_MB_ROTATION) ?
        Mat3f(slerp(mOrientation[0], mOrientation[1], time)) :
        mLocal2RenderRot[0].l;

    return v * r;
}

Vec3f
LocalParamLight::slerpVectorRender2LocalRot(const Vec3f &v, float time) const
{
    // inverse of Local2RenderRot is v * r
    // where r = render2localRot
    const Mat3f r = (mMb & LIGHT_MB_ROTATION) ?
        Mat3f(slerp(mOrientation[0], mOrientation[1], time)).transposed() : // inverse is transpose
        mRender2LocalRot[0].l;

    return v * r;
}

// normal transforms
Vec3f
LocalParamLight::slerpNormalLocal2Render(const Vec3f &n, float time) const
{
    // note: transformNormal in the math library simply does M * n,
    // assuming it is passed the inverse of the desired transform.
    // xformNormalLocal2Render means to transform a normal from local
    // to render space.

    // transformNormal is r * s * n
    // where r is render2localRot and s is the render2localScale
    const Mat3f r = (mMb & LIGHT_MB_ROTATION) ?
        Mat3f(slerp(mOrientation[0], mOrientation[1], time)).transposed() : // inverse is transpose
        mRender2LocalRot[0].l;
    const float s = (mMb & LIGHT_MB_SCALE) ?
        rcp(lerp(mLocal2RenderScale[0], mLocal2RenderScale[1], time)) :
        mRender2LocalScale[0];

    return r * (s * n);
}

Vec3f
LocalParamLight::slerpNormalLocal2RenderRot(const Vec3f &n, float time) const
{
    // note: transformNormal in the math library simply does M * n,
    // assuming it is passed the inverse of the desired transform.
    // xformNormalLocal2RenderRot means to transform a normal from local
    // to render space orientation.

    // transformNormal is r * n
    // where r is the render2localRot
    const Mat3f r = (mMb & LIGHT_MB_ROTATION) ?
        Mat3f(slerp(mOrientation[0], mOrientation[1], time)).transposed() : // inverse is transpose
        mRender2LocalRot[0].l;

    return r * n;
}

Vec3f
LocalParamLight::slerpNormalRender2LocalRot(const Vec3f &n, float time) const
{
    // note: transformNormal in the math library simply does M * n,
    // assuming it is passed the inverse of the desired transform.
    // xformNormalRender2LocalRot means to transform a normal from render
    // to local space orientation.

    // transformNormal is r * n
    // where r is the local2renderRot
    const Mat3f r = (mMb & LIGHT_MB_ROTATION)?
        Mat3f(slerp(mOrientation[0], mOrientation[1], time)) :
        mLocal2RenderRot[0].l;

    return r * n;
}

// uniform scale transforms
float
LocalParamLight::lerpLocal2RenderScale(float s, float time) const
{
    if (!(mMb & LIGHT_MB_SCALE)) return mLocal2RenderScale[0] * s;

    return lerp(mLocal2RenderScale[0], mLocal2RenderScale[1], time) * s;
}

float
LocalParamLight::lerpRender2LocalScale(float s, float time) const
{
    if (!(mMb & LIGHT_MB_SCALE)) return mRender2LocalScale[0] * s;

    return rcp(lerp(mLocal2RenderScale[0], mLocal2RenderScale[1], time)) * s;
}

//----------------------------------------------------------------------------

} // namespace pbr
} // namespace moonray

