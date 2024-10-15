// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include <scene_rdl2/common/platform/HybridUniformData.hh>

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
//----------------------------------------------------------------------------


// values for RodLightFilter::mMb
#define LIGHTFILTER_MB_NONE        (0)      // filter has no blurred motion
#define LIGHTFILTER_MB_TRANSLATION (1 << 0) // filter has blurred translational change
#define LIGHTFILTER_MB_ROTATION    (1 << 1) // filter has blurred rotational change
#define LIGHTFILTER_MB_SCALE       (1 << 2) // filter has blurred scale change

#define SIZEOF_STD_VECTOR           24

#define LIGHT_FILTER_MEMBERS                                                  \
    HUD_VIRTUAL_BASE_CLASS();                                                 \
                                                                              \
    /* ISPC virtual light intersection and sampling interface */              \
    HUD_ISPC_FNPTR(LightFilterCanIlluminateFn, mCanIlluminateFn);             \
    HUD_ISPC_FNPTR(LightFilterEvalFn, mEvalFn);                               \
                                                                              \
    /* Backpointer to the rdl2 light filter*/                                 \
    HUD_CPP_PTR(const scene_rdl2::rdl2::LightFilter *, mRdlLightFilter)

#define LIGHT_FILTER_VALIDATION                                               \
    HUD_BEGIN_VALIDATION(LightFilter);                                        \
    HUD_VALIDATE(LightFilter, mCanIlluminateFn);                              \
    HUD_VALIDATE(LightFilter, mEvalFn);                                       \
    HUD_VALIDATE(LightFilter, mRdlLightFilter);                               \
    HUD_END_VALIDATION

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

enum ColorRampInterpolationType
{
    NONE,
    LINEAR,
    EXPONENTIAL_UP,
    EXPONENTIAL_DOWN,
    SMOOTH,
    CATMULL_ROM,
    MONOTONE_CUBIC
};

enum ColorRampMode
{
    RADIAL,       // Use 3D distance between shaded point and filter/light position
    DIRECTIONAL   // Use distance along Z axis between shaded point and filter/light position
};

enum ColorRampWrapMode
{
    EXTEND,       // f(z) = f(0) for z < 0
    MIRROR        // f(z) = f(-z)
};

#define COLOR_RAMP_LIGHT_FILTER_MEMBERS                                                      \
    HUD_ARRAY(HUD_NAMESPACE(scene_rdl2::math, Mat4f), mXform, 2);                            \
    HUD_MEMBER(bool, mUseXform);                                                             \
    HUD_MEMBER(int, mPad);                                                                   \
    HUD_MEMBER(HUD_NAMESPACE(moonray::shading, ColorRampControl), mColorRamp);               \
    HUD_MEMBER(float, mIntensity);                                                           \
    HUD_MEMBER(float, mDensity);                                                             \
    HUD_MEMBER(int, mMode);                                                                  \
    HUD_MEMBER(int, mWrapMode)

#define COLOR_RAMP_LIGHT_FILTER_VALIDATION                                    \
    HUD_BEGIN_VALIDATION(ColorRampLightFilter);                               \
    HUD_VALIDATE(ColorRampLightFilter, mXform);                               \
    HUD_VALIDATE(ColorRampLightFilter, mUseXform);                            \
    HUD_VALIDATE(ColorRampLightFilter, mPad);                                 \
    HUD_VALIDATE(ColorRampLightFilter, mColorRamp);                           \
    HUD_VALIDATE(ColorRampLightFilter, mIntensity);                           \
    HUD_VALIDATE(ColorRampLightFilter, mDensity);                             \
    HUD_VALIDATE(ColorRampLightFilter, mMode);                                \
    HUD_VALIDATE(ColorRampLightFilter, mWrapMode);                            \
    HUD_END_VALIDATION

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

enum CombineMode
{
    MULTIPLY,
    MIN,
    MAX,
    ADD,
    SUBTRACT
};

#define COMBINE_LIGHT_FILTER_MEMBERS                                                      \
    HUD_CPP_MEMBER(std::vector<const LightFilter*>, mLightFiltersVec, SIZEOF_STD_VECTOR); \
    HUD_CPP_PTR(const LightFilter**, mLightFilters);                                      \
    HUD_MEMBER(int, mNumLightFilters);                                                    \
    HUD_MEMBER(int, mMode)

#define COMBINE_LIGHT_FILTER_VALIDATION                                       \
    HUD_BEGIN_VALIDATION(CombineLightFilter);                                 \
    HUD_VALIDATE(CombineLightFilter, mLightFiltersVec);                       \
    HUD_VALIDATE(CombineLightFilter, mLightFilters);                          \
    HUD_VALIDATE(CombineLightFilter, mNumLightFilters);                       \
    HUD_VALIDATE(CombineLightFilter, mMode);                                  \
    HUD_END_VALIDATION

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

enum CookieProjectorType
{
    PERSPECTIVE = 0,
    ORTHOGRAPHIC
};

enum CookieBlurType
{
    GAUSSIAN,
    CIRCULAR
};

enum CookieOutsideProjection
{
    BLACK,
    WHITE,
    DEFAULT
};

#define COOKIE_LIGHT_FILTER_MEMBERS                                           \
    HUD_PTR(const int64 *, mMapShader);                                       \
    HUD_ARRAY(HUD_NAMESPACE(scene_rdl2::math, Mat4f), mProjectorR2S, 2);      \
    HUD_ARRAY(HUD_NAMESPACE(scene_rdl2::math, Vec3f), mProjectorPos, 2);      \
    HUD_ARRAY(HUD_NAMESPACE(scene_rdl2::math, Vec3f), mProjectorDir, 2);      \
    HUD_MEMBER(float, mBlurNearDistance);                                     \
    HUD_MEMBER(float, mBlurMidpoint);                                         \
    HUD_MEMBER(float, mBlurFarDistance);                                      \
    HUD_MEMBER(float, mBlurNearValue);                                        \
    HUD_MEMBER(float, mBlurMidValue);                                         \
    HUD_MEMBER(float, mBlurFarValue);                                         \
    HUD_MEMBER(int, mBlurType);                                               \
    HUD_MEMBER(int, mOutsideProjection);                                      \
    HUD_MEMBER(float, mDensity);                                              \
    HUD_MEMBER(bool, mInvert)

#define COOKIE_LIGHT_FILTER_VALIDATION                                        \
    HUD_BEGIN_VALIDATION(CookieLightFilter);                                  \
    HUD_VALIDATE(CookieLightFilter, mMapShader);                              \
    HUD_VALIDATE(CookieLightFilter, mProjectorR2S);                           \
    HUD_VALIDATE(CookieLightFilter, mProjectorPos);                           \
    HUD_VALIDATE(CookieLightFilter, mProjectorDir);                           \
    HUD_VALIDATE(CookieLightFilter, mBlurNearDistance);                       \
    HUD_VALIDATE(CookieLightFilter, mBlurMidpoint);                           \
    HUD_VALIDATE(CookieLightFilter, mBlurFarDistance);                        \
    HUD_VALIDATE(CookieLightFilter, mBlurNearValue);                          \
    HUD_VALIDATE(CookieLightFilter, mBlurMidValue);                           \
    HUD_VALIDATE(CookieLightFilter, mBlurFarValue);                           \
    HUD_VALIDATE(CookieLightFilter, mBlurType);                               \
    HUD_VALIDATE(CookieLightFilter, mOutsideProjection);                      \
    HUD_VALIDATE(CookieLightFilter, mDensity);                                \
    HUD_VALIDATE(CookieLightFilter, mInvert);                                 \
    HUD_END_VALIDATION

#define COOKIE_LIGHT_FILTER_V2_MEMBERS                                        \
    HUD_ARRAY(HUD_NAMESPACE(scene_rdl2::math, Mat4f), mProjectorR2S, 2);      \
    HUD_ARRAY(HUD_NAMESPACE(scene_rdl2::math, Vec3f), mProjectorPos, 2);      \
    HUD_ARRAY(HUD_NAMESPACE(scene_rdl2::math, Vec3f), mProjectorDir, 2);      \
    HUD_MEMBER(float, mBlurNearDistance);                                     \
    HUD_MEMBER(float, mBlurMidpoint);                                         \
    HUD_MEMBER(float, mBlurFarDistance);                                      \
    HUD_MEMBER(float, mBlurNearValue);                                        \
    HUD_MEMBER(float, mBlurMidValue);                                         \
    HUD_MEMBER(float, mBlurFarValue);                                         \
    HUD_MEMBER(int, mBlurType);                                               \
    HUD_MEMBER(int, mOutsideProjection);                                      \
    HUD_MEMBER(float, mDensity);                                              \
    HUD_MEMBER(bool, mInvert);                                                \
    HUD_PTR(ImageDistribution *, mDistribution)

#define COOKIE_LIGHT_FILTER_V2_VALIDATION                                     \
    HUD_BEGIN_VALIDATION(CookieLightFilter_v2);                               \
    HUD_VALIDATE(CookieLightFilter_v2, mProjectorR2S);                        \
    HUD_VALIDATE(CookieLightFilter_v2, mProjectorPos);                        \
    HUD_VALIDATE(CookieLightFilter_v2, mProjectorDir);                        \
    HUD_VALIDATE(CookieLightFilter_v2, mBlurNearDistance);                    \
    HUD_VALIDATE(CookieLightFilter_v2, mBlurMidpoint);                        \
    HUD_VALIDATE(CookieLightFilter_v2, mBlurFarDistance);                     \
    HUD_VALIDATE(CookieLightFilter_v2, mBlurNearValue);                       \
    HUD_VALIDATE(CookieLightFilter_v2, mBlurMidValue);                        \
    HUD_VALIDATE(CookieLightFilter_v2, mBlurFarValue);                        \
    HUD_VALIDATE(CookieLightFilter_v2, mBlurType);                            \
    HUD_VALIDATE(CookieLightFilter_v2, mOutsideProjection);                   \
    HUD_VALIDATE(CookieLightFilter_v2, mDensity);                             \
    HUD_VALIDATE(CookieLightFilter_v2, mInvert);                              \
    HUD_VALIDATE(CookieLightFilter_v2, mDistribution);                        \
    HUD_END_VALIDATION

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

// mFalloffNear     Does the light fade in late?
// mFalloffFar      Does the light fade out early?
// mNearStart       Beginning of fade in
// mNearEnd         End of fade in
// mFarStart        Beginning of fade out
// mFarEnd          End of fade out

#define DECAY_LIGHT_FILTER_MEMBERS                                            \
    HUD_MEMBER(bool, mFalloffNear);                                           \
    HUD_MEMBER(bool, mFalloffFar);                                            \
    HUD_MEMBER(float, mNearStart);                                            \
    HUD_MEMBER(float, mNearEnd);                                              \
    HUD_MEMBER(float, mFarStart);                                             \
    HUD_MEMBER(float, mFarEnd)

#define DECAY_LIGHT_FILTER_VALIDATION                                         \
    HUD_BEGIN_VALIDATION(DecayLightFilter);                                   \
    HUD_VALIDATE(DecayLightFilter, mFalloffNear);                             \
    HUD_VALIDATE(DecayLightFilter, mFalloffFar);                              \
    HUD_VALIDATE(DecayLightFilter, mNearStart);                               \
    HUD_VALIDATE(DecayLightFilter, mNearEnd);                                 \
    HUD_VALIDATE(DecayLightFilter, mFarStart);                                \
    HUD_VALIDATE(DecayLightFilter, mFarEnd);                                  \
    HUD_END_VALIDATION

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

// mRender2Local0           Static general xform from render space to local space
// mRender2LocalRot0        Static-rotation xform from render space to local space
// mLocal2RenderScale       MB-scale xform from local to render space
// mRender2LocalScale0      Static-scale xform from render to local space
// mRender2LocalRotAndScale Precomputed rot * scale for isOutsideInfluence()
// mRadiusEdgeSum   Precomputed mEdge+mRadius for isOutsideInfluence()
// mPosition        MB-translation xform from local to render space
// mOrientation     MB-rotation xform from local to render space
// mBoxCorner       Precomputed postive corner of box
// mMb              Motion-blur status of (scale, rot, trans) components of Xform
// mWidth           Width of the blocker (X)
// mDepth           Depth of the blocker (Z)
// mHeight          Height of the blocker (Y)
// mRadius          Radius to add to inner box corners
// mEdge            Shell thickness from inner box to outer
// mEdgeInv         Precomputed inverse of edge for sdRoundBox()
// mColor           Filter color. Scales light within the volume.
//                  For each color channel, 0=full shadow 1=no shadow
// mIntensity       Scalar for multiplying the color. 0=black 1=color
// mDensity         Fades the filter effect. 0=no effect 1=full effect
// mInvert          Swap behavior of inside and outside the volume

#define ROD_LIGHT_FILTER_MEMBERS                                                          \
    /* These are for transforming values in and out of local light space. */              \
    HUD_MEMBER(HUD_NAMESPACE(scene_rdl2::math, Xform3f), mRender2Local0);                 \
    HUD_MEMBER(HUD_NAMESPACE(scene_rdl2::math, Xform3f), mRender2LocalRot0);              \
    HUD_ARRAY(HUD_NAMESPACE(scene_rdl2::math, Mat3f), mLocal2RenderScale, 2);             \
    HUD_MEMBER(HUD_NAMESPACE(scene_rdl2::math, Mat3f), mRender2LocalScale0);              \
    HUD_MEMBER(HUD_NAMESPACE(scene_rdl2::math, Mat3f), mRender2LocalRotAndScale);         \
    HUD_MEMBER(float, mRadiusEdgeSum);                                                    \
    HUD_ARRAY(HUD_NAMESPACE(scene_rdl2::math, Vec3f), mPosition, 2);                      \
    HUD_ARRAY(HUD_NAMESPACE(scene_rdl2::math, Quaternion3f), mOrientation, 2);            \
    HUD_MEMBER(HUD_NAMESPACE(scene_rdl2::math, Vec3f), mBoxCorner);                       \
    HUD_MEMBER(HUD_NAMESPACE(moonray::shading, FloatRampControl), mRamp);                 \
    HUD_MEMBER(int8_t, mMb);                                                              \
    HUD_MEMBER(float, mWidth);                                                            \
    HUD_MEMBER(float, mDepth);                                                            \
    HUD_MEMBER(float, mHeight);                                                           \
    HUD_MEMBER(float, mRadius);                                                           \
    HUD_MEMBER(float, mEdge);                                                             \
    HUD_MEMBER(float, mEdgeInv);                                                          \
    HUD_MEMBER(HUD_NAMESPACE(scene_rdl2::math, Color), mColor);                           \
    HUD_MEMBER(float, mIntensity);                                                        \
    HUD_MEMBER(float, mDensity);                                                          \
    HUD_MEMBER(bool, mInvert)

#define ROD_LIGHT_FILTER_VALIDATION                                           \
    HUD_BEGIN_VALIDATION(RodLightFilter);                                     \
    HUD_VALIDATE(RodLightFilter, mRender2Local0);                             \
    HUD_VALIDATE(RodLightFilter, mRender2LocalRot0);                          \
    HUD_VALIDATE(RodLightFilter, mLocal2RenderScale);                         \
    HUD_VALIDATE(RodLightFilter, mRender2LocalScale0);                        \
    HUD_VALIDATE(RodLightFilter, mRender2LocalRotAndScale);                   \
    HUD_VALIDATE(RodLightFilter, mRadiusEdgeSum);                             \
    HUD_VALIDATE(RodLightFilter, mPosition);                                  \
    HUD_VALIDATE(RodLightFilter, mOrientation);                               \
    HUD_VALIDATE(RodLightFilter, mBoxCorner);                                 \
    HUD_VALIDATE(RodLightFilter, mRamp);                                      \
    HUD_VALIDATE(RodLightFilter, mMb);                                        \
    HUD_VALIDATE(RodLightFilter, mWidth);                                     \
    HUD_VALIDATE(RodLightFilter, mDepth);                                     \
    HUD_VALIDATE(RodLightFilter, mHeight);                                    \
    HUD_VALIDATE(RodLightFilter, mRadius);                                    \
    HUD_VALIDATE(RodLightFilter, mEdge);                                      \
    HUD_VALIDATE(RodLightFilter, mEdgeInv);                                   \
    HUD_VALIDATE(RodLightFilter, mColor);                                     \
    HUD_VALIDATE(RodLightFilter, mIntensity);                                 \
    HUD_VALIDATE(RodLightFilter, mDensity);                                   \
    HUD_VALIDATE(RodLightFilter, mInvert);                                    \
    HUD_END_VALIDATION
   
   
//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

enum VdbLightFilterBlurType
{
    VDBLIGHTFILTERBLURTYPE_GAUSSIAN,
    VDBLIGHTFILTERBLURTYPE_SPHERICAL
};

#define VDB_LIGHT_FILTER_MEMBERS                                                 \
    HUD_CPP_PTR(moonray::shading::OpenVdbSampler*, mDensitySampler);             \
    HUD_MEMBER(intptr_t, mSampleFn);                                             \
    HUD_ARRAY(HUD_NAMESPACE(scene_rdl2::math, Mat4f), mVdbR2V, 2);               \
    HUD_ARRAY(HUD_NAMESPACE(scene_rdl2::math, Vec3f), mVdbPos, 2);               \
    HUD_MEMBER(HUD_NAMESPACE(moonray::shading, FloatRampControl), mDensityRamp); \
    HUD_MEMBER(HUD_NAMESPACE(scene_rdl2::math, Color), mColorTint);              \
    HUD_MEMBER(int, mVdbInterpolation);                                          \
    HUD_MEMBER(float, mBlurValue);                                               \
    HUD_MEMBER(int, mBlurType);                                                  \
    HUD_MEMBER(bool, mInvertDensity)

#define VDB_LIGHT_FILTER_VALIDATION                                        \
    HUD_BEGIN_VALIDATION(VdbLightFilter);                                  \
    HUD_VALIDATE(VdbLightFilter, mDensitySampler);                         \
    HUD_VALIDATE(VdbLightFilter, mSampleFn);                               \
    HUD_VALIDATE(VdbLightFilter, mVdbR2V);                                 \
    HUD_VALIDATE(VdbLightFilter, mVdbPos);                                 \
    HUD_VALIDATE(VdbLightFilter, mDensityRamp);                            \
    HUD_VALIDATE(VdbLightFilter, mColorTint);                              \
    HUD_VALIDATE(VdbLightFilter, mVdbInterpolation);                       \
    HUD_VALIDATE(VdbLightFilter, mBlurValue);                              \
    HUD_VALIDATE(VdbLightFilter, mBlurType);                               \
    HUD_VALIDATE(VdbLightFilter, mInvertDensity);                          \
    HUD_END_VALIDATION

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

enum BarnMode
{
    ANALYTIC,
    PHYSICAL
};

enum BarnDoorEdge
{
    BARNDOOR_EDGE_LEFT,
    BARNDOOR_EDGE_BOTTOM,
    BARNDOOR_EDGE_RIGHT,
    BARNDOOR_EDGE_TOP
};

// mRender2Local0        Static render-to-projector space xform
// mRender2LocalRot0     Static rotation component of render-to-projector space xform
// mLocal2RenderScale    MB scale component of render-to-projector space xform
// mRender2LocalScale0   Static scale component of render-to-projector space xform
// mPosition             MB translation component of render-to-projector space xform
// mOrientation          MB rotation component of render-to-projector space xform
// mMb                   Motion-blur status of node_xform
// mFocalDist            Distance from the projector of the flap opening
// mProjector2Screen     Projection matrix (perspective or orthographic)
// mMinCorner            Aperture min corner at the virtual focal length (1.0)
// mMaxCorner            Aperture max corner at the virtual focal length (1.0)
// mMode                 Analytic or Physical mode toggle
// mUseLightXform        Bind projector to the light?
// mPreBarnMode          Choice of behavior for region before the Barn Door
// mPreBarnDist          Distance from the projector where the Barn Door begins
// mRadius               Radius of the rounded corners in screen space
// mEdge                 Size of flap opening edge transition zone in screen space
// mReciprocalEdgeScales Per-side scale factors of edge size, reciprocal of
// mDensity              Fades the filter effect (0 means no effect, 1 means full)
// mInvert               Swap application of filter from inside to outside
// mColor                Color within the Barn Door lit region (default white)

#define BARN_DOOR_LIGHT_FILTER_MEMBERS                                        \
    HUD_MEMBER(HUD_NAMESPACE(scene_rdl2::math, Xform3f), mRender2Local0);     \
    HUD_MEMBER(HUD_NAMESPACE(scene_rdl2::math, Xform3f), mRender2LocalRot0);  \
    HUD_ARRAY(HUD_NAMESPACE(scene_rdl2::math, Mat3f), mLocal2RenderScale, 2); \
    HUD_MEMBER(HUD_NAMESPACE(scene_rdl2::math, Mat3f), mRender2LocalScale0);  \
    HUD_ARRAY(HUD_NAMESPACE(scene_rdl2::math, Vec3f), mPosition, 2);          \
    HUD_ARRAY(HUD_NAMESPACE(scene_rdl2::math, Quaternion3f), mOrientation, 2);\
    HUD_MEMBER(int8_t, mMb);                                                  \
    HUD_MEMBER(float, mFocalDist);                                            \
    HUD_MEMBER(HUD_NAMESPACE(scene_rdl2::math, Mat4f), mProjector2Screen);    \
    HUD_MEMBER(HUD_NAMESPACE(scene_rdl2::math, Vec2f), mMinCorner);           \
    HUD_MEMBER(HUD_NAMESPACE(scene_rdl2::math, Vec2f), mMaxCorner);           \
    HUD_MEMBER(int, mMode);                                                   \
    HUD_MEMBER(bool, mUseLightXform);                                         \
    HUD_MEMBER(int, mPreBarnMode);                                            \
    HUD_MEMBER(float, mPreBarnDist);                                          \
    HUD_MEMBER(float, mRadius);                                               \
    HUD_MEMBER(float, mEdge);                                                 \
    HUD_ARRAY(float, mReciprocalEdgeScales, 4);                               \
    HUD_MEMBER(float, mDensity);                                              \
    HUD_MEMBER(bool, mInvert);                                                \
    HUD_MEMBER(HUD_NAMESPACE(scene_rdl2::math, Color), mColor)

#define BARN_DOOR_LIGHT_FILTER_VALIDATION                                     \
    HUD_BEGIN_VALIDATION(BarnDoorLightFilter);                                \
    HUD_VALIDATE(BarnDoorLightFilter, mRender2Local0);                        \
    HUD_VALIDATE(BarnDoorLightFilter, mRender2LocalRot0);                     \
    HUD_VALIDATE(BarnDoorLightFilter, mLocal2RenderScale);                    \
    HUD_VALIDATE(BarnDoorLightFilter, mRender2LocalScale0);                   \
    HUD_VALIDATE(BarnDoorLightFilter, mPosition);                             \
    HUD_VALIDATE(BarnDoorLightFilter, mOrientation);                          \
    HUD_VALIDATE(BarnDoorLightFilter, mMb);                                   \
    HUD_VALIDATE(BarnDoorLightFilter, mFocalDist);                            \
    HUD_VALIDATE(BarnDoorLightFilter, mProjector2Screen);                     \
    HUD_VALIDATE(BarnDoorLightFilter, mMinCorner);                            \
    HUD_VALIDATE(BarnDoorLightFilter, mMaxCorner);                            \
    HUD_VALIDATE(BarnDoorLightFilter, mMode);                                 \
    HUD_VALIDATE(BarnDoorLightFilter, mUseLightXform);                        \
    HUD_VALIDATE(BarnDoorLightFilter, mPreBarnMode);                          \
    HUD_VALIDATE(BarnDoorLightFilter, mPreBarnDist);                          \
    HUD_VALIDATE(BarnDoorLightFilter, mRadius);                               \
    HUD_VALIDATE(BarnDoorLightFilter, mEdge);                                 \
    HUD_VALIDATE(BarnDoorLightFilter, mReciprocalEdgeScales);                 \
    HUD_VALIDATE(BarnDoorLightFilter, mDensity);                              \
    HUD_VALIDATE(BarnDoorLightFilter, mInvert);                               \
    HUD_VALIDATE(BarnDoorLightFilter, mColor);                                \
    HUD_END_VALIDATION

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

// mRadianceMod   Modifies light radiance

#define INTENSITY_LIGHT_FILTER_MEMBERS                                        \
    HUD_MEMBER(HUD_NAMESPACE(scene_rdl2::math, Color), mRadianceMod)

#define INTENSITY_LIGHT_FILTER_VALIDATION                                     \
    HUD_BEGIN_VALIDATION(IntensityLightFilter);                               \
    HUD_VALIDATE(IntensityLightFilter, mRadianceMod);                         \
    HUD_END_VALIDATION

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#define LIGHT_FILTER_LIST_MEMBERS                                             \
    HUD_CPP_MEMBER(std::unique_ptr<const LightFilter* []>, mLightFilters, 8); \
    HUD_MEMBER(int32_t, mLightFilterCount);                                   \
    HUD_MEMBER(bool, mNeedsLightXform)


#define LIGHT_FILTER_LIST_VALIDATION                                          \
    HUD_BEGIN_VALIDATION(LightFilterList);                                    \
    HUD_VALIDATE(LightFilterList, mLightFilters);                             \
    HUD_VALIDATE(LightFilterList, mLightFilterCount);                         \
    HUD_VALIDATE(LightFilterList, mNeedsLightXform);                          \
    HUD_END_VALIDATION

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

