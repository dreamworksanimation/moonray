// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include <scene_rdl2/common/platform/HybridUniformData.hh>

// values for Light::mMb
#define LIGHT_MB_NONE        (0)      // light has no blurred motion
#define LIGHT_MB_TRANSLATION (1 << 0) // light has blurred translational change
#define LIGHT_MB_ROTATION    (1 << 1) // light has blurred rotational change
#define LIGHT_MB_SCALE       (1 << 2) // light has blurred scale change


#define LIGHT_SIDEDNESS_ENUM        \
    LIGHT_SIDEDNESS_REGULAR = 0,    \
    LIGHT_SIDEDNESS_REVERSE,        \
    LIGHT_SIDEDNESS_2_SIDED,        \
    LIGHT_SIDEDNESS_NUM_TYPES

#define LIGHT_SIDEDNESS_ENUM_VALIDATION                                                 \
    MNRY_ASSERT_REQUIRE(LIGHT_SIDEDNESS_REGULAR   == ispc::LIGHT_SIDEDNESS_REGULAR);     \
    MNRY_ASSERT_REQUIRE(LIGHT_SIDEDNESS_NUM_TYPES == ispc::LIGHT_SIDEDNESS__NUM_TYPES)

enum LightSidednessType
{
    LIGHT_SIDEDNESS_ENUM
};


//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#define LIGHT_MEMBERS                                                       \
    HUD_VIRTUAL_BASE_CLASS();                                               \
                                                                            \
    /* ISPC virtual light intersection and sampling interface */            \
    HUD_ISPC_FNPTR(LightCanIlluminateFn, mCanIlluminateFn);                 \
    HUD_ISPC_FNPTR(LightIntersectFn, mIntersectFn);                         \
    HUD_ISPC_FNPTR(LightOccludedFn, mOccludedFn);                           \
    HUD_ISPC_FNPTR(LightSampleFn, mSampleFn);                               \
    HUD_ISPC_FNPTR(LightEvalFn, mEvalFn);                                   \
    HUD_ISPC_FNPTR(LightGetThetaOFn, mGetThetaOFn);                         \
    HUD_ISPC_FNPTR(LightGetThetaEFn, mGetThetaEFn);                         \
                                                                            \
    /* Backpointer to the rdl2 light */                                     \
    HUD_CPP_PTR(const scene_rdl2::rdl2::Light *, mRdlLight);                \
                                                                            \
    /* Set to false if the "on" attribute was set to false or there was */  \
    /* a problem updating the light. */                                     \
    HUD_MEMBER(int8_t, mOn);                                                \
                                                                            \
    /* Is this light visible from the camera and if so, */                  \
    /* is it opaque in the alpha channel ? */                               \
    HUD_MEMBER(int8_t, mIsVisibleInCamera);                                 \
    HUD_MEMBER(int8_t, mIsOpaqueInAlpha);                                   \
                                                                            \
    /* Is this light visible in the diffuse, glossy, or mirror lobes? */    \
    HUD_MEMBER(int32_t, mVisibilityMask);                                   \
                                                                            \
    /* Should light motion be taken into account?  This is true  */         \
    /* if and only if ALL the following conditions are met: */              \
    /* 1: global motion-blur is on (scene variables setting) */             \
    /* 2: light is enabled for mb (light attribute setting) */              \
    /* 3: light has distinct transforms at the motion steps */              \
    /* The light sub-class is responsible for setting this member */        \
    /* when it sets up its transforms (update). */                          \
    /* The LIGHT_MB defines are used to describe the type */                \
    /* of light motion: translation, rotational, and/or scale */            \
    HUD_MEMBER(int8_t, mMb);                                                \
                                                                            \
    /* Render space surface area and inverse surface area */                \
    /* For Distant and Env (non-local lights), this is the solid angle, */  \
    /* which turns out to be useful for the LightTester unittest */         \
    HUD_MEMBER(float, mArea);                                               \
    HUD_MEMBER(float, mInvArea);                                            \
                                                                            \
    /* Render space position and orientation */                             \
    /* Placed here so getPosition and getDirection don't have to be */      \
    /* virtual. The derived classes must fill these in in their update */   \
    /* functions. */                                                        \
    /* [0] = value at rayTime = 0 */                                        \
    /* [1] = value at rayTime = 1 */                                        \
    /* mDirection is at rayTime = 0 */                                      \
    HUD_ARRAY(HUD_NAMESPACE(scene_rdl2::math, Vec3f), mPosition, 2);                    \
    HUD_ARRAY(HUD_NAMESPACE(scene_rdl2::math, Quaternion3f), mOrientation, 2);          \
    HUD_MEMBER(HUD_NAMESPACE(scene_rdl2::math, Vec3f), mDirection);                     \
                                                                            \
    HUD_MEMBER(HUD_NAMESPACE(scene_rdl2::math, Color), mRadiance);                      \
                                                                            \
    /* TODO: this should be a shared resource */                            \
    HUD_PTR(ImageDistribution *, mDistribution);                            \
    HUD_CPP_MEMBER(Distribution2D::Mapping, mDistributionMapping, 4);       \
                                                                            \
    /* Label used in light aov expressions */                               \
    HUD_MEMBER(int32_t, mLabelId);                                          \
                                                                            \
    HUD_MEMBER(int8_t, mPresenceShadows);                                   \
    HUD_MEMBER(uint32_t, mHash);                                            \
    HUD_MEMBER(int8_t, mIsRayTerminator);                                   \
    HUD_MEMBER(TextureFilterType, mTextureFilter);                          \
    HUD_MEMBER(LightSidednessType, mSidedness);                             \
    HUD_MEMBER(float, mClearRadius);                                        \
    HUD_MEMBER(float, mClearRadiusFalloffDistance);                         \
    HUD_MEMBER(float, mClearRadiusInterpolation);                           \
    HUD_MEMBER(float, mMaxShadowDistance)


#define LIGHT_VALIDATION                                \
    HUD_BEGIN_VALIDATION(Light);                        \
    HUD_VALIDATE(Light, mCanIlluminateFn);              \
    HUD_VALIDATE(Light, mIntersectFn);                  \
    HUD_VALIDATE(Light, mSampleFn);                     \
    HUD_VALIDATE(Light, mEvalFn);                       \
    HUD_VALIDATE(Light, mRdlLight);                     \
    HUD_VALIDATE(Light, mOn);                           \
    HUD_VALIDATE(Light, mIsVisibleInCamera);            \
    HUD_VALIDATE(Light, mIsOpaqueInAlpha);              \
    HUD_VALIDATE(Light, mMb);                           \
    HUD_VALIDATE(Light, mArea);                         \
    HUD_VALIDATE(Light, mInvArea);                      \
    HUD_VALIDATE(Light, mPosition);                     \
    HUD_VALIDATE(Light, mOrientation);                  \
    HUD_VALIDATE(Light, mDirection);                    \
    HUD_VALIDATE(Light, mRadiance);                     \
    HUD_VALIDATE(Light, mDistribution);                 \
    HUD_VALIDATE(Light, mDistributionMapping);          \
    HUD_VALIDATE(Light, mLabelId);                      \
    HUD_VALIDATE(Light, mPresenceShadows);              \
    HUD_VALIDATE(Light, mHash);                         \
    HUD_VALIDATE(Light, mIsRayTerminator);              \
    HUD_VALIDATE(Light, mTextureFilter);                \
    HUD_VALIDATE(Light, mSidedness);                    \
    HUD_VALIDATE(Light, mClearRadius);                  \
    HUD_VALIDATE(Light, mClearRadiusFalloffDistance);   \
    HUD_VALIDATE(Light, mClearRadiusInterpolation);     \
    HUD_VALIDATE(Light, mMaxShadowDistance);            \
    HUD_END_VALIDATION


//----------------------------------------------------------------------------

#define LOCAL_PARAM_LIGHT_MEMBERS                                               \
    /* For array members, */                                                    \
    /* [0] : rayTime == 0 */                                                    \
    /* [1] : rayTime == 1 */                                                    \
                                                                                \
    /* These are for transforming geometry in and out of local light space. */  \
    HUD_ARRAY(HUD_NAMESPACE(scene_rdl2::math, Xform3f), mLocal2Render, 2);      \
    HUD_ARRAY(HUD_NAMESPACE(scene_rdl2::math, Xform3f), mRender2Local, 2);      \
                                                                                \
    /* These contain no scale and are for transforming normalized directions */ \
    /* in and out of local light space. */                                      \
    HUD_ARRAY(HUD_NAMESPACE(scene_rdl2::math, Xform3f), mLocal2RenderRot, 2); /* mLocal2Render * mRender2LocalScale */ \
    HUD_ARRAY(HUD_NAMESPACE(scene_rdl2::math, Xform3f), mRender2LocalRot, 2); /* mRender2Local * mLocal2RenderScale */ \
                                                                                \
    /* These are for transforming values in and out of local light space. */    \
    HUD_ARRAY(float, mLocal2RenderScale, 2);                                    \
    HUD_ARRAY(float, mRender2LocalScale, 2);                                    \
                                                                                \
    /* Map a local position to a uv coordinate */                               \
    HUD_MEMBER(HUD_NAMESPACE(scene_rdl2::math, Vec2f), mUvScale);               \
    HUD_MEMBER(HUD_NAMESPACE(scene_rdl2::math, Vec2f), mUvOffset)



#define LOCAL_PARAM_LIGHT_VALIDATION                    \
    HUD_BEGIN_VALIDATION(LocalParamLight);              \
    HUD_VALIDATE(LocalParamLight, mLocal2Render);       \
    HUD_VALIDATE(LocalParamLight, mRender2Local);       \
    HUD_VALIDATE(LocalParamLight, mLocal2RenderRot);    \
    HUD_VALIDATE(LocalParamLight, mRender2LocalRot);    \
    HUD_VALIDATE(LocalParamLight, mLocal2RenderScale);  \
    HUD_VALIDATE(LocalParamLight, mRender2LocalScale);  \
    HUD_VALIDATE(LocalParamLight, mUvScale);            \
    HUD_VALIDATE(LocalParamLight, mUvOffset);           \
    HUD_END_VALIDATION

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#define CYLINDER_LIGHT_MEMBERS                      \
    /* Cylinder axis is in the local y-direction */ \
    HUD_MEMBER(float, mLocalRadius);                \
    HUD_MEMBER(float, mLocalHalfHeight);            \
    HUD_MEMBER(float, mActualRadius);               \
    HUD_MEMBER(float, mRcpActualRadius);            \
    HUD_MEMBER(float, mActualRadiusSquared);        \
    HUD_MEMBER(float, mActualHalfHeight)


#define CYLINDER_LIGHT_VALIDATION                       \
    HUD_BEGIN_VALIDATION(CylinderLight);                \
    HUD_VALIDATE(CylinderLight, mLocalRadius);          \
    HUD_VALIDATE(CylinderLight, mLocalHalfHeight);      \
    HUD_VALIDATE(CylinderLight, mActualRadius);         \
    HUD_VALIDATE(CylinderLight, mRcpActualRadius);      \
    HUD_VALIDATE(CylinderLight, mActualRadiusSquared);  \
    HUD_VALIDATE(CylinderLight, mActualHalfHeight);     \
    HUD_END_VALIDATION


//----------------------------------------------------------------------------

#define DISK_LIGHT_MEMBERS                                      \
    /* Additional pdf multiplier when using a distribution */   \
    HUD_MEMBER(float, mDistributionPdfScale);                   \
    HUD_MEMBER(float, mSpread);                                 \
    HUD_MEMBER(float, mTanSpreadTheta);                         \
    /* cached plane at rayTime = 0 */                           \
    HUD_MEMBER(Plane, mRenderPlane)


#define DISK_LIGHT_VALIDATION                       \
    HUD_BEGIN_VALIDATION(DiskLight);                \
    HUD_VALIDATE(DiskLight, mDistributionPdfScale); \
    HUD_VALIDATE(DiskLight, mSpread);               \
    HUD_VALIDATE(DiskLight, mTanSpreadTheta);       \
    HUD_VALIDATE(DiskLight, mRenderPlane);          \
    HUD_END_VALIDATION


//----------------------------------------------------------------------------

#define DISTANT_LIGHT_MEMBERS                                            \
    /* The Sun is typically 0.53 degrees */                              \
    HUD_MEMBER(float, mAngularExtent);                                   \
                                                                         \
    HUD_MEMBER(HUD_NAMESPACE(scene_rdl2::math, ReferenceFrame), mFrame); \
    HUD_MEMBER(float, mCullThreshold);                                   \
    HUD_MEMBER(float, mCosThetaMax);                                     \
    HUD_MEMBER(float, mVersineThetaMax)


#define DISTANT_LIGHT_VALIDATION                    \
    HUD_BEGIN_VALIDATION(DistantLight);             \
    HUD_VALIDATE(DistantLight, mAngularExtent);     \
    HUD_VALIDATE(DistantLight, mFrame);             \
    HUD_VALIDATE(DistantLight, mCullThreshold);     \
    HUD_VALIDATE(DistantLight, mCosThetaMax);       \
    HUD_VALIDATE(DistantLight, mVersineThetaMax);   \
    HUD_END_VALIDATION


//----------------------------------------------------------------------------

#define ENV_LIGHT_MEMBERS                                                \
    HUD_MEMBER(HUD_NAMESPACE(scene_rdl2::math, ReferenceFrame), mFrame); \
                                                                         \
    /* Are we upper-hemisphere-only ? */                                 \
    HUD_MEMBER(bool, mHemispherical);                                    \
    HUD_MEMBER(float, mLog2TexelAngle)


#define ENV_LIGHT_VALIDATION                \
    HUD_BEGIN_VALIDATION(EnvLight);         \
    HUD_VALIDATE(EnvLight, mFrame);         \
    HUD_VALIDATE(EnvLight, mHemispherical); \
    HUD_VALIDATE(EnvLight, mLog2TexelAngle);\
    HUD_END_VALIDATION


//----------------------------------------------------------------------------

#define MESH_LIGHT_MEMBERS                           \
    HUD_PTR(Node*, mBVHPtr);                         \
    HUD_MEMBER(uint32_t, mBVHSize);                  \
    HUD_MEMBER(uint32_t, mMbSteps);                  \
    HUD_MEMBER(uint32_t, mFaceCount);                \
    HUD_MEMBER(bool, mDeformationMb);                \
    HUD_PTR(HUD_NAMESPACE(scene_rdl2::math, Vec3f*), mVerticesPtr); \
    HUD_PTR(uint32_t*, mVertexOffsetPtr);            \
    HUD_PTR(uint32_t*, mFaceOffsetPtr);              \
    HUD_PTR(uint32_t*, mFaceVertexCountPtr);         \
    HUD_PTR(int*, mPrimIDToNodeIDPtr);               \
    HUD_MEMBER(RTCScene, mRtcScene);                 \
    HUD_PTR(const int64 *, mMapShader);              \
    HUD_ISPC_PAD(mPad, 388)


#define MESH_LIGHT_VALIDATION                     \
    HUD_BEGIN_VALIDATION(MeshLight);              \
    HUD_VALIDATE(MeshLight, mBVHPtr);             \
    HUD_VALIDATE(MeshLight, mBVHSize);            \
    HUD_VALIDATE(MeshLight, mMbSteps);            \
    HUD_VALIDATE(MeshLight, mDeformationMb);      \
    HUD_VALIDATE(MeshLight, mVerticesPtr);        \
    HUD_VALIDATE(MeshLight, mFaceVertexCountPtr); \
    HUD_VALIDATE(MeshLight, mPrimIDToNodeIDPtr);  \
    HUD_VALIDATE(MeshLight, mRtcScene);           \
    HUD_VALIDATE(MeshLight, mMapShader);          \
    HUD_END_VALIDATION


//----------------------------------------------------------------------------

#define RECT_LIGHT_MEMBERS                          \
    /* Defined along the local space xy plane */    \
    HUD_MEMBER(float, mWidth);                      \
    HUD_MEMBER(float, mHeight);                     \
    HUD_MEMBER(float, mHalfWidth);                  \
    HUD_MEMBER(float, mHalfHeight);                 \
    HUD_MEMBER(float, mSpread);                     \
    HUD_MEMBER(float, mTanSpreadTheta);             \
                                                    \
    /* Used in canIlluminate() */                   \
    HUD_MEMBER(Plane, mRenderPlane);                \
    HUD_MEMBER(HUD_NAMESPACE(scene_rdl2::math, Vec3f), mRenderCorners[4])



#define RECT_LIGHT_VALIDATION                       \
    HUD_BEGIN_VALIDATION(RectLight);                \
    HUD_VALIDATE(RectLight, mWidth);                \
    HUD_VALIDATE(RectLight, mHeight);               \
    HUD_VALIDATE(RectLight, mHalfWidth);            \
    HUD_VALIDATE(RectLight, mHalfHeight);           \
    HUD_VALIDATE(RectLight, mSpread);               \
    HUD_VALIDATE(RectLight, mTanSpreadTheta);       \
    HUD_VALIDATE(RectLight, mRenderPlane);          \
    HUD_VALIDATE(RectLight, mRenderCorners);        \
    HUD_END_VALIDATION


//----------------------------------------------------------------------------

#define SPHERE_LIGHT_MEMBERS        \
    HUD_MEMBER(float, mRadius);     \
    HUD_MEMBER(float, mRadiusSqr);  \
    HUD_MEMBER(float, mRcpRadius)


#define SPHERE_LIGHT_VALIDATION             \
    HUD_BEGIN_VALIDATION(SphereLight);      \
    HUD_VALIDATE(SphereLight, mRadius);     \
    HUD_VALIDATE(SphereLight, mRadiusSqr);  \
    HUD_VALIDATE(SphereLight, mRcpRadius);  \
    HUD_END_VALIDATION

//----------------------------------------------------------------------------


#define SPOT_LIGHT_MEMBERS                          \
    HUD_MEMBER(float, mLensRadius);                 \
    HUD_MEMBER(float, mRcpLensRadius);              \
    HUD_MEMBER(float, mLensRadiusY);                \
    HUD_MEMBER(float, mRcpLensRadiusY);             \
    HUD_MEMBER(float, mFocalRadius);                \
    HUD_MEMBER(float, mRcpFocalRadius);             \
    HUD_MEMBER(float, mFocalRadiusY);               \
    HUD_MEMBER(float, mRcpFocalRadiusY);            \
    HUD_MEMBER(float, mRcpAspectRatio);             \
    HUD_MEMBER(float, mFocalDistance);              \
    HUD_MEMBER(float, mFalloffGradient);            \
    HUD_MEMBER(float, mCrossOverDistance);          \
    HUD_MEMBER(float, mFocalPlanePdfConst);         \
    HUD_MEMBER(HUD_NAMESPACE(scene_rdl2::math, Vec3f), mRenderCorners[4]); \
    HUD_MEMBER(FalloffCurve, mFalloffCurve);        \
    HUD_MEMBER(float, mBlackLevel)


#define SPOT_LIGHT_VALIDATION                       \
    HUD_BEGIN_VALIDATION(SpotLight);                \
    HUD_VALIDATE(SpotLight, mLensRadius);           \
    HUD_VALIDATE(SpotLight, mRcpLensRadius);        \
    HUD_VALIDATE(SpotLight, mLensRadiusY);          \
    HUD_VALIDATE(SpotLight, mRcpLensRadiusY);       \
    HUD_VALIDATE(SpotLight, mFocalRadius);          \
    HUD_VALIDATE(SpotLight, mRcpFocalRadius);       \
    HUD_VALIDATE(SpotLight, mFocalRadiusY);         \
    HUD_VALIDATE(SpotLight, mRcpFocalRadiusY);      \
    HUD_VALIDATE(SpotLight, mRcpAspectRatio);       \
    HUD_VALIDATE(SpotLight, mFocalDistance);        \
    HUD_VALIDATE(SpotLight, mFalloffGradient);      \
    HUD_VALIDATE(SpotLight, mCrossOverDistance);    \
    HUD_VALIDATE(SpotLight, mFocalPlanePdfConst);   \
    HUD_VALIDATE(SpotLight, mRenderCorners);        \
    HUD_VALIDATE(SpotLight, mFalloffCurve);         \
    HUD_VALIDATE(SpotLight, mBlackLevel);           \
    HUD_END_VALIDATION


//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

