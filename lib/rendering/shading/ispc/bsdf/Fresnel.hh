// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#pragma once
#include <scene_rdl2/common/platform/HybridUniformData.hh>
#include <scene_rdl2/common/platform/HybridVaryingData.hh>

//----------------------------------------------------------------------------

#define SCHLICK_FRESNEL_MEMBERS                         \
    HVD_MEMBER(HVD_NAMESPACE(scene_rdl2::math, Color), mSpec);      \
    HVD_MEMBER(float, mFactor);                         \
    HVD_MEMBER(float, mNeta)

#define ONE_MINUS_ROUGH_SCHLICK_FRESNEL_MEMBERS         \
    HVD_REAL_PTR(const SchlickFresnel *, mTopFresnel);  \
    HVD_MEMBER(float, mSpecRoughness)

#define DIELECTRIC_FRESNEL_MEMBERS                      \
    HVD_MEMBER(float, mEtaI);                           \
    HVD_MEMBER(float, mEtaT)

#define LAYERED_DIELECTRIC_FRESNEL_MEMBERS              \
    HVD_MEMBER(float, mNumLayers)

#define CONDUCTOR_FRESNEL_MEMBERS                       \
    HVD_MEMBER(HVD_NAMESPACE(scene_rdl2::math, Color), mEta);       \
    HVD_MEMBER(HVD_NAMESPACE(scene_rdl2::math, Color), mAbsorption)

#define ONE_MINUS_FRESNEL_MEMBERS                       \
    HVD_REAL_PTR(const Fresnel *, mTopFresnel)

#define ONE_MINUS_VELVET_FRESNEL_MEMBERS                \
    HVD_MEMBER(float, mRoughness)

#define ONE_MINUS_ROUGH_FRESNEL_MEMBERS                             \
    HVD_REAL_PTR(const Fresnel *, mTopFresnel);                     \
    HVD_MEMBER(HVD_NAMESPACE(scene_rdl2::math, Color), mOneMinusFresnelAt0);    \
    HVD_MEMBER(float, mSpecRoughness);                              \
    HVD_MEMBER(float, mInterpolator)

#define MULTIPLE_TRANSMISSION_FRESNEL_MEMBERS           \
    HUD_MEMBER(HUD_UNIFORM int, mNumFresnels);          \
    HVD_REAL_PTR_ARRAY(const Fresnel *, mFresnels, 16)

//----------------------------------------------------------------------------


