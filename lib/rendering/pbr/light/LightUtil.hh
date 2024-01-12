// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include <scene_rdl2/common/platform/HybridUniformData.hh>


//----------------------------------------------------------------------------

#define PLANE_MEMBERS                                       \
    HUD_MEMBER(HUD_NAMESPACE(scene_rdl2::math, Vec3f), mN); \
    HUD_MEMBER(float, mOffset)

#define PLANE_VALIDATION            \
    HUD_BEGIN_VALIDATION(Plane);    \
    HUD_VALIDATE(Plane, mN);        \
    HUD_VALIDATE(Plane, mOffset);   \
    HUD_END_VALIDATION


#define FALLOFF_CURVE_TYPE_ENUM     \
    FALLOFF_CURVE_TYPE_NONE,        \
    FALLOFF_CURVE_TYPE_LINEAR,      \
    FALLOFF_CURVE_TYPE_EASEIN,      \
    FALLOFF_CURVE_TYPE_EASEOUT,     \
    FALLOFF_CURVE_TYPE_EASEINOUT,   \
    FALLOFF_CURVE_TYPE_NUM_TYPES

#define FALLOFF_CURVE_TYPE_ENUM_VALIDATION                                                                      \
    MNRY_ASSERT_REQUIRE(FALLOFF_CURVE_TYPE_NONE == (FalloffCurveType)ispc::FALLOFF_CURVE_TYPE_NONE);             \
    MNRY_ASSERT_REQUIRE(FALLOFF_CURVE_TYPE_NUM_TYPES == (FalloffCurveType)ispc::FALLOFF_CURVE_TYPE_NUM_TYPES)


#define FALLOFF_CURVE_MEMBERS               \
    HUD_MEMBER(FalloffCurveType, mType)

#define FALLOFF_CURVE_VALIDATION            \
    HUD_BEGIN_VALIDATION(FalloffCurve);     \
    HUD_VALIDATE(FalloffCurve, mType);      \
    HUD_END_VALIDATION


#define OLD_FALLOFF_CURVE_TYPE_ENUM                                 \
    OLD_FALLOFF_CURVE_TYPE_NONE,       /* Exponent is ignored */    \
    OLD_FALLOFF_CURVE_TYPE_EASEOUT,                                 \
    OLD_FALLOFF_CURVE_TYPE_GAUSSIAN,                                \
    OLD_FALLOFF_CURVE_TYPE_LINEAR,                                  \
    OLD_FALLOFF_CURVE_TYPE_SQUARED,    /* Exponent is ignored */    \
    OLD_FALLOFF_CURVE_TYPE_NATURAL,                                 \
    OLD_FALLOFF_CURVE_TYPE_NUM_TYPES

#define OLD_FALLOFF_CURVE_TYPE_ENUM_VALIDATION                                                                          \
    MNRY_ASSERT_REQUIRE(OLD_FALLOFF_CURVE_TYPE_NONE == (OldFalloffCurveType)ispc::OLD_FALLOFF_CURVE_TYPE_NONE);          \
#define OLD_FALLOFF_CURVE_TYPE_ENUM_VALIDATION                                                                          \
    MNRY_ASSERT_REQUIRE(OLD_FALLOFF_CURVE_TYPE_NUM_TYPES == (OldFalloffCurveType)ispc::OLD_FALLOFF_CURVE_TYPE_NUM_TYPES)


#define OLD_FALLOFF_CURVE_MEMBERS                   \
    HUD_MEMBER(OldFalloffCurveType, mType);         \
    HUD_MEMBER(float, mExp);                        \
    /* cached vars related to natural falloff */    \
    HUD_MEMBER(float, mG1);                         \
    HUD_MEMBER(float, mInvH0)

#define OLD_FALLOFF_CURVE_VALIDATION            \
    HUD_BEGIN_VALIDATION(OldFalloffCurve);      \
    HUD_VALIDATE(OldFalloffCurve, mType);       \
    HUD_VALIDATE(OldFalloffCurve, mExp);        \
    HUD_VALIDATE(OldFalloffCurve, mG1);         \
    HUD_VALIDATE(OldFalloffCurve, mInvH0);      \
HUD_END_VALIDATION


//----------------------------------------------------------------------------


