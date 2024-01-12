// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include <scene_rdl2/common/platform/HybridUniformData.hh>


//----------------------------------------------------------------------------

#define ELLIPTICAL_FALLOFF_MEMBERS                  \
    HUD_MEMBER(float, mRoundness);                  \
    HUD_MEMBER(float, mElliptical);                 \
    HUD_MEMBER(OldFalloffCurve, mOldFalloffCurve);  \
    HUD_MEMBER(float, mInnerW);                     \
    HUD_MEMBER(float, mInnerH);                     \
    HUD_MEMBER(float, mOuterW);                     \
    HUD_MEMBER(float, mOuterH)

#define ELLIPTICAL_FALLOFF_VALIDATION                   \
    HUD_BEGIN_VALIDATION(EllipticalFalloff);            \
    HUD_VALIDATE(EllipticalFalloff, mRoundness);        \
    HUD_VALIDATE(EllipticalFalloff, mElliptical);       \
    HUD_VALIDATE(EllipticalFalloff, mOldFalloffCurve);  \
    HUD_VALIDATE(EllipticalFalloff, mInnerW);           \
    HUD_VALIDATE(EllipticalFalloff, mInnerH);           \
    HUD_VALIDATE(EllipticalFalloff, mOuterW);           \
    HUD_VALIDATE(EllipticalFalloff, mOuterH);           \
    HUD_END_VALIDATION


//----------------------------------------------------------------------------


