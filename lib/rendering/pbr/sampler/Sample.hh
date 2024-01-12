// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <scene_rdl2/common/platform/HybridUniformData.hh>

// pixelX in filter extents (centered about 0.5)
// pixelY in filter extents (centered about 0.5)
// lensU in (-1, 1)
// lensV in (-1, 1)
// time in [0, 1)
#define SAMPLE_MEMBERS         \
    HUD_MEMBER(float, pixelX); \
    HUD_MEMBER(float, pixelY); \
    HUD_MEMBER(float, lensU);  \
    HUD_MEMBER(float, lensV);  \
    HUD_MEMBER(float, time)

#define SAMPLE_VALIDATION         \
    HUD_BEGIN_VALIDATION(Sample); \
    HUD_VALIDATE(Sample, pixelX); \
    HUD_VALIDATE(Sample, pixelY); \
    HUD_VALIDATE(Sample, lensU);  \
    HUD_VALIDATE(Sample, lensV);  \
    HUD_VALIDATE(Sample, time);   \
    HUD_END_VALIDATION

// u in [0, 1)
// v in [0, 1)
// w in [0, 1)
#define SAMPLE_3D_MEMBERS \
    HUD_MEMBER(float, u); \
    HUD_MEMBER(float, v); \
    HUD_MEMBER(float, w)

#define SAMPLE_3D_VALIDATION        \
    HUD_BEGIN_VALIDATION(Sample3D); \
    HUD_VALIDATE(Sample3D, u);      \
    HUD_VALIDATE(Sample3D, v);      \
    HUD_VALIDATE(Sample3D, w);      \
    HUD_END_VALIDATION

// u in [0, 1)
// v in [0, 1)
#define SAMPLE_2D_MEMBERS \
    HUD_MEMBER(float, u); \
    HUD_MEMBER(float, v)

#define SAMPLE_2D_VALIDATION        \
    HUD_BEGIN_VALIDATION(Sample2D); \
    HUD_VALIDATE(Sample2D, u);      \
    HUD_VALIDATE(Sample2D, v);      \
    HUD_END_VALIDATION

