// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

//
//

#pragma once

#include <scene_rdl2/common/math/Color.h>
#include <scene_rdl2/common/math/MathUtil.h>

namespace moonray {
namespace shading {

static const scene_rdl2::math::Color sPbrValidityValidColor     = scene_rdl2::math::Color(0.0f, 1.0f, 0.0f); // red
static const scene_rdl2::math::Color sPbrValidityInvalidColor   = scene_rdl2::math::Color(1.0f, 0.0f, 0.0f); // green

static const float sPbrValidityAlbedoMin = 0.031896f;
static const float sPbrValidityAlbedoMax = 0.871367f;

/// PBR Validity values for metals
// Values between the below two limits are to transition from invalid to valid
static const float sPbrValidityConductorInvalid = 0.456411f; // Below this value is invalid (70% sRGB)
static const float sPbrValidityConductorValid = 0.533276f; // Above this value is valid (75% sRGB)

/// PBR Validity values for dielectrics
// Values in IOR are typically in 1.3 to ~(1.7 to 1.8) range.
static const float sPbrValidityDielectricValidLowBegin = 1.3f;
static const float sPbrValidityDielectricValidLowEnd = 1.33f; // 0.02 linear
static const float sPbrValidityDielectricValidHighBegin = 1.576f; // 0.05 linear
static const float sPbrValidityDielectricValidHighEnd = 1.788f; // 0.08 linear (only some gem stones)

/// Computes PBR Validity for albedo values used in Lambert lobe and
/// Dipole/Normalized Diffusion BSSRDF
scene_rdl2::math::Color
computeAlbedoPbrValidity(const scene_rdl2::math::Color& albedo);

} // namespace shading
} // namespace moonray

