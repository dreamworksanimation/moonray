// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file MipSelector.h

#pragma once

namespace moonray {
namespace shading {

float computeMipSelector(float dsdx, float dtdx, float dsdy, float dtdy);

} // namespace moonray
} // namespace shading


