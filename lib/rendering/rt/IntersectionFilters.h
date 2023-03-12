// Copyright 2023 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <embree3/rtcore.h>

namespace moonray {
namespace rt {

void vdbVolumeIntervalFilter(const RTCFilterFunctionNArguments* args);

void manifoldVolumeIntervalFilter(const RTCFilterFunctionNArguments* args);

void bssrdfTraceSetIntersectionFilter(const RTCFilterFunctionNArguments* args);

void backFaceCullingFilter(const RTCFilterFunctionNArguments* args);

void skipOcclusionFilter(const RTCFilterFunctionNArguments* args);

} // namespace rt
} // namespace moonray

