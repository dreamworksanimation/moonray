// Copyright 2023-2024 DreamWorks Animation LLC
// SPDX-License-Identifier: Apache-2.0

/// @file AovLabels.h
#pragma once

#include "AovLabels.hh"

#include <vector>

namespace moonray {
namespace shading {

inline int
aovEncodeLabels(int label,
                const int materialLabelId,
                const int lpeMaterialLabelId,
                const std::vector<int> &lobeLabelIds,
                const std::vector<int> &lpeLobeLabelIds)
{
    return SHADING_aovEncodeLabels(label, materialLabelId, lpeMaterialLabelId,
                                   lobeLabelIds.empty()? nullptr : &lobeLabelIds[0],
                                   lpeLobeLabelIds.empty()? nullptr : &lpeLobeLabelIds[0]);
}

inline bool
aovLabelIsTransformed(int label)
{
    return SHADING_aovLabelIsTransformed(label);
}

inline int
aovDecodeLabel(int encodedLabel)
{
    return SHADING_aovDecodeLabel(encodedLabel);
}

inline int
aovDecodeLpeLabel(int encodedLabel)
{
    return SHADING_aovDecodeLpeLabel(encodedLabel);
}

} // namespace shading
} // namespace moonray



